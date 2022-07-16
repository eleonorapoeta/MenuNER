import argparse
import json

from tqdm import tqdm
import logging
import torch.cuda
from torch.utils.data import DataLoader
import random
from ner.preprocess import dataner_preprocess
from ner.dataset import MenuDataset
from model import BiLSTM_CRF
from util_preprocess import pos2ix, convert_examples_to_feature
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets.metric import temp_seed
from torch.cuda.amp import GradScaler
from ner.util import to_device, evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../data/menu/')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--train', type=str, default='train.txt')
    parser.add_argument('--val', type=str, default='valid.txt')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bert_checkpoint', type=str, default='../bert_checkpoint', help='Folder in which there is '
                                                                                          'BERT domain')
    parser.add_argument('--pos', type=bool, default=True)
    parser.add_argument('--pos_embedding_dim', type=int, default=64)
    parser.add_argument('--char', type=bool, default=True)
    parser.add_argument('--char_embedding_dim', type=int, default=25)
    parser.add_argument('--attention', type=bool, default=True)
    parser.add_argument('--eval_and_save_steps', type=int, default=500)
    opt = parser.parse_args()

    # set seed
    random.seed(opt.seed)

    # Processing of data and creation of features - TRAIN/VAL

    train_examples = dataner_preprocess(opt.data_dir, opt.train)
    val_examples = dataner_preprocess(opt.data_dir, opt.val)

    # Create Feature
    input_feats_train = convert_examples_to_feature(train_examples)
    input_feats_val = convert_examples_to_feature(val_examples)

    # Create train_loader and val_loader

    dataset_train = MenuDataset(input_feats_train)
    dataset_val = MenuDataset(input_feats_val)

    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=8,
                              shuffle=True)

    valid_loader = DataLoader(dataset=dataset_val,
                              batch_size=8,
                              shuffle=True)

    tag_to_idx = {"PAD": 0, "MENU": 1, "O": 2}

    if torch.cuda.is_available():
        logger.info("%s", torch.cuda.get_device_name(0))
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with temp_seed(opt.seed):
        model = BiLSTM_CRF(tagset_size=len(tag_to_idx),
                           bert_checkpoint=opt.bert_checkpoint,
                           max_length=opt.max_length,
                           embedding_dim=768,
                           hidden_dim=512,
                           pos2ix=pos2ix(train_examples),
                           pos_embedding_dim=opt.pos_embedding_dim,
                           pos=opt.pos,
                           char=opt.char,
                           char_embedding_dim=opt.char_embedding_dim,
                           attention=opt.attention).to(device)

        # create optimizer, scheduler, scaler, early_stopping
        optimizer = AdamW(params=model.parameters(),
                          lr=1e-3,
                          eps=1e-8,
                          weight_decay=0.01)

        num_training_steps_per_epoch = len(train_loader)
        num_training_steps = num_training_steps_per_epoch * opt.epochs

        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=num_training_steps)

        scaler = GradScaler()

        best_f1_score = - float('inf')
        best_f1_score_epoch = 0
        path_checkpoint = '/content/drive/MyDrive/DNLP_Polito/Model/ELEONORA/model.pt'
        for epoch in range(opt.epochs):
            train_loss = 0.
            local_best_eval_loss = float('inf')
            local_best_eval_f1 = 0

            optimizer.zero_grad()

            for step, (x, y) in enumerate(tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}")):
                model.train()
                x = to_device(x, device)
                y = to_device(y, device)

                mask = torch.sign(torch.abs(x[1])).to(torch.uint8)
                logits, predictions = model(x)
                log_likelihood = model.crf_module(logits, y, mask=mask, reduction='mean')

                loss = log_likelihood * (-1)

                # backpropagation
                if device == 'cpu':
                    loss.backward()
                else:
                    scaler.scale(loss).backward()

                if device == 'cpu':
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                optimizer.zero_grad()
                scheduler.step()

                print(f"Epoch {epoch}, local_step: {step}, loss: {loss:.3f}, curr_lr: {scheduler.get_last_lr()[0]:.7f}")
                global_step = (len(train_loader) * epoch) + step

                if opt.eval_and_save_steps > 0 and global_step != 0 and global_step % opt.eval_and_save_steps == 0:
                    report = evaluation(model, valid_loader, device, epoch)

                    if local_best_eval_loss > report['loss']:
                        local_best_eval_loss = report['loss']

                    if local_best_eval_f1 < report['f1']:
                        local_best_eval_f1 = report['f1']

                    if report['f1'] > best_f1_score:
                        best_f1_score = report['f1']
                        with open(path_checkpoint, 'wb') as f:
                            checkpoint = model.state_dict()
                            torch.save(checkpoint, f)

                train_loss += loss.item()

            # Evaluate in each epoch
            report = evaluation(model, valid_loader, device, epoch)

            if local_best_eval_loss > report['loss']:
                local_best_eval_loss = report['loss']

            if local_best_eval_f1 < report['f1']:
                local_best_eval_f1 = report['f1']

            if report['f1'] > best_f1_score:
                best_f1_score = report['f1']
                best_f1_score_epoch = epoch
                with open(path_checkpoint, 'wb') as f:
                    checkpoint = model.state_dict()
                    torch.save(checkpoint, f)

            logs = {
                'epoch': epoch,
                'local_step': step + 1,
                'epoch_step': len(train_loader),
                'local_best_eval_loss': local_best_eval_loss,
                'local_best_eval_f1': local_best_eval_f1,
                'best_eval_f1': best_f1_score,
            }

            logger.info(json.dumps(logs, indent=4))
            if (epoch - best_f1_score_epoch) >= 7:  # Our Early stopping
                print(f"Early stopped Training at epoch:{epoch}")
                break


if __name__ == '__main__':
    main()
