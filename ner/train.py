import argparse
import json

from tqdm import tqdm
import numpy as np
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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from seqeval.metrics import f1_score, recall_score, precision_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../data/menu/')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--train', type=str, default='train.txt')
    parser.add_argument('--test', type=str, default='test.txt')
    parser.add_argument('--val', type=str, default='valid.txt')
    parser.add_argument('--lr', type=float, default=10e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--eval_and_save_steps', type=int, default=500, help="Save checkpoint every X steps.")
    opt = parser.parse_args()

    # set seed
    random.seed(opt.seed)

    # Processing of data and creation of features - TRAIN/TEST/VAL

    train_examples = dataner_preprocess(opt.data_dir, opt.train)
    test_examples = dataner_preprocess(opt.data_dir, opt.test)
    val_examples = dataner_preprocess(opt.data_dir, opt.val)

    # Create Feature
    input_feats_train = convert_examples_to_feature(train_examples)
    input_feats_val = convert_examples_to_feature(val_examples)

    # Create train_loader and val_loader

    dataset_train = MenuDataset(input_feats_train)
    dataset_val = MenuDataset(input_feats_val)

    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=64,
                              shuffle=True)

    valid_loader = DataLoader(dataset=dataset_val,
                              batch_size=64,
                              shuffle=True)

    tag_to_idx = {"PAD": 0, "B-MENU": 1, "I-MENU": 2, "O": 3, "STOP_TAG": 4}

    if torch.cuda.is_available():
        logger.info("%s", torch.cuda.get_device_name(0))
        device = torch.device('cuda')

    device = torch.device('cpu')

    with temp_seed(opt.seed):
        model = BiLSTM_CRF(tagset_size=len(tag_to_idx),
                           embedding_dim=768,
                           hidden_dim=512,
                           pos2ix=pos2ix(train_examples),
                           pos_dim=64,
                           pos=True,
                           char=True,
                           attention=True)

        # create optimizer, scheduler, scaler, early_stopping
        optimizer = AdamW(params=model.parameters(),
                          lr=10e-3,
                          eps=10e-8,
                          weight_decay=0.01)

        num_training_steps_per_epoch = len(train_loader)
        num_training_steps = num_training_steps_per_epoch * opt.epochs

        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=num_training_steps)

        scaler = GradScaler()
        early_stopping = EarlyStopping(monitor='f1',
                                       patience=7,  # paper parameter
                                       verbose=True,
                                       mode='max')

        num_batches = len(train_loader)

        best_f1_score = - float('inf')
        path_checkpoint = '../model_checkpoint'
        for epoch in range(opt.epochs):
            train_loss = 0.
            local_best_eval_loss = float('inf')
            local_best_eval_f1 = 0

            optimizer.zero_grad()

            for step, (x, y) in enumerate(tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}")):
                model.train()
                x = to_device(x, device)
                y = to_device(y, device)

                loss = model.neg_log_likelihood(x, y)

                # backpropagation
                if device == 'cpu':
                    loss.backward()
                else:
                    scaler.scale(loss).backward()

                if device == 'cpu':
                    optimizer.step()
                else:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                    scaler.step(optimizer)

                    # Updates the scale for next iteration.
                    scaler.update()

                optimizer.zero_grad()
                scheduler.step()

                print(f"Epoch {epoch}, local_step: {step}, loss: {loss:.3f}, curr_lr: {scheduler.get_last_lr()[0]:.7f}")
                global_step = (len(train_loader) * epoch) + step

                if opt.eval_and_save_steps > 0 and global_step != 0 and global_step % opt.eval_and_save_steps == 0:
                    report = eval(model, valid_loader, device)

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
            report = eval(model, valid_loader, device)

            if local_best_eval_loss > report['loss']:
                local_best_eval_loss = report['loss']

            if local_best_eval_f1 < report['f1']:
                local_best_eval_f1 = report['f1']

            if report['f1'] > best_f1_score:
                best_f1_score = report['f1']
                best_f1_score_epoch = epoch  # Our Early stopping
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


def eval(model, valid_loader, eval_device):
    eval_loss = 0.
    n_batches = len(valid_loader)

    first_pred = True
    with torch.no_grad():
        for step, (x, y) in enumerate(tqdm(valid_loader, total=len(valid_loader), desc="fEvaluate")):
            model.eval()
            x = to_device(x, eval_device)
            y = to_device(y, eval_device)

            predictions = model(x)
            loss = model.neg_log_likelihood(x, y)

            if first_pred:
                preds = to_numpy(predictions)
                labels = to_numpy(y)
            else:
                preds = np.append(preds, to_numpy(predictions), axis=0)
                labels = np.append(labels, to_numpy(labels), axis=0)

            eval_loss += loss.item()

    eval_loss = eval_loss / n_batches
    list_labels = [[] for _ in range(labels.shape)[0]]
    list_preds = [[] for _ in range(preds.shape)[0]]

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] != 0:  # pad
                list_labels.append(labels[i][j])
                list_preds.append(preds[i][j])

    report = {
        'loss': eval_loss,
        'precision': precision_score(list_labels, list_preds),
        'f1': f1_score(list_labels, list_preds),
        'recall': recall_score(list_labels, list_preds),
        'class_report': classification_report(list_labels, list_preds)
    }

    print(report)
    return report


def to_numpy(x):
    if type(x) != list:  # torch.tensor
        x = x.detach().cpu().numpy()
    else:  # list of torch.tensor
        for i in range(len(x)):
            x[i] = x[i].detach().cpu().numpy()
    return x


def to_device(x, device):
    if type(x) != list:  # torch.tensor
        x = x.to(device)
    else:  # list of torch.tensor
        for i in range(len(x)):
            x[i] = x[i].to(device)
    return x


if __name__ == '__main__':
    main()
