import argparse
import torch
import json
import logging
from ner.preprocess import dataner_preprocess
from ner.util import evaluation
from ner.util_preprocess import convert_examples_to_feature
from ner.dataset import MenuDataset
from model import BiLSTM_CRF
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../data/menu/')
    parser.add_argument('--pos2ix', type=str, default='../data/pos2ix_dict.json')
    parser.add_argument('--model_path', type=str, default='../data/model_checkpoint.pt')
    parser.add_argument('--bert_checkpoint', type=str, default='../bert_checkpoint')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--test', type=str, default='test.txt')
    parser.add_argument('--pos', type=bool, default=True)
    parser.add_argument('--pos_embedding_dim', type=int, default=64)
    parser.add_argument('--char', type=bool, default=True)
    parser.add_argument('--char_embedding_dim', type=int, default=25)
    parser.add_argument('--attention', type=bool, default=True)
    opt = parser.parse_args()

    tag_to_idx = {"PAD": 0, "MENU": 1, "O": 2}

    test_examples = dataner_preprocess(opt.data_dir, opt.test)
    input_feats_test = convert_examples_to_feature(test_examples)
    dataset_test = MenuDataset(input_feats_test)
    test_loader = DataLoader(dataset=dataset_test,
                              batch_size=8,
                              shuffle=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with open(opt.pos2ix, 'r') as f:
        pos2ix = json.load(f)

    model = BiLSTM_CRF(tagset_size=len(tag_to_idx),
                       bert_checkpoint=opt.bert_checkpoint,
                       max_length=opt.max_length,
                       embedding_dim=768,
                       hidden_dim=512,
                       pos2ix=pos2ix,
                       pos=opt.pos,
                       char=opt.char,
                       pos_embedding_dim=opt.pos_embedding_dim,
                       char_embedding_dim=opt.char_embedding_dim,
                       attention=opt.attention).to(device)

    checkpoint = torch.load(opt.model_path)
    model.load_state_dict(checkpoint)

    result = evaluation(model, test_loader, device)

    logs = {
        'precision': result['precision'],
        'f1': result['f1'],
        'recall': result['recall']
    }

    logger.info(json.dumps(logs, indent=4))

    print(result['classification_report'])


if __name__ == '__main__':
    main()
