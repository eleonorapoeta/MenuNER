import argparse
import torch
import json
import logging
from ner.preprocess import dataner_preprocess
from ner.util import evaluation
from ner.util_preprocess import convert_examples_to_feature, pos2ix
from ner.dataset import MenuDataset
from model import BiLSTM_CRF
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../data/menu/')
    parser.add_argument('--model_path', type=str, default='../data/model_checkpoint')
    parser.add_argument('--test', type=str, default='test.txt')
    opt = parser.parse_args()

    tag_to_idx = {"PAD": 0, "MENU": 1, "O": 2}

    test_examples = dataner_preprocess(opt.data_dir, opt.test)
    input_feats_test = convert_examples_to_feature(test_examples)
    dataset_val = MenuDataset(input_feats_test)
    valid_loader = DataLoader(dataset=dataset_val,
                              batch_size=8,
                              shuffle=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = BiLSTM_CRF(tagset_size=len(tag_to_idx),
                       bert_checkpoint=opt.bert_checkpoint,
                       max_length=opt.max_length,
                       embedding_dim=768,
                       hidden_dim=512,
                       pos2ix=pos2ix(test_examples),
                       pos_dim=64,
                       pos=False,
                       char=False,
                       attention=True).to(device)

    checkpoint = torch.load(opt.model_path)
    model.load_state_dict(checkpoint)

    report = evaluation(model, valid_loader, device)

    logs = {
        'precision': report['precision'],
        'f1': report['f1_score'],
        'recall': report['recall']
    }

    logger.info(json.dumps(logs, indent=4))

    print(report)


if __name__ == '__main__':
    main()
