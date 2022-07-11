import argparse
import torch
import os
import random

from torch.utils.data import DataLoader

from ner.preprocess import dataner_preprocess
from ner.dataset import MenuDataset
from model import BiLSTM_CRF
from util import pos2ix


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../data/menu/')
    parser.add_argument('--alphabet_path', type=str, default='../data/alphabet.json')
    opt = parser.parse_args()

    # set seed
    #random.seed(opt.seed)

    # Processing of data and creation of features
    train_features, train_labels, train_examples = dataner_preprocess(opt.data_dir, opt.alphabet_path)

    dataset = DataLoader(dataset=MenuDataset(train_features, train_labels),
                         batch_size=64,
                         shuffle=True)

    train_features, train_labels = next(iter(dataset))
    tag_to_idx = {"PAD": 0, "B-MENU": 1, "I-MENU": 2, "O": 3, "STOP_TAG": 4}
    model = BiLSTM_CRF(tagset_size=len(tag_to_idx), embedding_dim=768,
                       hidden_dim=512,
                       pos2ix=pos2ix(train_examples),
                       pos=True)

    model(train_features)


if __name__ == '__main__':
    main()
