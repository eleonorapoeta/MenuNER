import argparse
import torch
import os
import random

from torch.utils.data import DataLoader

from ner.preprocess import dataner_preprocess
from ner.dataset import MenuDataset
from model import BiLSTM_CRF
from util import pos2ix, convert_examples_to_feature


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../data/menu/')
    opt = parser.parse_args()

    # set seed
    # random.seed(opt.seed)

    # Processing of data and creation of features - TRAIN
    train_examples = dataner_preprocess(opt.data_dir)

    # Create Feature
    input_feats = convert_examples_to_feature(train_examples)

    d = MenuDataset(input_feats)
    menu_dataset = DataLoader(dataset=d,
                              batch_size=64,
                              shuffle=True)

    train_features, train_labels = next(iter(menu_dataset))
    tag_to_idx = {"PAD": 0, "B-MENU": 1, "I-MENU": 2, "O": 3, "STOP_TAG": 4}
    model = BiLSTM_CRF(tagset_size=len(tag_to_idx), embedding_dim=768,
                       hidden_dim=512,
                       pos2ix=pos2ix(train_examples),
                       pos=True)

    model(train_features)


if __name__ == '__main__':
    main()
