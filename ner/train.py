import argparse
import os
import random

from torch.utils.data import DataLoader

from ner.preprocess import dataner_preprocess
from ner.dataset import MenuDataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../data/menu/')

    opt = parser.parse_args()

    # set seed
    random.seed(opt.seed)

    # Processing of data and creation of features
    train_features, train_labels = dataner_preprocess(opt.data_dir)

    # Creation of the dataset

    dataset = DataLoader(dataset=MenuDataset(train_features, train_labels),
                         batch_size=64,
                         shuffle=True)


if __name__ == '__main__':
    main()
