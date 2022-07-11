import argparse
import os
import random

from transformers import BertTokenizer
from util import read_examples_data
from util import pos2ix
from util import convert_examples_to_feature
from dataset import MenuDataset
from torch.utils.data import DataLoader
from model import Embedders
from model import BiLSTM_CRF


def dataner_preprocess(data_dir, alphabet_path, train_name="train.txt", valid_name="valid.txt", test_name="test.txt"):
    examples = []
    bucket = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    print(data_dir)

    train_path = os.path.join(data_dir, train_name)
    test_path = os.path.join(data_dir, test_name)
    valid_path = os.path.join(data_dir, valid_name)

    # Read of examples
    train_examples = read_examples_data(train_path, tokenizer)
    # test_examples = read_examples_data(test_path, tokenizer)
    # valid_examples = read_examples_data(valid_path, tokenizer)

    # Create Feature
    train_features, train_labels = convert_examples_to_feature(train_examples, tokenizer, alphabet_path)

    #
    # data = MenuDataset(train_features, train_labels)
    # dataset = DataLoader(dataset=data, batch_size=64, shuffle=True)
    #
    # train_features, train_labels = next(iter(dataset))
    # tag_to_idx = {"PAD": 0, "B-MENU": 1, "I-MENU": 2, "O": 3, "STOP_TAG": 4}
    # model = BiLSTM_CRF(tagset_size=len(tag_to_idx), embedding_dim=768,
    #                    hidden_dim=512,
    #                    pos2ix=pos2ix(train_examples),
    #                    pos=True)
    #
    # model(train_features)
    #
    return train_features, train_labels, train_examples
