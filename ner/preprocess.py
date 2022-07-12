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


def dataner_preprocess(data_dir, train_name="train.txt", valid_name="valid.txt", test_name="test.txt"):
    examples = []
    bucket = []

    print(data_dir)

    train_path = os.path.join(data_dir, train_name)
    test_path = os.path.join(data_dir, test_name)
    valid_path = os.path.join(data_dir, valid_name)

    # Read of examples
    train_examples = read_examples_data(train_path)
    # test_examples = read_examples_data(test_path)
    # valid_examples = read_examples_data(valid_path)

    return train_examples
