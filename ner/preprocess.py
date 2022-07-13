import os
from transformers import BertTokenizer
from util_preprocess import convert_examples_to_feature
from util_preprocess import read_examples_data


def dataner_preprocess(data_dir, train_name="train.txt", valid_name="valid.txt", test_name="test.txt"):
    print(data_dir)

    train_path = os.path.join(data_dir, train_name)
    test_path = os.path.join(data_dir, test_name)
    valid_path = os.path.join(data_dir, valid_name)

    # Read of examples
    train_examples = read_examples_data(train_path)
    # test_examples = read_examples_data(test_path)
    # valid_examples = read_examples_data(valid_path)

    return train_examples
