import argparse
import os
import random

from tqdm import tqdm
from transformers import BertTokenizer
from util import read_examples_data
from util import convert_examples_to_feature


def dataner_preprocess(data_dir, train_name="train.txt", valid_name="valid.txt", test_name="test.txt"):
    examples = []
    bucket = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    print(data_dir)

    train_path = os.path.join(data_dir, train_name)
    test_path = os.path.join(data_dir, test_name)
    valid_path = os.path.join(data_dir, valid_name)

    # Read of examples
    train_examples = read_examples_data(train_path, tokenizer)
    test_examples = read_examples_data(test_path, tokenizer)
    valid_examples = read_examples_data(valid_path, tokenizer)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='../configs/config-bert.json')
    parser.add_argument('--data_dir', type=str, default='../data/menu/')
    parser.add_argument("--seed", default=5, type=int)
    # for BERT
    parser.add_argument("--bert_model_name_or_path", type=str, default='bert-base-uncased',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument('--bert_use_sub_label', action='store_true',
                        help="Set this flag to use sub label instead of using pad label for sub tokens.")
    opt = parser.parse_args()

    # set seed
    random.seed(opt.seed)

    # set config
    # config = load_config(opt)
    # config['opt'] = opt
    # logger.info("%s", config)

    dataner_preprocess(opt.data_dir)


if __name__ == '__main__':
    main()
