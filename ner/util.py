import os

from tqdm import tqdm
from transformers import BertTokenizer


class InputExample(object):
    def __init__(self, sent, tokens, poss, labels):
        self.sent = sent
        self.tokens = tokens  # token di nltk
        self.poss = poss
        self.labels = labels


class InputFeature(object):
    def __int__(self, input_ids, pos_ids, char_ids, label_ids, word2token_idx):
        self.input_ids = input_ids
        # self.input_mask = input_mask
        # self.segment_ids = segment_ids
        self.pos_ids = pos_ids
        # self.chunk_ids = chunk_ids
        self.char_ids = char_ids
        self.label_ids = label_ids
        self.word2token_idx = word2token_idx


def pos2ix(train_ex):
    p_dict = {}
    for ex in train_ex:
        for w in ex.poss:
            if w not in p_dict:
                p_dict[w] = len(p_dict)
    return p_dict
    # AGGIUNGERE CARATTERE PER OUT OF VOCAB


def read_examples_data(file: str, tokenizer):
    examples = []
    tot_num_line = sum(1 for _ in open(file, 'r'))

    with open(file, encoding="utf-8") as f:
        bucket = []

        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if (line == "") or ((len(bucket) != 0) and (idx == tot_num_line)):  # It's the end of the sentence and
                # Check
                # for the last entry if it doesn't match the (line == "") condition
                tokens = []
                pos_seq = []
                # chunkseq = []
                label_seq = []
                for entry in bucket:
                    token = entry[0]  # first column from CoNLL2003 dataset (Token)
                    pos = entry[1]  # second column from CoNLL2003 dataset (POS)
                    # chunk = entry[2]  # third column from CoNLL2003 dataset (Chunk)
                    label = entry[3]  # last column from CoNLL2003 dataset (Label)

                    tokens.append(token)
                    t_w = tokenizer.tokenize(token)
                    if len(t_w) > 1:  # if word is split we have to align the number of labels and pos tags
                        # tokens.extend(t_w)
                        for i in range(len(t_w)):
                            pos_seq.append(pos)
                            if i != 0:
                                if label == 'B-MENU' or label == 'I-MENU':
                                    # divido in B-MENU e I-MENU anche se la
                                    # parola è una
                                    label_seq.append('I-MENU')
                                else:
                                    label_seq.append(label)
                            else:
                                label_seq.append(label)
                    else:
                        # tokens.append(token)
                        pos_seq.append(pos)
                        label_seq.append(label)
                sent = ' '.join(tokens)
                examples.append(InputExample(sent=sent,
                                             tokens=tokens,
                                             poss=pos_seq,
                                             # chunks=chunkseq,
                                             labels=label_seq))

                bucket = []
            else:
                entry = line.split()
                assert (len(entry) == 4)
                bucket.append(entry)

    return examples


def convert_single_example_to_feature(self, example, tokenizer, tag_to_idx, pos_to_idx):
    label_ids = [tag_to_idx[l] for l in example.labels]
    poss_ids = [pos_to_idx[p] for p in example.poss]

    token_ids = []
    tokens = []
    # Word extension
    for w in example.words:
        t = tokenizer.tokenize(w)
        tokens.extend(t)
        token_ids.extend(tokenizer.convert_tokens_to_ids(t))
    attention = [1 for i in range(len(token_ids))]
    segment = [0 for i in range(len(token_ids))]


def convert_examples_to_feature(self, examples, tokenizer):
    features = []

    tag_to_idx = {"PAD": 0, "B-MENU": 1, "I-MENU": 2, "O": 3, "STOP_TAG": 4}
    pos_to_idx = pos2ix(examples)
    for (ex_index, example) in enumerate(tqdm(examples)):
        '''
        if ex_index % 1000 == 0:get_examples_dataset
            logger.info("Writing example %d of %d", ex_index, len(examples))
        '''

        feature = convert_single_example_to_feature(example, tokenizer, tag_to_idx, pos_to_idx)

    features.append(feature)

    return features
