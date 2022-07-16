import json

from allennlp.modules.elmo import batch_to_ids
from tqdm import tqdm
from transformers import BertTokenizer


class InputExample(object):
    def __init__(self, sent, tokens, poss, labels):
        self.sent = sent
        self.tokens = tokens
        self.poss = poss
        self.labels = labels


class InputFeature(object):
    def __init__(self, token_id, label_id, pos_id, char_ids, attention, segment):
        self.token_id = token_id
        self.label_id = label_id
        self.pos_id = pos_id
        self.attention = attention
        self.char_ids = char_ids
        self.segment = segment


def pos2ix(train_ex):
    p_dict = {'PAD': 0}
    for ex in train_ex:
        for w in ex.poss:
            if w not in p_dict:
                p_dict[w] = len(p_dict)

    with open('../data/pos2ix_dict.json', 'w') as f:
        json.dumps(p_dict)
    return p_dict


def read_examples_data(file: str):
    examples = []
    tot_num_line = sum(1 for _ in open(file, 'r'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    with open(file, encoding="utf-8") as f:
        bucket = []

        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if (line == "") or ((len(bucket) != 0) and (idx == tot_num_line)):
                # It's the end of the sentence and
                # Check
                # for the last entry if it doesn't match the (line == "") condition
                tokens = []
                pos_seq = []
                label_seq = []
                for entry in bucket:
                    token = entry[0]
                    pos = entry[1]
                    label = entry[3]
                    if label == 'B-MENU' or label == 'I-MENU':
                        label = label.split('-')[1]

                    tokens.append(token)
                    t_w = tokenizer.tokenize(token)
                    if len(t_w) > 1:  # if word is split we have to align the number of labels and pos tags
                        for i in range(len(t_w)):
                            pos_seq.append(pos)
                            label_seq.append(label)
                    else:
                        # tokens.append(token)
                        pos_seq.append(pos)
                        label_seq.append(label)
                sent = ' '.join(tokens)
                examples.append(InputExample(sent=sent,
                                             tokens=tokens,
                                             poss=pos_seq,
                                             labels=label_seq))

                bucket = []
            else:
                entry = line.split()
                assert (len(entry) == 4)
                bucket.append(entry)

    examples.pop(0)  # Removal of the first since it is empty
    return examples


def convert_single_example_to_feature(example, tokenizer, tag_to_idx, pos_to_idx):
    label_id = [tag_to_idx[l] for l in example.labels]
    poss_id = [pos_to_idx[p] for p in example.poss]
    len_char_sequence = 50
    token_id = []
    pad_vec = [261 for i in range(len_char_sequence)]  # idx designed as 'pad' for this vocabulary
    char_ids = []
    max_length = 512

    for w in example.tokens:
        t = tokenizer.tokenize(w)
        c_ids = batch_to_ids([t])[0].detach().numpy().tolist()
        char_ids.extend(c_ids)
        token_id.extend(tokenizer.convert_tokens_to_ids(t))

    attention = [1 for i in range(len(token_id) + 1)]
    segment = [0 for i in range(len(token_id) + 1)]

    # ADD CLS
    char_ids.insert(0, pad_vec)
    token_id.insert(0, 101)
    label_id.insert(0, tag_to_idx['PAD'])
    poss_id.insert(0, pos_to_idx['PAD'])
    # ADD SEP
    token_id.append(102)

    # ADD PAD
    token_id.extend([0 for i in range(max_length - len(token_id))])
    label_id.extend([tag_to_idx['PAD'] for i in range(max_length - len(label_id))])
    poss_id.extend([pos_to_idx['PAD'] for i in range(max_length - len(poss_id))])
    char_ids.extend(pad_vec for i in range(max_length - len(char_ids)))
    attention.extend([0 for i in range(max_length - len(attention))])
    segment.extend([1 for i in range(max_length - len(segment))])

    return InputFeature(token_id, label_id, poss_id,
                        char_ids, attention, segment)


def convert_examples_to_feature(examples):
    input_feats = []
    tag_to_idx = {"PAD": 0, "MENU": 1, "O": 2}
    pos_to_idx = pos2ix(examples)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    for (ex_index, example) in enumerate(tqdm(examples)):
        input_feat = convert_single_example_to_feature(example, tokenizer, tag_to_idx, pos_to_idx)
        input_feats.append(input_feat)

    return input_feats
