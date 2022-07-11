import json

from allennlp.modules.elmo import batch_to_ids
from tqdm import tqdm
import torch


class InputExample(object):
    def __init__(self, sent, tokens, poss, labels):
        self.sent = sent
        self.tokens = tokens  # token di nltk
        self.poss = poss
        self.labels = labels


def word2charix(word, alpha_dict):
    w_c = []
    len_v = 25
    for c in word:
        w_c.append(alpha_dict[c])
    if len(w_c) < len_v:
        w_c.extend([alpha_dict['PAD'] for i in range(len_v - len(w_c))])
    return w_c


def char2ix(alphabet_path: str):
    with open(alphabet_path, 'r') as f:
        data = json.load(f)
    alpha_dict = {'PAD': 0}
    for i, a in enumerate(data):
        if a not in alpha_dict:
            alpha_dict[a] = i + 1

    return alpha_dict


def pos2ix(train_ex):
    p_dict = {'PAD': 0}
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
                label_seq = []
                for entry in bucket:
                    token = entry[0]
                    pos = entry[1]
                    label = entry[3]

                    tokens.append(token)
                    t_w = tokenizer.tokenize(token)
                    if len(t_w) > 1:  # if word is split we have to align the number of labels and pos tags
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
                                             labels=label_seq))

                bucket = []
            else:
                entry = line.split()
                assert (len(entry) == 4)
                bucket.append(entry)

    examples.pop(0)  # Removal of the first since it is empty
    return examples


def convert_single_example_to_feature(example, tokenizer, tag_to_idx, pos_to_idx, alpha_dict):
    max_length = 512
    len_padded_word = 50
    label_ids = [tag_to_idx[l] for l in example.labels]
    poss_ids = [pos_to_idx[p] for p in example.poss]
    padding_char_vector = [alpha_dict['PAD'] for i in range(len_padded_word)]
    word_char = []
    token_ids = []
    char_ids = []
    tokens = []
    # Word extension
    for w in example.tokens:
        t = tokenizer.tokenize(w)
        tokens.extend(t)
        token_ids.extend(tokenizer.convert_tokens_to_ids(t))
        # c_ids = batch_to_ids([t])[0].detach().cpu().numpy().tolist()  # get char ids
        # word_char.extend(c_ids)  # extends char_ids to word_token level (for ['Order', '##ed'], [[234, 232,..], [234, 232,..]])
        # w = w.lower()
        # if len(t) > 1:
        #     for i in range(len(t)):
        #         word_char.append(word2charix(w, alpha_dict))
        # else:
        #     word_char.append(word2charix(w, alpha_dict))

    attention = [1 for i in range(len(token_ids) + 1)]  # +1 because we are counting also [CLS] token
    segment = [0 for i in range(len(token_ids) + 1)]

    # ADD CLS
    # word_char.insert(0, padding_char_vector)
    token_ids.insert(0, 101)  # [CLS] = 101
    label_ids.insert(0, tag_to_idx['PAD'])  # Padding because there is no correspondence between [CLS] and labels
    poss_ids.insert(0, pos_to_idx['PAD'])  # Padding because there is no correspondence between [CLS] and poss

    # ADD SEP
    token_ids.append(102)  # [SEP] = 102

    # ADD PAD - padding to max_length for batching

    token_ids.extend([0 for i in range(max_length - len(token_ids))])
    label_ids.extend([tag_to_idx['PAD'] for i in range(max_length - len(label_ids))])
    poss_ids.extend([pos_to_idx['PAD'] for i in range(max_length - len(poss_ids))])
    attention.extend([0 for i in range(max_length - len(attention))])
    segment.extend([1 for i in range(max_length - len(segment))])
    # word_char.extend([padding_char_vector for i in range(512 - len(word_char))])

    feature = torch.tensor([token_ids, poss_ids, attention, segment])

    return feature, torch.tensor(label_ids)


def convert_examples_to_feature(examples, tokenizer, alphabet_path):
    features = []
    labels = []
    word_chars = []
    tag_to_idx = {"PAD": 0, "B-MENU": 1, "I-MENU": 2, "O": 3, "STOP_TAG": 4}
    pos_to_idx = pos2ix(examples)
    alpha_dict = char2ix(alphabet_path)
    for (ex_index, example) in enumerate(tqdm(examples)):
        '''
        if ex_index % 1000 == 0:get_examples_dataset
            logger.info("Writing example %d of %d", ex_index, len(examples))
        '''

        feature, label = convert_single_example_to_feature(example, tokenizer, tag_to_idx, pos_to_idx, alpha_dict)
        features.append(feature)
        labels.append(label)

    return features, labels
