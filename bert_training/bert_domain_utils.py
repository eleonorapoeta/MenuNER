from torch.utils.data.dataset import Dataset
import torch
from transformers import BertTokenizer


class FoodDomainDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings  # pass the input

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def input_data_prep(sentences_a, sentences_b, labels):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    inputs = tokenizer(sentences_a,
                       sentences_b,
                       return_tensors='pt',
                       max_length=512,
                       truncation=True,
                       padding='max_length'
                       )

    inputs['next_sentence_label'] = torch.LongTensor([labels]).T  # insertion of the labels for NSP
    inputs['labels'] = inputs.input_ids.detach().clone()  # labels creation per MLM where label = word itself
    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

    for i in range(inputs.input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()  # selection of indices to mask
        inputs.input_ids[i, selection] = 103

    return inputs
