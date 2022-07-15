import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Dataset


class MenuDataset(Dataset):
    def __init__(self, input_features):
        total_input_ids = torch.tensor([f.token_id for f in input_features], dtype=torch.long)
        total_attention = torch.tensor([f.attention for f in input_features], dtype=torch.long)
        total_segment = torch.tensor([f.segment for f in input_features], dtype=torch.long)
        total_pos_ids = torch.tensor([f.pos_id for f in input_features], dtype=torch.long)
        total_char_ids = torch.tensor([f.char_ids for f in input_features], dtype=torch.long)
        total_label_ids = torch.tensor([f.label_id for f in input_features], dtype=torch.long)
        total_word2token_idx = torch.tensor([f.word2token_idx for f in input_features], dtype=torch.long)

        self.x = TensorDataset(total_input_ids, total_attention, total_segment,
                               total_pos_ids, total_char_ids, total_word2token_idx)
        self.y = total_label_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]