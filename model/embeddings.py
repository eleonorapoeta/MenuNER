from torch import nn
from transformers import BertForPreTraining
import torch


class Embedders(nn.Module):
    def __init__(self, bert_path, char_vocab_size, char_embedding_dim, char_len, max_length, pos2ix=None, pos_dim=2,
                 char=False,
                 pos=False):

        from model import CharCNN

        params = {
            'in_channel': 25,
            'out_channel': 30,  # Set by us
            'kernel': [3, 9],
        }

        super(Embedders, self).__init__()
        self.pos = pos
        self.char = char
        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.char_len = char_len
        self.max_length = max_length
        self.last_dim = len(params['kernel']) * params['out_channel']

        if char:
            self.char_emb_layer = nn.Embedding(self.char_vocab_size, self.char_embedding_dim, padding_idx=0)
            self.char_embedder = CharCNN(params['in_channel'], params['out_channel'], params['kernel'])

        if pos:
            self.pos_dim = pos_dim
            self.pos2ix = pos2ix
            self.pos_embedder = nn.Embedding(len(pos2ix) + 1, pos_dim, padding_idx=0)

        self.bert = BertForPreTraining.from_pretrained(bert_path)

    def char_cnn(self, x):

        # Embedding layer
        char_embed_out = self.char_emb_layer(x)
        char_embed_out = char_embed_out.view(-1, self.char_len, self.char_embedding_dim)
        mask = x.view(-1, self.char_len).ne(0)
        mask = mask.unsqueeze(2).to(torch.float)
        char_embed_out *= mask

        out = self.char_embedder(char_embed_out)
        charcnn_out = out.view(-1, self.max_length, out.shape[-1])
        return charcnn_out

    def bert_emb(self, x):
        params = {
            'input_ids': x[0],
            'attention_mask': x[1],
            'output_hidden_states': True,
            'output_attentions': True,
            'return_dict': True
        }

        out = self.bert(**params)
        last_four_hidden_states = out.hidden_states[-4:]
        stack = torch.stack(last_four_hidden_states, dim=-1)
        embedded = torch.mean(stack, dim=-1)
        # embedded = embedded.view(embedded.size(1), embedded.size(0), embedded.size(2))

        return embedded

    def forward(self, x):

        x_p = x[3]
        x_b = self.bert_emb(x)
        x_c = x[4]

        if self.pos:
            x_p = self.pos_embedder(x_p)
            out = torch.cat([x_b, x_p], dim=-1)
        else:
            out = x_b

        if self.char:
            x_c = self.char_cnn(x_c)
            out = torch.cat([out, x_c], dim=-1)

        return out
