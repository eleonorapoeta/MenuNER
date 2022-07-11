import torch.nn as nn
from transformers import BertForPreTraining
import torch
from torchcrf import CRF


class Embedders(nn.Module):
    def __init__(self, bert_path, pos2ix=None, pos_dim=2, char=False, pos=False):  # pos_dim scelta arbitrariamente
        # per test, non sappiamo quale sia da usare
        super(Embedders, self).__init__()
        print("ciao")
        print(len(pos2ix))
        self.pos = pos

        if pos:
            self.pos_dim = pos_dim
            self.pos2ix = pos2ix
            self.pos_embedder = nn.Embedding(len(pos2ix) + 1, pos_dim, padding_idx=0)

        self.char = char

        self.bert = bert_model = BertForPreTraining.from_pretrained('../model_checkpoint/bert')

    def bert_emb(self, x):
        params = {
            'input_ids': x[:, 0, :],
            'attention_mask': x[:, 2, :],
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

    def forward(self, x, p=None):

        x_p = x[:, 1, :]
        x = self.bert_emb(x)
        if self.pos:
            x_p = self.pos_embedder(x_p)
            x = torch.cat([x, x_p], dim=-1)

        return x


class BiLSTM_CRF(nn.Module):

    def __init__(self, tagset_size, embedding_dim, hidden_dim, attention=False, num_layers=2, num_heads=4,
                 model_checkpoint='../model_checkpoint/bert', pos2ix=None, pos_dim=2, char=False,
                 pos=False):
        super(BiLSTM_CRF, self).__init__()
        if pos:
            self.embedding_dim = embedding_dim + pos_dim
        else:
            self.embedding_dim = embedding_dim
        self.pos = pos
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tagset_size = tagset_size

        self.attention = attention
        self.num_heads = num_heads

        # pos2ix=None,pod_dim=1,char=False,pos=False
        self.embedder = Embedders(bert_path=model_checkpoint, pos2ix=pos2ix, pos=self.pos, pos_dim=pos_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            num_layers=2, bidirectional=True)
        # ADD ATTENTION
        self.multihead_attn = nn.MultiheadAttention(self.hidden_dim * 2, self.num_heads)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)

        self.crf_module = CRF(self.tagset_size, batch_first=True)

    def init_hidden(self, x):
        return ((torch.randn(self.num_layers * 2, 1, self.hidden_dim)),  # eliminato .cuda()
                (torch.randn(self.num_layers * 2, 1, self.hidden_dim)))

    def _get_attention_out_(self, x):
        query = x
        key = x
        value = x
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        lstm_feats = attn_output
        # if self.layer_norm:
        # lstm_feats= self.layernorm_mha(mha_out + lstm_feats) #da capire perché li somma

        return lstm_feats

    def _get_lstm_features(self, x):

        h = self.init_hidden(x)
        embeds = self.embedder(x)

        # embeds=embeds.view(len(sentence), 1, -1)
        att = x[:, 2, :]
        lengths = torch.sum(att, 1)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True,
                                                         enforce_sorted=False)
        lstm_out, _ = self.lstm(embeds)

        if self.attention:
            lstm_out = self._get_attention_out_(lstm_out)

        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=180)
        print(lstm_out.size())
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def neg_log_likelihood(self, x, tags):
        feats = self._get_lstm_features(x)
        gold_score = self.crf(feats, tags)
        return -1 * gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM

        lstm_feats = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        print(lstm_feats.size())
        tag_seq = self.crf_module.decode(lstm_feats)
        return tag_seq
