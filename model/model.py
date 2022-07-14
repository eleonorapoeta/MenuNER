import torch.nn as nn
import torch
from torchcrf import CRF
from torch.nn.functional import relu
from .embeddings import Embedders


class CharCNN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                               kernel_size=self.kernel_sizes[0])
        self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                               kernel_size=self.kernel_sizes[1])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        convolutions = [relu(self.conv1(x.type(torch.FloatTensor))), relu(self.conv2(x.type(torch.FloatTensor)))]
        max_pooled = [torch.max(c, dim=2)[0] for c in convolutions]
        print(max_pooled)
        cat = torch.cat(max_pooled, dim=1)
        print(cat.size())
        # cat : [batch_size, len(kernel_sizes) * num_filters]
        return cat


class BiLSTM_CRF(nn.Module):

    def __init__(self, tagset_size, embedding_dim, hidden_dim, attention=False, num_layers=2, num_heads=4,
                 bert_checkpoint='../bert_checkpoint', pos2ix=None, pos_dim=2, char=False,
                 pos=False):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        params = {
            'char_voc_size': 262,
            'char_embedding_dim': 25,
            'num_filter': 30,
            'kernels': [3, 9]
        }

        self.pos = True
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tagset_size = tagset_size

        self.attention = attention
        self.num_heads = num_heads

        # pos2ix=None,pod_dim=1,char=False,pos=False
        self.embedder = Embedders(bert_path=bert_checkpoint, pos2ix=pos2ix, pos=self.pos,
                                  pos_dim=pos_dim, char_vocab_size=262,
                                  char_len=50,
                                  max_length=512,
                                  char_embedding_dim=25,  # Set by us
                                  char=True)

        if pos:
            self.embedding_dim += pos_dim

        if char:
            self.embedding_dim += self.embedder.last_dim

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            num_layers=2, bidirectional=True)
        # ADD ATTENTION
        self.multihead_attn = nn.MultiheadAttention(self.hidden_dim * 2, self.num_heads, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)

        self.crf_module = CRF(self.tagset_size, batch_first=True)

    def init_hidden(self):
        return ((torch.randn(self.num_layers * 2, 1, self.hidden_dim)),
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

        h = self.init_hidden()
        embeds = self.embedder(x)
        print("Embedding DONE")

        # embeds=embeds.view(len(sentence), 1, -1)
        att = x[1]
        lengths = torch.sum(att, 1)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True,
                                                         enforce_sorted=False)
        lstm_out, _ = self.lstm(embeds)
        print("lstm DONE")
        print(lstm_out.size())
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        if self.attention:
            lstm_out = self._get_attention_out_(lstm_out)
            print("attention DONE")

        print(lstm_out.size())
        lstm_feats = self.hidden2tag(lstm_out)
        print("Linear DONE")
        dim = lstm_feats.size(1)
        print(att.size())
        att = att[:, :dim]
        att = att.byte()
        return lstm_feats, att

    def neg_log_likelihood(self, x, tags):

        feats, att = self._get_lstm_features(x)
        gold_score = self.crf_module(feats, tags, mask=att)
        print("Neg_Likelihood DONE")
        return -1 * gold_score

    def forward(self, sentence):
        lstm_feats, att = self._get_lstm_features(sentence)
        print(lstm_feats.size())
        tag_seq = self.crf_module.decode(lstm_feats, att)
        print("MODEL DONE")
        return tag_seq
