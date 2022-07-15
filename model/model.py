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

        # self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
        #                        kernel_size=self.kernel_sizes[0])
        # self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
        #                        kernel_size=self.kernel_sizes[1])

        convs = []
        for ks in kernel_sizes:
            convs.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=ks))
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # convolutions = [relu(self.conv1(x.type(torch.FloatTensor))), relu(self.conv2(x.type(torch.FloatTensor)))]
        conved = [relu(conv(x)) for conv in self.convs]
        max_pooled = [torch.max(c, dim=2)[0] for c in conved]
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
        self.dropout = nn.Dropout(0.1)
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
        self.dim_multiHeadAtt = self.hidden_dim * 2
        self.multihead_attn = nn.MultiheadAttention(self.dim_multiHeadAtt, self.num_heads, batch_first=True)
        self.norm_layer = nn.LayerNorm(self.dim_multiHeadAtt)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.dim_multiHeadAtt, self.tagset_size)

        self.crf_module = CRF(self.tagset_size, batch_first=True)

    def init_hidden(self):
        return ((torch.randn(self.num_layers * 2, 1, self.hidden_dim)),
                (torch.randn(self.num_layers * 2, 1, self.hidden_dim)))

    def _get_attention_out_(self, x, att):

        padding_mask = att.ne(1)
        query = x
        key = x
        value = x
        attn_output, attn_output_weights = self.multihead_attn(query, key, value, key_padding_mask=padding_mask)
        return attn_output

    def _get_lstm_features(self, x):

        h = self.init_hidden()
        embeds = self.embedder(x)
        print("Embedding DONE")
        embeds = self.dropout(embeds)
        # embeds=embeds.view(len(sentence), 1, -1)
        att = x[1]
        lengths = torch.sum(att, 1)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True,
                                                         enforce_sorted=False)
        lstm_out, _ = self.lstm(embeds)
        print("lstm DONE")
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=512)
        lstm_out = self.dropout(lstm_out)

        if self.attention:
            att_out = self._get_attention_out_(lstm_out, att)
            lstm_out = self.norm_layer(att_out + lstm_out)
            lstm_out = self.dropout(lstm_out)
            print("attention DONE")

        print(lstm_out.size())
        lstm_feats = self.hidden2tag(lstm_out)
        print("Linear DONE")
        dim = lstm_feats.size(1)
        print(att.size())
        att = att.byte()
        return lstm_feats, att

    def neg_log_likelihood(self, x, tags):
        mask = torch.sign(torch.abs(x[1])).to(torch.uint8)
        feats, att = self._get_lstm_features(x)
        gold_score = self.crf_module(feats, tags, mask=mask)
        print("Neg_Likelihood DONE")
        return -1 * gold_score

    def forward(self, sentence):
        lstm_feats, att = self._get_lstm_features(sentence)
        print(lstm_feats.size())
        logits = lstm_feats
        tag_seq = self.crf_module.decode(lstm_feats)
        prediction = tag_seq
        # tag_seq = torch.as_tensor(tag_seq, dtype=torch.long)
        print("MODEL DONE")
        return logits, prediction
