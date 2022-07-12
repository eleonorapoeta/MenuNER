import torch.nn as nn
from transformers import BertForPreTraining
import torch
from torchcrf import CRF


class Embedders(nn.Module):
    def __init__(self, bert_path, pos2ix=None, pos_dim=2, char=False, pos=False):  # pos_dim scelta arbitrariamente
        # per test, non sappiamo quale sia da usare
        params = {
            'char_voc_size': 262,
            'char_embedding_dim': 25,
            'num_filter': 30,
            'kernels': [3, 9]
        }
        super(Embedders, self).__init__()
        print("ciao")
        print(len(pos2ix))
        self.pos = pos

        if pos:
            self.pos_dim = pos_dim
            self.pos2ix = pos2ix
            self.pos_embedder = nn.Embedding(len(pos2ix) + 1, pos_dim, padding_idx=0)

        self.bert = bert_model = BertForPreTraining.from_pretrained(bert_path)
        self.char_embedder = self.CharCNN(in_channels=params['char_embedding_dim'],
                                          out_channels=params['num_filter'], kernel_sizes=params['kernels'])

    def char_cnn(self, x):
        #
        # char_emb = nn.Embedding(num_embeddings=params['char_voc_size'],
        #                         embedding_dim=params['char_embedding_dim'],
        #                         padding_idx=0)

        out = self.char_embedder(x)
        print(out.size())

        charcnn_out = out.view(-1, 180, out.shape[-1])
        # charcnn_out : [batch_size, seq_size, last_dim]
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

    def forward(self, x, p=None):

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
        # print(x.size())
        x = x.permute(0, 2, 1)
        convolutions = [self.conv1(x), self.conv2(x)]
        max_pooled = [torch.max(c, dim=2)[0] for c in convolutions]
        print(max_pooled)
        cat = torch.cat(max_pooled, dim=1)
        print(cat.size())
        # cat : [batch_size, len(kernel_sizes) * num_filters]
        return cat


class BiLSTM_CRF(nn.Module):

    def __init__(self, tagset_size, embedding_dim, hidden_dim, attention=False, num_layers=2, num_heads=4,
                 model_checkpoint='../model_checkpoint', pos2ix=None, pos_dim=2, char=False,
                 pos=False):
        super(BiLSTM_CRF, self).__init__()
        if pos:
            self.embedding_dim = embedding_dim + pos_dim
        else:
            self.embedding_dim = embedding_dim
        self.pos = True
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tagset_size = tagset_size

        self.attention = attention
        self.num_heads = num_heads

        # pos2ix=None,pod_dim=1,char=False,pos=False
        self.embedder = Embedders(bert_path=model_checkpoint, pos2ix=pos2ix, pos=self.pos, pos_dim=pos_dim, char=True)

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
        att = x[1]
        lengths = torch.sum(att, 1)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True,
                                                         enforce_sorted=False)
        lstm_out, _ = self.lstm(embeds)

        if self.attention:
            lstm_out = self._get_attention_out_(lstm_out)

        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        print(lstm_out.size())
        lstm_feats = self.hidden2tag(lstm_out)

        dim = lstm_feats.size(1)
        print(att.size())
        att = att[:, :dim]
        att = att.byte()
        return lstm_feats, att

    def neg_log_likelihood(self, x, tags):
        feats = self._get_lstm_features(x)
        gold_score = self.crf(feats, tags)
        return -1 * gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM

        lstm_feats, att = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        print(lstm_feats.size())

        tag_seq = self.crf_module.decode(lstm_feats, att)
        return tag_seq
