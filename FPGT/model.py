import math

import torch
import torch.nn as nn


class CommonEmbedding(nn.Module):
    def __init__(self, num, embed_dim):
        super(CommonEmbedding, self).__init__()

        self.get_embedding = nn.Embedding(num_embeddings=num, embedding_dim=embed_dim, padding_idx=0)

    def forward(self, args, idx_list):
        idx_list = torch.LongTensor([idx_list]).to(device=args.device)
        embed_list = self.get_embedding(idx_list)
        return embed_list


class CheckInEmbedding(nn.Module):
    def __init__(self):
        super(CheckInEmbedding, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, args, user_embedding, region_embedding_list, hotness_embedding_list):
        checkin_embeddings = torch.randn(0, args.checkin_embed_dim).to(device=args.device)
        for i in range(len(region_embedding_list)):
            poi_embedding = torch.cat((hotness_embedding_list[i], region_embedding_list[i]), 0)
            poi_embedding = self.leaky_relu(poi_embedding)

            checkin_embedding = torch.cat((poi_embedding, user_embedding), 0)
            checkin_embedding = checkin_embedding.reshape(1, args.checkin_embed_dim)
            checkin_embeddings = torch.cat((checkin_embeddings, checkin_embedding), 0)
        return checkin_embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder = nn.Linear(embed_size, num)
        self.init_weights()
        self.softmax = nn.Softmax()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)

        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)

        pred = self.decoder(x)
        return pred
