from models.PositionalEncoding import PositionalEncoding
import math
import os
import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder


class MLP(nn.Module):
    def __init__(self, hidden_dim=32, num_att_head=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_att_head)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP2(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=32, num_att_head=4):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_att_head)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class encoderLayer(nn.Module):
    def __init__(self, embedding_size, output_dim=256, special_cnt=2):
        super(encoderLayer, self).__init__()

        self.special_cnt = special_cnt
        self.embedding = nn.Embedding(self.special_cnt, embedding_size)

        self.fc1 = nn.Linear(embedding_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, x, extern_feats=None):
        if extern_feats is not None:
            special_emb = self.embedding.weight[:self.special_cnt]  # [special_cnt, emb]
            full_feats = torch.cat((special_emb, extern_feats), dim=0)  # [total_tokens, emb]
            x = full_feats[x]  # gather
        else:
            x = self.embedding(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Transformer(nn.Module):
    def __init__(self, embedding_size, reduced_size, nhead, num_encoder_layers, dim_feedforward, dropout, max_len):
        super(Transformer, self).__init__()
        self.reduced_size = reduced_size

        self.embedding = encoderLayer(embedding_size, reduced_size)

        self.pos_encoder = PositionalEncoding(reduced_size, dropout, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=reduced_size, nhead=nhead,
                                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src, extern_feats=None):
        src_key_padding_mask = (src == 0)
        src = self.embedding(src, extern_feats) * math.sqrt(self.reduced_size)
        src = self.pos_encoder(src)
        emb = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return src.clone(), emb


class Topoformer(nn.Module):
    def __init__(self, embedding_size, reduced_size, nhead, num_encoder_layers, dim_feedforward, dropout, max_len):
        super(Topoformer, self).__init__()
        self.reduced_size = reduced_size
        self.nhead = nhead

        self.embedding = encoderLayer(embedding_size, reduced_size)
        self.pos_encoder = PositionalEncoding(reduced_size, dropout, max_len)

        self.bit_width_encoder = DegreeEncoder(max_degree=128, embedding_dim=reduced_size, direction='in')
        self.density_encoder = MLP(hidden_dim=32, num_att_head=nhead)
        self.max_path_encoder = MLP(hidden_dim=32, num_att_head=nhead)
        self.virtual_distance = nn.Embedding(1, nhead)
        self.virtual_distance_mlp = MLP2(in_dim=2, hidden_dim=32, num_att_head=nhead)
        self.vir_layer_norm = nn.LayerNorm(2)
        self.emb_layer_norm = nn.LayerNorm(reduced_size)

        self.density_weight = nn.Parameter(torch.ones(1))  # scalar
        self.path_weight = nn.Parameter(torch.ones(1))  # scalar

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=reduced_size,
                    hidden_size=dim_feedforward,
                    num_heads=nhead,
                    dropout=dropout,
                    activation=nn.GELU(),
                    norm_first=False,
                )
                for _ in range(num_encoder_layers)
            ]
        )

    def generate_attention_mask(self, src):
        src_key_padding_mask = (src == 0)  # [batch_size, max_length]
        mask = src_key_padding_mask.unsqueeze(1) | src_key_padding_mask.unsqueeze(2)
        # mask: [batch_size, max_length, max_length]
        all_true_rows = mask.all(dim=-1)  # [batch_size, max_length]
        if all_true_rows.any():
            batch_indices, row_indices = torch.where(all_true_rows)
            mask[batch_indices, row_indices, 0] = False
        return mask

    def construct_positional_embedding(self, src, bit_centrality=None, density_spatial=None, path_spatial=None):
        '''
        :param src: [batch_size, max_length, reduced_size] (max_length = max_nodes + 1)
        :param bit_centrality: [batch_size, max_nodes]
        :param density_spatial: [batch_size, max_nodes, max_nodes]
        :param path_spatial: [batch_size, max_nodes, max_nodes]
        :return:
        '''

        num_graphs, max_length, emb_size = src.shape
        centra_emb = torch.zeros(num_graphs, max_length, emb_size, device=src.device)
        attn_bias = torch.zeros(num_graphs, max_length, max_length, self.nhead, device=src.device)

        if bit_centrality is not None:
            bit_emb = self.bit_width_encoder(bit_centrality.long())
            centra_emb[:, 1:, :] = bit_emb
        if density_spatial is not None:
            density_emb = self.density_encoder(density_spatial.unsqueeze(-1))
            attn_bias[:, 1:, 1:, :] += self.density_weight * density_emb
        if path_spatial is not None:
            path_emb = self.max_path_encoder(path_spatial.unsqueeze(-1))
            attn_bias[:, 1:, 1:, :] += self.path_weight * path_emb

        if density_spatial is not None or path_spatial is not None:
            t = self.virtual_distance.weight.reshape(1, 1, self.nhead)
            attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
            attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t

        return centra_emb, attn_bias

    def forward(self, src, extern_feats=None, bit_centrality=None, density_spatial=None, path_spatial=None):
        attn_mask = self.generate_attention_mask(src)
        if bit_centrality is None and density_spatial is None and path_spatial is None:
            src = self.embedding(src, extern_feats) * math.sqrt(self.reduced_size)
            src = self.pos_encoder(src)
            src_clone = src.clone()
            emb = self.emb_layer_norm(src)
            for layer in self.layers:
                emb = layer(emb, attn_mask=attn_mask)
        else:
            src = self.embedding(src, extern_feats) * math.sqrt(self.reduced_size)
            centra_emb, attn_bias = self.construct_positional_embedding(src, bit_centrality, density_spatial, path_spatial)
            src += centra_emb
            src_clone = src.clone()
            emb = self.emb_layer_norm(src)
            for layer in self.layers:
                emb = layer(emb, attn_mask=attn_mask, attn_bias=attn_bias)

        return src_clone, emb

