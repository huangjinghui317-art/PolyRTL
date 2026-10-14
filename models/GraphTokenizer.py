import torch
from torch import nn
import numpy as np
from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder
from .utils import create_activation, MLP_dec, setup_loss_fn, MLP_dec2d, TripletLoss


class Graphormer(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.config = config
        dropout = config["dropout"]
        feat_dim = config["feat_dim"]
        self.num_heads = num_heads = config["num_heads"]
        self.encoding = config["encoding"]
        activation_fn = config["activation"]
        pre_layernorm = config["pre_layernorm"]
        ffn_embedding_dim = config["num_hidden"]
        out_dim = config["embed_dim"]
        edge_dim = config["edge_dim"]
        max_degree = config["max_degree"]
        num_spatial = config["num_spatial"]
        multi_hop_max_dist = config["multi_hop_max_dist"]
        num_layers = config["num_layers"]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.graph_token = nn.Embedding(1, feat_dim)

        self.degree_encoder = DegreeEncoder(
            max_degree=max_degree, embedding_dim=feat_dim
        )

        self.path_encoder = PathEncoder(
            max_len=multi_hop_max_dist,
            feat_dim=edge_dim,
            num_heads=num_heads,
        )

        self.spatial_encoder = SpatialEncoder(
            max_dist=num_spatial, num_heads=num_heads
        )
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.emb_layer_norm = nn.LayerNorm(feat_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=feat_dim,
                    hidden_size=ffn_embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=create_activation(activation_fn),
                    norm_first=pre_layernorm,
                )
                for _ in range(num_layers)
            ]
        )

        # map graph_rep to num_classes
        self.lm_head_transform_weight = nn.Linear(
            feat_dim, feat_dim
        )
        self.layer_norm = nn.LayerNorm(feat_dim)
        self.activation_fn = create_activation(activation_fn)
        self.embed_out = nn.Linear(feat_dim, out_dim, bias=False)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(out_dim))

        ### add a fc layer to map the graph representation to the output (from embedding_dim to 256)
        self.fc = nn.Linear(feat_dim, out_dim, bias=False)

        # for masked node modeling task
        self.graph_proj = nn.Linear(config['embed_dim'], config['embed_dim'])
        self.gtmae_decoder = MLP_dec2d(
            input_dim=config['embed_dim'],
            hidden_dim=512,
            num_layers=3,
            output_dim=config['feat_dim'],
            activation="relu",
            norm="batchnorm"
        )

        self.encoder_to_decoder = nn.Linear(config['embed_dim'], config['embed_dim'], bias=False)
        self.gsm_head = nn.Linear(config['embed_dim'], 2)
        self.loss_gtmae = setup_loss_fn("mse")

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.embed_out.reset_parameters()

    def forward(
            self,
            batched_data,
            mask=None,
    ):
        (attn_mask, node_feat, in_degree, out_degree, path_data, dist) = batched_data
        num_graphs, max_num_nodes, _ = node_feat.shape
        deg_emb = self.degree_encoder(torch.stack((in_degree, out_degree)))

        if self.encoding:
            node_feat_comb = node_feat + deg_emb
            graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(
                num_graphs, 1, 1
            )
            x = torch.cat([graph_token_feat, node_feat_comb], dim=1)

        attn_bias = torch.zeros(
            num_graphs,
            max_num_nodes + 1,
            max_num_nodes + 1,
            self.num_heads,
            device=dist.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

        t = self.graph_token_virtual_distance.weight.reshape(
            1, 1, self.num_heads
        )

        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t

        if (mask is not None) and self.encoding:
            x = x[:, ~mask]  # [n graph, n non-masked nodes, n hidden]
            attn_bias = attn_bias[:, ~mask, :, :][:, :, ~mask,
                        :]  # [n graph, n non-masked nodes, n non-masked nodes, n heads,]
            attn_mask = attn_mask[:, ~mask, :][:, :, ~mask]

        x = self.emb_layer_norm(x)

        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        graph_rep = x[:, 0, :]
        node_rep = x[:, 1:, :]

        node_rep = self.embed_out(node_rep)
        graph_rep = self.embed_out(graph_rep) + self.lm_output_learned_bias

        return node_rep, graph_rep

    def mask_attr_prediction(self, batched_data, mask):
        (attn_mask, node_feat, in_degree, out_degree, path_data, dist) = batched_data

        output, graph_emb = self.forward(batched_data, mask)
        output = self.encoder_to_decoder(output)

        mask = mask[1:]  ## remove the first bit of mask tensor
        gold = node_feat[:, mask]
        num_masked = mask.nonzero().view(-1).shape[0]  # real masked size

        mask_token = nn.Parameter(torch.zeros(output.shape[0], num_masked, output.shape[2]))
        dec_input = torch.cat([output, mask_token.to(attn_mask.device)], dim=1)

        dec_outputs = self.gtmae_decoder(dec_input)
        dec_outputs = dec_outputs[:, -num_masked:, :]

        return dec_outputs, gold, graph_emb

    def mlm_forward(self, graph_data):
        num_nodes = graph_data[1].size(1)
        mask_rate = self.config['mask_rate']
        num_mask = int(mask_rate * num_nodes)
        mask = np.hstack([
            np.zeros(num_nodes - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(mask)
        mask = torch.Tensor(mask).bool()

        ### add Fasle at the beginning of mask tensor
        mask = torch.cat([torch.tensor([False]), mask])

        x_rec_masked, x_init_mased, graph_emb = self.mask_attr_prediction(graph_data, mask)
        x_init = x_init_mased.view(-1)
        x_rec = x_rec_masked.contiguous().view(-1)  # [n graphs, n nodes, n feature]->[n graphs * n nodes,  n feature]

        if x_init.shape[0] == 0:
            loss = torch.tensor(0.0)
            loss_item = {"loss": loss.item()}
            print("No masked nodes")
            input()
            return loss, loss_item, graph_emb

        loss_gtmae = self.loss_gtmae(x_rec, x_init)

        return loss_gtmae


class Graphormer2(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        # dropout = nn.Dropout(p=config["dropout"])
        self.config = config
        dropout = config["dropout"]
        feat_dim = config["feat_dim"]
        self.num_heads = num_heads = config["num_heads"]
        self.encoding = config["encoding"]
        activation_fn = config["activation"]
        pre_layernorm = config["pre_layernorm"]
        ffn_embedding_dim = config["num_hidden"]
        out_dim = config["embed_dim"]
        edge_dim = config["edge_dim"]
        max_degree = config["max_degree"]
        num_spatial = config["num_spatial"]
        multi_hop_max_dist = config["multi_hop_max_dist"]
        num_layers = config["num_layers"]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.graph_token = nn.Embedding(1, feat_dim)

        self.degree_encoder = DegreeEncoder(
            max_degree=max_degree, embedding_dim=feat_dim
        )

        self.path_encoder = PathEncoder(
            max_len=multi_hop_max_dist,
            feat_dim=edge_dim,
            num_heads=num_heads,
        )

        self.spatial_encoder = SpatialEncoder(
            max_dist=num_spatial, num_heads=num_heads
        )
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.emb_layer_norm = nn.LayerNorm(feat_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=feat_dim,
                    hidden_size=ffn_embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=create_activation(activation_fn),
                    norm_first=pre_layernorm,
                )
                for _ in range(num_layers)
            ]
        )

        # map graph_rep to num_classes
        self.lm_head_transform_weight = nn.Linear(
            feat_dim, feat_dim
        )
        self.layer_norm = nn.LayerNorm(feat_dim)
        self.activation_fn = create_activation(activation_fn)
        self.embed_out = nn.Linear(feat_dim, out_dim, bias=False)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(out_dim))

        ### add a fc layer to map the graph representation to the output (from embedding_dim to 256)
        self.fc = nn.Linear(feat_dim, out_dim, bias=False)

        # for masked node modeling task
        self.graph_proj = nn.Linear(config['embed_dim'], config['embed_dim'])
        self.gtmae_decoder = MLP_dec2d(
            input_dim=config['embed_dim'],
            hidden_dim=512,
            num_layers=3,
            output_dim=config['feat_dim'],
            activation="relu",
            norm="batchnorm"
        )

        self.encoder_to_decoder = nn.Linear(config['embed_dim'], config['embed_dim'], bias=False)
        self.gsm_head = nn.Linear(config['embed_dim'], 2)
        self.loss_gtmae = setup_loss_fn("mse")
        self.loss_cl = TripletLoss(margin=1.0)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.embed_out.reset_parameters()

    def graph_encode(self, graph_data, mask=None):
        (attn_mask, node_feat, in_degree, out_degree, path_data, dist) = graph_data
        num_graphs, max_num_nodes, _ = node_feat.shape
        deg_emb = self.degree_encoder(torch.stack((in_degree, out_degree)))

        if self.encoding:
            node_feat_comb = node_feat + deg_emb
            graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(
                num_graphs, 1, 1
            )
            x = torch.cat([graph_token_feat, node_feat_comb], dim=1)

        attn_bias = torch.zeros(
            num_graphs,
            max_num_nodes + 1,
            max_num_nodes + 1,
            self.num_heads,
            device=dist.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

        t = self.graph_token_virtual_distance.weight.reshape(
            1, 1, self.num_heads
        )

        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t

        if (mask is not None) and self.encoding:
            x = x[:, ~mask]  # [n graph, n non-masked nodes, n hidden]
            attn_bias = attn_bias[:, ~mask, :, :][:, :, ~mask,
                        :]  # [n graph, n non-masked nodes, n non-masked nodes, n heads,]
            attn_mask = attn_mask[:, ~mask, :][:, :, ~mask]

        x = self.emb_layer_norm(x)

        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        graph_rep = x[:, 0, :]
        node_rep = x[:, 1:, :]

        node_rep = self.embed_out(node_rep)
        graph_rep = self.embed_out(graph_rep) + self.lm_output_learned_bias

        return node_rep, graph_rep

    def mask_attr_prediction(self, batched_data, mask):
        (attn_mask, node_feat, in_degree, out_degree, path_data, dist) = batched_data

        output, graph_emb = self.graph_encode(batched_data, mask)
        output = self.encoder_to_decoder(output)

        mask = mask[1:]  ## remove the first bit of mask tensor
        gold = node_feat[:, mask]
        num_masked = mask.nonzero().view(-1).shape[0]  # real masked size

        mask_token_expanded = self.mask_token.clone().expand(output.shape[0], num_masked, output.shape[2])
        dec_input = torch.cat([output, mask_token_expanded], dim=1)

        dec_outputs = self.gtmae_decoder(dec_input)
        dec_outputs = dec_outputs[:, -num_masked:, :]

        return dec_outputs, gold, graph_emb

    def mlm_forward(self, graph_data):
        num_nodes = graph_data[1].size(1)
        mask_rate = self.config['mask_rate']
        num_mask = int(mask_rate * num_nodes)
        mask = np.hstack([
            np.zeros(num_nodes - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(mask)
        mask = torch.Tensor(mask).bool()

        ### add Fasle at the beginning of mask tensor
        mask = torch.cat([torch.tensor([False]), mask])

        x_rec_masked, x_init_mased, graph_emb = self.mask_attr_prediction(graph_data, mask)
        x_init = x_init_mased.view(-1)
        x_rec = x_rec_masked.contiguous().view(-1)  # [n graphs, n nodes, n feature]->[n graphs * n nodes,  n feature]

        if x_init.shape[0] == 0:
            loss = torch.tensor(0.0)
            loss_item = {"loss": loss.item()}
            print("No masked nodes")
            input()
            return loss, loss_item, graph_emb

        loss_gtmae = self.loss_gtmae(x_rec, x_init)

        return loss_gtmae

    def forward(self, ori_data, pos_data=None, neg_data=None):

        node_embeds_ori, graph_embs_ori = self.graph_encode(ori_data)

        # task 1: node mask prediction
        loss_mlm = self.mlm_forward(ori_data)

        # task 2: contrastive
        if pos_data is not None and neg_data is not None:
            _, graph_embs_pos = self.graph_encode(pos_data)
            _, graph_embs_neg = self.graph_encode(neg_data)

            loss_cl = self.loss_cl(graph_embs_ori, graph_embs_pos, graph_embs_neg)
        else:
            loss_cl = torch.tensor(0, device=ori_data.device)

        return node_embeds_ori, graph_embs_ori, loss_mlm, loss_cl


