import torch
from torch import nn
from .Transformer import Topoformer
from .TransferGNN import TransferGNN
from torch import nn
import torch.nn.functional as F
from icecream import ic


class TopoRTL(nn.Module):
    def __init__(self, graph_features=None, graph_nhead=4, graph_num_encoder_layers=5, graph_dim_feedforward=256,
                 graph_embedding_size=768, graph_dropout=0.1, graph_max_len=420,
                 summary_features=None, summary_nhead=4, summary_num_encoder_layers=5, summary_dim_feedforward=256,
                 summary_embedding_size=768, summary_dropout=0.1, summary_max_len=420,
                 nhid=256, predict=False, opt=None):
        super(TopoRTL, self).__init__()

        self.opt = opt
        self.graph_features = graph_features
        self.summary_features = summary_features

        graph_embedding_size = graph_embedding_size
        summary_embedding_size = summary_embedding_size
        self.graph_reduce_size = nhid
        self.summary_reduce_size = nhid

        self.graph_model = Topoformer(
            graph_embedding_size, self.graph_reduce_size, graph_nhead, graph_num_encoder_layers,
            graph_dim_feedforward, graph_dropout, graph_max_len)
        self.summary_model = Topoformer(
            summary_embedding_size, self.summary_reduce_size, summary_nhead, summary_num_encoder_layers,
            summary_dim_feedforward, summary_dropout, summary_max_len)

        self.struc_fc1 = nn.Linear(self.graph_reduce_size + self.summary_reduce_size,
                                   (self.graph_reduce_size + self.summary_reduce_size) // 2
                                   )
        self.struc_fc2 = nn.Linear(self.graph_reduce_size + self.summary_reduce_size,
                                   (self.graph_reduce_size + self.summary_reduce_size) // 2
                                   )

        self.func_graph = TransferGNN(in_dim=self.graph_reduce_size + self.summary_reduce_size,
                                      message_dim=self.graph_reduce_size + self.summary_reduce_size,
                                      num_layers=3, opt=opt)

        self.predict = predict
        self.graph_max_len = graph_max_len

    def set_graph_features(self, feats: torch.Tensor):
        self.graph_features = feats

    def set_summary_features(self, feats: torch.Tensor):
        self.summary_features = feats

    def process_position(self, max_path, density):
        '''
        :param max_path: [batch_size, max_nodes]
        :param density: [batch_size, max_nodes]
        :return: [batch_size, max_nodes, max_nodes]
        '''
        path_relative_matrix = None
        density_relative_matrix = None
        if max_path is not None:
            max_path = torch.clamp(max_path, min=0, max=128)
            path_difference_matrix = max_path.unsqueeze(dim=2) - max_path.unsqueeze(dim=1)
            path_relative_matrix = torch.abs(path_difference_matrix).to(max_path.device)
        if density is not None:
            density = density * 100
            density_difference_matrix = density.unsqueeze(dim=2) - density.unsqueeze(dim=1)
            density_relative_matrix = torch.abs(density_difference_matrix).to(density.device)
        return path_relative_matrix, density_relative_matrix

    def forward(self, index, graph_edges, edge_type=None, bit_width=None, max_path=None, density=None):
        if self.opt.wo_bit_width:
            bit_width = None
        if self.opt.wo_density:
            density = None
        if self.opt.wo_max_path:
            max_path = None

        path_relative_matrix, density_relative_matrix = self.process_position(max_path, density)

        graph, graph_emb = self.graph_model(index, self.graph_features,
                                            bit_centrality=bit_width,
                                            path_spatial=path_relative_matrix,
                                            density_spatial=density_relative_matrix)
        # graph/graph_emb: [batch_size, max_graph_length, graph_dim]

        summary, summary_emb = self.summary_model(index, self.summary_features,
                                                  bit_centrality=bit_width,
                                                  path_spatial=path_relative_matrix,
                                                  density_spatial=density_relative_matrix,)
        # summary/summary_emb: [batch_size, max_summary_length, summary_dim]

        if self.predict:
            x_output = torch.cat((graph_emb, summary_emb), dim=-1)
            x = x_output.mean(1)

            return x_output, x, graph, graph_emb, summary, summary_emb
        else:
            y_output = torch.cat((graph_emb, summary), dim=-1)
            y = self.struc_fc1(y_output.mean(1))

            z_output = torch.cat((graph, summary_emb), dim=-1)
            z = self.struc_fc2(z_output.mean(1))

            return graph, graph_emb, summary, summary_emb, y, z


