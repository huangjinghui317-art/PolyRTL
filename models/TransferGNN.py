# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv, GINConv, RGCNConv
import argparse


def merge_graphs(x, edge_index, edge_type):
    batch_size, num_nodes, num_features = x.shape
    _, _, num_edges = edge_index.shape
    merged_x = x.reshape(batch_size * num_nodes, num_features)
    row, col = edge_index[:, 0, :], edge_index[:, 1, :]
    row_offset = torch.arange(batch_size, device=edge_index.device).repeat_interleave(num_edges) * num_nodes
    merged_row = (row.reshape(-1) + row_offset).reshape(-1)
    merged_col = (col.reshape(-1) + row_offset).reshape(-1)
    merged_edge_index = torch.stack([merged_row, merged_col], dim=0)
    merged_edge_type = edge_type.reshape(batch_size * num_edges)
    return merged_x, merged_edge_index, merged_edge_type


def unmerge_graphs(merged_x, batch_size, num_nodes, num_features):
    return merged_x.reshape(batch_size, num_nodes, num_features)


class StreamConv(MessagePassing):
    def __init__(self, feature_dim: int, message_dim: int):
        super(StreamConv, self).__init__(aggr='add')
        self.lin = nn.Linear(feature_dim, message_dim)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return self.lin(aggr_out)


class TransferGNNConv(nn.Module):
    def __init__(self, feature_dim: int, message_dim: int, opt):
        super(TransferGNNConv, self).__init__()
        self.opt = opt
        self.feature_dim = feature_dim
        self.message_dim = message_dim

        self.data_conv = StreamConv(feature_dim, message_dim)
        self.control_conv = StreamConv(feature_dim, message_dim)

        self.gate_nn = nn.GRUCell(message_dim, message_dim)
        self.update_func = nn.Linear(message_dim + feature_dim, feature_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): node feature [num_nodes, feature_dim]
            edge_index (Tensor): total edges [2, num_edges]
            edge_type (Tensor): edge type [num_edges]，0=data, 1=control, -1=padding
        """

        data_mask = (edge_type == 0)
        control_mask = (edge_type == 1)

        data_edge_index = edge_index[:, data_mask]
        control_edge_index = edge_index[:, control_mask]

        msg_data = self.data_conv(x, data_edge_index)  # [num_nodes, message_dim]
        msg_control = self.control_conv(x, control_edge_index)  # [num_nodes, message_dim]

        fused_message = self.gate_nn(msg_control, msg_data)

        x_new = self.update_func(torch.cat((fused_message, x), dim=-1))

        return x_new

    def __repr__(self):
        return (f'{self.__class__.__name__}(feature_dim={self.feature_dim}, '
                f'message_dim={self.message_dim})')


class TransferGNN(nn.Module):
    def __init__(self, in_dim: int, message_dim: int, num_layers: int, opt):
        super(TransferGNN, self).__init__()
        self.opt = opt
        self.in_dim = in_dim
        self.num_layer = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = TransferGNNConv(feature_dim=in_dim, message_dim=message_dim, opt=opt)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        edge_index = edge_index.transpose(1, 2)
        batch_size, num_nodes, _ = x.shape
        merged_x, merged_edge_index, merged_edge_type = merge_graphs(x, edge_index, edge_type)
        x = merged_x
        for i, layer in enumerate(self.layers):
            x = layer(x, merged_edge_index, merged_edge_type)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        x = unmerge_graphs(x, batch_size, num_nodes, self.in_dim)
        return x
