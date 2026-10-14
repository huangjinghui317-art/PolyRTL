# coding=utf-8
import torch
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F
import numpy as np
from torch import nn
from scipy.stats import kendalltau, spearmanr


class CircuitDataset(Dataset):
    def __init__(self, file_path):

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.design_list = data['design_list']

        # graph and summary index: [num_data, max_node_length]: contain 0[pad], 1[cls]
        self.data_index = torch.tensor(data['data_index'])

        # labels
        self.wns = torch.tensor(data['wns'])
        self.tns = torch.tensor(data['tns'])
        self.area = torch.tensor(data['area'])
        self.power = torch.tensor(data['power'])
        self.slack = torch.tensor(data['slack'])

        # reg features
        self.bit_width = torch.tensor(data['bit_width'])
        self.in_degree = torch.tensor(data['in_degree'])
        self.out_degree = torch.tensor(data['out_degree'])
        self.max_path = torch.tensor(data['max_path'])
        self.min_path = torch.tensor(data['min_path'])
        self.density = torch.tensor(data['density'])

        # rig connection
        self.graph_edges = torch.tensor(data['graph_edges'])
        self.edge_type = torch.tensor(data['edge_type'])

        # features: [num_graph, dim]
        self.graph_features = data['graph_features']
        self.summary_features = data['summary_features']
        self.pad_token_id = 0
        self.cls_token_id = 1
        # 0[pad] 1[cls]

    def __len__(self):
        return self.data_index.size(0)

    def __getitem__(self, idx):
        # 0[pad] 1[cls]
        index = self.data_index[idx]

        bit_width = self.bit_width[idx][1:]
        density = self.density[idx][1:].float()
        max_path = self.max_path[idx][1:].float()

        label_dict = {
            'wns': self.wns[idx],
            'tns': self.tns[idx],
            'area': self.area[idx],
            'power': self.power[idx],
            'slack': self.slack[idx]
        }

        graph_edges = self.graph_edges[idx]
        edge_types = self.edge_type[idx].long()

        return index, graph_edges, edge_types, bit_width, density, max_path, label_dict


def mask_tokens(inputs, mask_prob=0.15):
    pad_token_id = 0
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    target_masked = inputs.clone()
    maskable = (inputs != pad_token_id) & (inputs != cls_token_id) & (inputs != sep_token_id)
    input_mask = torch.full(inputs.shape, fill_value=pad_token_id, device=inputs.device)
    mask = torch.bernoulli(torch.full(inputs.shape, mask_prob).to(inputs.device) * maskable).bool()
    mask_indices = torch.multinomial(maskable.float(), num_samples=1, replacement=False).to(inputs.device)
    mask = torch.scatter(mask, dim=1, index=mask_indices,
                         src=torch.ones(mask_indices.size(), dtype=torch.bool, device=inputs.device))
    input_mask[mask] = mask_token_id
    input_mask[~mask] = inputs[~mask]
    target_masked = inputs.clone()
    target_masked[~mask] = pad_token_id
    return input_mask, inputs.clone(), target_masked


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask


def shuffle_tensor(tensor):
    original_indices = torch.arange(tensor.size(0), device=tensor.device)
    while True:
        indices = np.random.permutation(tensor.size(0))
        shuffled_indices = torch.tensor(indices).to(tensor.device)
        if not torch.any(shuffled_indices == original_indices):
            break
    shuffled_tensor = tensor[shuffled_indices]

    return shuffled_tensor


def mean_squared_error(y_true, y_pred):
    mse = torch.sum((y_true - y_pred) ** 2)
    return mse.item()


def mean_absolute_error(y_true, y_pred):
    mae = torch.sum(torch.abs(y_true - y_pred))
    return mae.item()


def mean_absolute_relative_error(y_true, y_pred):
    mask = y_true != 0
    abs_relative_error = torch.abs((y_true[mask] - y_pred[mask]))
    ground_truth = y_true[mask]
    abs_relative_error = torch.sum(abs_relative_error)
    ground_truth = torch.sum(ground_truth)
    return abs_relative_error.item(), ground_truth.item()


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    abs_percentage_error = torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100.0
    mape = torch.sum(abs_percentage_error)
    return mape.item()


def kendall_tau_torch(y_true, y_pred):
    tau = []
    for i in range(y_true.shape[0]):
        tau.append(kendalltau(y_true[i].cpu().numpy(), y_pred[i].cpu().numpy())[0])
    return torch.tensor(tau).sum()


def spearman_r_torch(y_true, y_pred):
    rho = []
    for i in range(y_true.shape[0]):
        rho.append(spearmanr(y_true[i].cpu().numpy(), y_pred[i].cpu().numpy())[0])
    return torch.tensor(rho).sum()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        positive_similarity = F.cosine_similarity(anchor, positive).unsqueeze(1)

        negative_similarity = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1)

        similarities = torch.cat([positive_similarity, negative_similarity], dim=1)

        probabilities = F.softmax(similarities / self.temperature, dim=1)

        positive_probabilities = probabilities[:, 0]

        loss = -torch.log(positive_probabilities).mean()
        return loss
