#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-branch TopoRTL: EVERY SSL epoch -> JOINT 5-task MLP -> VAL -> TEST.

Run from the official TopoRTL repository root. Requires the official
models/TopoRTL.py, utils/utils.py and preprocessed data/. No PyG installation
is NOT needed: graph message passing uses native PyTorch scatter/index_add.

Scheme 1 (3 x 128 = 384 graph dims): global, central, plain.
Scheme 2 (8 x 48 = 384 graph dims): global, plain, fanout, fanin,
                                  chain, junction, depth, reverse.
The unchanged official Summary Topoformer emits 384 dims. Final per-token
representation is graph-concat(384) || summary(384) = 768 dims.

Terminology caveat: role masks are *heuristics on register-interaction graphs*;
"depth" is graph-path depth, not an STA-derived timing critical path.


mkdir -p logs

python train.py \
  --stage all \
  --scheme scheme2 \
  --data-root data \
  --split-ratio 0.3-0.3-0.4 \
  --epochs 50 \
  --batch-size 2 \
  --gnn-layers 3 \
  --ssl-lr 1e-4 \
  --mlp-epochs 100 \
  --mlp-batch-size 8 \
  --mlp-lr 1e-3 \
  --mlp-val-every 1 \
  --mlp-log-every 10 \
  --output results/scheme2 \
  2>&1 | tee results/train.log
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from models.TopoRTL import TopoRTL
    from utils.utils import CircuitDataset
except ImportError as e:
    raise SystemExit(
        'Run this file from the BUPT-GAMMA/TopoRTL repository root. '
        'Install the official dependencies first. Original import error: ' + str(e)
    )

TASKS = ('area', 'power', 'wns', 'tns', 'slack')
BRANCHES = {
    'scheme1': ('global', 'central', 'plain'),
    'scheme2': ('global', 'plain', 'fanout', 'fanin',
                'chain', 'junction', 'depth', 'reverse'),
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path: Path, obj):
    # Replace atomically so a running experiment never exposes half a JSON file.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def save_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def resolve_device(device):
    d = torch.device(device)
    if d.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'--device={device} specified but CUDA is not available')
    return d


def prefix_sum_degrees(src, dst, size):
    # In/out are calculated from valid RIG edges only, without virtual CLS links.
    ind = torch.bincount(dst, minlength=size)
    outd = torch.bincount(src, minlength=size)
    return ind, outd


def top_per_graph(degree: torch.Tensor, real_mask: torch.Tensor,
                  batch_size: int, nodes_per_graph: int, fraction=.20):
    selected = torch.zeros_like(real_mask)
    deg = degree.view(batch_size, nodes_per_graph)
    real = real_mask.view(batch_size, nodes_per_graph)
    sel = selected.view(batch_size, nodes_per_graph)
    for b in range(batch_size):
        choices = real[b].nonzero(as_tuple=True)[0]
        if len(choices) == 0:
            continue
        k = max(1, math.ceil(len(choices) * fraction))
        selected_nodes = choices[torch.topk(deg[b, choices], k).indices]
        sel[b, selected_nodes] = True
    return selected


def high_path_per_graph(path: torch.Tensor, real_mask: torch.Tensor,
                        batch_size: int, nodes_per_graph: int):
    # High max_path means deep in the RIG. It is NOT an STA critical path.
    selected = torch.zeros_like(real_mask)
    p = path.view(batch_size, nodes_per_graph)
    real = real_mask.view(batch_size, nodes_per_graph)
    sel = selected.view(batch_size, nodes_per_graph)
    for b in range(batch_size):
        choices = real[b].nonzero(as_tuple=True)[0]
        if len(choices) == 0:
            continue
        k = max(1, math.ceil(len(choices) * .25))
        deep_nodes = choices[torch.topk(p[b, choices], k).indices]
        sel[b, deep_nodes] = True
    return selected


class SparseMessageBlock(nn.Module):
    """Distinct trainable MPNN block, no torch_geometric dependency.

    operator='attn' uses per-destination edge-softmax attention;
    operator='gin' uses sum aggregation with learned epsilon;
    operator='mean' uses GraphSAGE-like mean aggregation.
    """
    def __init__(self, d: int, operator: str, dropout=.20):
        super().__init__()
        assert operator in ('attn', 'gin', 'mean')
        self.operator = operator
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.value = nn.Linear(d, d, bias=False)
        self.edge = nn.Linear(6, d, bias=False)
        if operator == 'attn':
            self.query = nn.Linear(d, d, bias=False)
            self.key = nn.Linear(d, d, bias=False)
            self.attn_edge = nn.Linear(6, 1, bias=False)
        if operator == 'gin':
            self.eps = nn.Parameter(torch.zeros(1))
        self.update = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                    nn.Linear(2 * d, d))
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor):
        h = self.norm1(x)
        src, dst = edge_index[0], edge_index[1]
        num_nodes, d = h.shape
        messages = self.value(h[src]) + self.edge(edge_attr)
        aggregate = h.new_zeros((num_nodes, d))
        if self.operator == 'attn':
            q = self.query(h[dst])
            k = self.key(h[src])
            logits = (q * k).sum(-1) / math.sqrt(d)
            logits = logits + self.attn_edge(edge_attr).flatten()
            # Detach only the softmax max for numerical stability; probabilities
            # retain gradient with respect to logits.
            max_per_dst = torch.full((num_nodes,), -float('inf'),
                                     device=x.device, dtype=logits.dtype)
            max_per_dst.scatter_reduce_(0, dst, logits.detach(),
                                        reduce='amax', include_self=True)
            exp = (logits - max_per_dst[dst]).exp()
            denom = exp.new_zeros(num_nodes)
            denom.index_add_(0, dst, exp)
            attention = exp / denom[dst].clamp_min(1e-12)
            aggregate.index_add_(0, dst, messages * attention.unsqueeze(-1))
            update = self.update(aggregate)
        else:
            aggregate.index_add_(0, dst, messages)
            if self.operator == 'mean':
                counts = h.new_zeros(num_nodes)
                counts.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
                aggregate = aggregate / counts.clamp_min(1).unsqueeze(-1)
                update = self.update(aggregate)
            else:
                update = self.update((1 + self.eps) * h + aggregate)
        x = x + F.dropout(update, p=self.dropout, training=self.training)
        x = x + F.dropout(self.ffn(self.norm2(x)),
                          p=self.dropout, training=self.training)
        return x


class SpecializedBranch(nn.Module):
    def __init__(self, in_dim, branch_dim, operator, layers, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, branch_dim)
        self.layers = nn.ModuleList([
            SparseMessageBlock(branch_dim, operator, dropout)
            for _ in range(layers)
        ])
        self.out_norm = nn.LayerNorm(branch_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return self.out_norm(x)


class MultiGNNGraphEncoder(nn.Module):
    def __init__(self, original, scheme, layers=3, dropout=.2):
        super().__init__()
        self.scheme = scheme
        self.names = BRANCHES[scheme]
        self.branch_dim = 384 // len(self.names)
        assert len(self.names) * self.branch_dim == 384
        # These parameters are the original TopoRTL input/position encoders.
        self.embedding = original.embedding
        self.bit_width_encoder = original.bit_width_encoder
        self.max_path_encoder = original.max_path_encoder
        self.density_encoder = original.density_encoder
        self.virtual_distance = original.virtual_distance
        self.emb_layer_norm = original.emb_layer_norm
        self.path_weight = original.path_weight
        self.density_weight = original.density_weight
        operators = {
            'global': 'attn', 'central': 'attn', 'plain': 'gin',
            'fanout': 'gin', 'fanin': 'mean', 'chain': 'mean',
            'junction': 'attn', 'depth': 'gin', 'reverse': 'mean',
        }
        self.branches = nn.ModuleDict({
            name: SpecializedBranch(384, self.branch_dim, operators[name],
                                    layers, dropout)
            for name in self.names
        })

    def _topology(self, index, edges, types, path):
        b, n = index.shape
        # Important: 0 is PAD; 1 is CLS; >1 are real RIG register nodes.
        flat_real = (index.reshape(-1) > 1)
        valid = (types >= 0) & (types <= 1)
        valid &= ((edges[..., 0] >= 1) & (edges[..., 0] < n) &
                  (edges[..., 1] >= 1) & (edges[..., 1] < n))
        eb = torch.arange(b, device=index.device).unsqueeze(1).expand_as(types)[valid]
        endpoints = edges[valid].long()
        src = endpoints[:, 0] + eb * n
        dst = endpoints[:, 1] + eb * n
        real_edge = flat_real[src] & flat_real[dst]
        eb, endpoints, src, dst = eb[real_edge], endpoints[real_edge], src[real_edge], dst[real_edge]
        selected_types = types[valid][real_edge]
        ind, outd = prefix_sum_degrees(src, dst, b * n)
        degree = ind + outd
        center = top_per_graph(degree, flat_real, b, n)
        # 'path' arrives without CLS in official CircuitDataset.__getitem__.
        padded_path = F.pad(path.float(), (1, 0)).reshape(-1)
        deep = high_path_per_graph(padded_path, flat_real, b, n)
        roles = {
            'global': flat_real, 'plain': flat_real, 'central': center,
            'fanout': flat_real & (outd >= 2),
            'fanin': flat_real & (ind >= 2),
            'chain': flat_real & (outd == 1) & (ind == 1),
            'junction': flat_real & (ind >= 2) & (outd >= 2),
            'depth': deep, 'reverse': flat_real,
        }
        masks = {
            'global': torch.ones(src.numel(), dtype=torch.bool, device=src.device),
            'plain': torch.ones(src.numel(), dtype=torch.bool, device=src.device),
            'central': center[src] | center[dst],
            'fanout': outd[src] >= 2,
            'fanin': ind[dst] >= 2,
            'chain': (outd[src] == 1) & (ind[dst] == 1),
            'junction': (outd[src] >= 2) & (ind[dst] >= 2),
            'depth': deep[src] | deep[dst],
            'reverse': torch.ones(src.numel(), dtype=torch.bool, device=src.device),
        }
        # On small graphs, strict branch->merge edges can be absent.
        # Loosen the junction mask to incident fan-out/merge edges.
        if 'junction' in self.names and not masks['junction'].any():
            masks['junction'] = (outd[src] >= 2) | (ind[dst] >= 2)
            roles['junction'] = roles['fanout'] | roles['fanin']
        return eb, endpoints, src, dst, selected_types, roles, masks

    def _batch_edges(self, name, index, eb, endpoints, src, dst, etype,
                     roles, masks, path, density):
        b, n = index.shape
        chosen = masks[name]
        u, v = src[chosen], dst[chosen]
        # Explicit directed reverse branch, unlike the remaining operators.
        if name == 'reverse':
            u, v = v, u
        local_source = endpoints[chosen, 0]
        local_target = endpoints[chosen, 1]
        batch_id = eb[chosen]
        path_pad = F.pad(path.float().clamp(0, 128), (1, 0))
        density_pad = F.pad(density.float() * 100.0, (1, 0))
        dp = (path_pad[batch_id, local_source] -
              path_pad[batch_id, local_target]).abs().unsqueeze(-1)
        dd = (density_pad[batch_id, local_source] -
              density_pad[batch_id, local_target]).abs().unsqueeze(-1)
        # TopoRTL encodes RIG path/density differences into 4 attention heads.
        bias = self.path_weight * self.max_path_encoder(dp)
        bias = bias + self.density_weight * self.density_encoder(dd)
        relation = F.one_hot(etype[chosen].long(), num_classes=2).to(bias.dtype)
        real_attr = torch.cat([bias, relation], -1)  # [E, 4+2]

        flat_real = index.reshape(-1) > 1
        flat_id = torch.arange(b * n, device=index.device)
        graph_no = torch.div(flat_id, n, rounding_mode='floor')
        cls_id = graph_no * n
        # CLS transmits to PAD/real tokens; PAD never transmits to CLS.
        from_cls_u, from_cls_v = cls_id, flat_id
        selected_real = roles[name] & flat_real
        # Ensure that each design sends at least one real node to CLS.
        if selected_real.any():
            count = torch.bincount(graph_no[selected_real], minlength=b)
        else:
            count = torch.zeros(b, dtype=torch.long, device=index.device)
        selected_real |= (flat_real & (count[graph_no] == 0))
        to_cls_u = flat_id[selected_real]
        to_cls_v = cls_id[selected_real]
        final_u = torch.cat([u, from_cls_u, to_cls_u])
        final_v = torch.cat([v, from_cls_v, to_cls_v])
        num_virtual = from_cls_u.numel() + to_cls_u.numel()
        virtual = self.virtual_distance.weight.expand(num_virtual, -1)
        virtual_attr = torch.cat([virtual, virtual.new_zeros(num_virtual, 2)], -1)
        all_attr = torch.cat([real_attr, virtual_attr], 0)
        return torch.stack([final_u, final_v], 0), all_attr

    def forward(self, index, extern_feats, graph_edges, edge_type,
                bit_width, max_path, density):
        b, n = index.shape
        x = self.embedding(index, extern_feats) * math.sqrt(384)
        centrality = x.new_zeros(x.shape)
        if bit_width is not None:
            centrality[:, 1:] = self.bit_width_encoder(bit_width.long())
        initial = x + centrality  # exactly the official pre-topoformer input
        h = self.emb_layer_norm(initial).reshape(b * n, 384)
        eb, endpoints, src, dst, etype, roles, masks = self._topology(
            index, graph_edges, edge_type, max_path)
        outputs = []
        for name in self.names:
            ei, ea = self._batch_edges(name, index, eb, endpoints, src,
                                       dst, etype, roles, masks, max_path, density)
            outputs.append(self.branches[name](h, ei, ea))
        graph_concatenated = torch.cat(outputs, -1).reshape(b, n, 384)
        return initial, graph_concatenated


class MultiGNNTopoRTL(TopoRTL):
    def __init__(self, scheme, gn_layers, opt):
        super().__init__(
            graph_nhead=4, graph_num_encoder_layers=3, graph_dropout=opt.dropout,
            graph_embedding_size=768, graph_max_len=2000,
            graph_dim_feedforward=768,
            summary_nhead=4, summary_num_encoder_layers=3,
            summary_dropout=opt.dropout, summary_embedding_size=768,
            summary_max_len=2000, summary_dim_feedforward=768,
            nhid=384, predict=False,
            opt=SimpleNamespace(wo_bit_width=False, wo_density=False,
                                wo_max_path=False)
        )
        old = self.graph_model
        self.graph_model = MultiGNNGraphEncoder(old, scheme, gn_layers,
                                                 opt.dropout)
        self.predict = False

    def forward(self, index, graph_edges, edge_type=None, bit_width=None,
                max_path=None, density=None):
        # Keep original signature/order; official DataLoader returns density
        # BEFORE max_path, so unpack correctly at the caller.
        path_relative, density_relative = self.process_position(max_path, density)
        graph, graph_emb = self.graph_model(
            index, self.graph_features, graph_edges, edge_type,
            bit_width, max_path, density)
        summary, summary_emb = self.summary_model(
            index, self.summary_features, bit_centrality=bit_width,
            path_spatial=path_relative, density_spatial=density_relative)
        if self.predict:
            tokens = torch.cat([graph_emb, summary_emb], -1)  # 768
            return tokens, tokens.mean(1), graph, graph_emb, summary, summary_emb
        y = self.struc_fc1(torch.cat([graph_emb, summary], -1).mean(1))
        z = self.struc_fc2(torch.cat([graph, summary_emb], -1).mean(1))
        return graph, graph_emb, summary, summary_emb, y, z


def new_model(args, predict=False):
    m = MultiGNNTopoRTL(args.scheme, args.gnn_layers, args)
    m.predict = predict
    return m


def attach_feats(m, ds, device):
    # Official CircuitDataset embeds a distinct per-split feature table.
    m.set_graph_features(ds.graph_features.to(device))
    m.set_summary_features(ds.summary_features.to(device))


def get_dataset(args, split):
    path = Path(args.data_root) / 'processed' / args.split_ratio / split / (split + '.pkl')
    if not path.exists():
        raise FileNotFoundError(f'Expected preprocessed TopoRTL dataset: {path}')
    return CircuitDataset(str(path))


def unpack(batch, device):
    idx, edges, types, width, density, path, labels = batch
    return (idx.to(device), edges.to(device), types.to(device),
            width.to(device), path.to(device), density.to(device), labels)


def model_path(root, epoch):
    return root / 'checkpoints' / f'ssl_epoch_{epoch:03d}.pt'


def load_model(root, epoch, args, device):
    file = model_path(root, epoch)
    if not file.exists():
        raise FileNotFoundError(f'Missing SSL checkpoint: {file}')
    ckpt = torch.load(file, map_location='cpu', weights_only=False)
    if ckpt['scheme'] != args.scheme:
        raise ValueError(f'Checkpoint scheme {ckpt["scheme"]} != {args.scheme}')
    if ckpt.get('gnn_layers', args.gnn_layers) != args.gnn_layers:
        raise ValueError('--gnn-layers differs from checkpoint')
    model = new_model(args, predict=True).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model



def selected_epochs(args, root):
    all_paths = sorted((root / 'checkpoints').glob('ssl_epoch_*.pt'))
    epochs = [int(p.stem.split('_')[-1]) for p in all_paths]
    if not epochs:
        raise FileNotFoundError(f'No SSL checkpoints under {root / "checkpoints"}')
    if args.eval_epochs == 'last':
        return [epochs[-1]]
    if args.eval_epochs == 'all':
        return epochs
    requested = [int(s.strip()) for s in args.eval_epochs.split(',') if s.strip()]
    absent = sorted(set(requested) - set(epochs))
    if absent:
        raise FileNotFoundError(f'--eval-epochs missing SSL checkpoints: {absent}')
    return sorted(set(requested))


@torch.inference_mode()
def extract_for_split(m, ds, split, out_dir, device, skip_existing=False):
    attach_feats(m, ds, device)
    m.eval()
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    for i, batch in enumerate(tqdm(loader, desc=f'Extract {split}', leave=False)):
        name = str(ds.design_list[i])
        path_out = out_dir / split / f'{name}.pkl'
        if skip_existing and path_out.exists():
            continue
        idx, edges, types, width, path, density, labels = unpack(batch, device)
        tokens, *_ = m(idx, edges, types, width, path, density)
        real = idx[0] > 1
        local = tokens[0, real].float().cpu().numpy()
        y_slack = labels['slack'][0, real.cpu()].float().cpu().numpy()
        data = {
            'global_emb': tokens[0, 0].float().cpu().numpy()[None, :],
            'local_emb': local,
            'slack': y_slack,
            'area': float(labels['area'].flatten()[0]),
            'power': float(labels['power'].flatten()[0]),
            'wns': float(labels['wns'].flatten()[0]),
            'tns': float(labels['tns'].flatten()[0]),
        }
        path_out.parent.mkdir(parents=True, exist_ok=True)
        with path_out.open('wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_extract(args, root, device):
    for epoch in selected_epochs(args, root):
        out_dir = root / 'embeddings' / f'epoch_{epoch:03d}'
        model = load_model(root, epoch, args, device)
        for split in ('train', 'valid', 'test'):
            ds = get_dataset(args, split)
            extract_for_split(model, ds, split, out_dir, device,
                              skip_existing=args.skip_existing)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def signed_log1p(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.expm1(np.abs(x))


def metrics(y, pred):
    y, pred = np.asarray(y).reshape(-1), np.asarray(pred).reshape(-1)
    keep = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[keep], pred[keep]
    if y.size == 0:
        raise ValueError('No valid evaluation labels')
    mse = float(np.mean((y - pred) ** 2))
    mae = float(np.mean(np.abs(y - pred)))
    den = float(np.sum((y - np.mean(y)) ** 2))
    if y.size >= 2 and den > 1e-12:
        pcc = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-12 else 0.0
        r2 = float(1.0 - np.sum((y - pred) ** 2) / den)
        rrse = float(np.sqrt(np.sum((y - pred) ** 2) / den))
    else:
        pcc, r2, rrse = 0.0, 0.0, 0.0
    mape_cap = float(100.0 * np.mean(np.clip(
        np.abs(y - pred) / (np.abs(y) + 1e-4), 0, 1)))
    return dict(n=int(y.size), pcc=pcc, r2=r2, mae=mae,
                rmse=float(np.sqrt(mse)), rrse=rrse,
                mape_capped_pct=mape_cap,
                custom_score=abs(pcc) + 1.0 - mape_cap / 100.0)



# ---------------- JOINT FIVE-TASK PPA HEAD ----------------
# Unlike the five separately trained official-style heads, a single optimizer
# trains ONE shared trunk and FIVE heads in every MLP minibatch.
GLOBAL_TASKS = ('area', 'power', 'wns', 'tns')


@contextmanager
def preserve_rng():
    """MLP training cannot perturb the SSL data permutation/dropout RNG."""
    py = random.getstate()
    np_state = np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class JointPPADataset(Dataset):
    """One example = one design and ALL its 5 targets.

    Read one design at a time, rather than keeping all [nodes, 768] arrays in
    memory. The saved embedding format matches the original TopoRTL PPA script.
    """
    def __init__(self, path):
        self.files = sorted(Path(path).glob('*.pkl'))
        if not self.files:
            raise FileNotFoundError(f'No design embeddings under: {path}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, pos):
        with self.files[pos].open('rb') as f:
            record = pickle.load(f)
        g = torch.as_tensor(np.asarray(record['global_emb'], dtype=np.float32).reshape(-1))
        local = torch.as_tensor(np.asarray(record['local_emb'], dtype=np.float32).reshape(-1, 768))
        if g.numel() != 768 or local.shape[1] != 768:
            raise ValueError(f'Bad embedding dimensions: {self.files[pos]}')
        if not torch.isfinite(g).all() or not torch.isfinite(local).all():
            raise ValueError(f'Nonfinite input embeddings: {self.files[pos]}')
        if len(local):
            mean, maximum = local.mean(dim=0), local.max(dim=0).values
        else:
            mean = torch.zeros_like(g)
            maximum = torch.zeros_like(g)
        global_y = torch.tensor([record[t] for t in GLOBAL_TASKS], dtype=torch.float32)
        slack = torch.as_tensor(np.asarray(record['slack'], dtype=np.float32).reshape(-1))
        if slack.numel() != local.size(0):
            raise ValueError(f'Slack/node mismatch: {self.files[pos]}')
        if not len(local):
            # Empty graphs still contribute (where labels exist) to area/power.
            local = g.new_empty((0, 768))
        return g, mean, maximum, local, global_y, slack, self.files[pos].stem


def collate_joint(batch):
    g = torch.stack([r[0] for r in batch])
    mean = torch.stack([r[1] for r in batch])
    maximum = torch.stack([r[2] for r in batch])
    local = torch.cat([r[3] for r in batch], 0)
    glob_y = torch.stack([r[4] for r in batch])
    slack = torch.cat([r[5] for r in batch], 0)
    owner = torch.repeat_interleave(torch.arange(len(batch), dtype=torch.long),
                                    torch.tensor([len(r[3]) for r in batch]))
    return g, mean, maximum, local, glob_y, slack, owner


class JointMLP(nn.Module):
    """One shared feature trunk with simultaneous, task-specific heads.

    Area/Power consume the CLS embedding; WNS/TNS consume mean+max pooled
    register embeddings; Slack consumes each register embedding. Pooling is
    performed BEFORE the shared trunk, preserving official input conventions.
    """
    def __init__(self, hidden=512, dropout=0.0):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(768, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.circuit_heads = nn.ModuleDict({
            t: nn.Linear(hidden if t in ('area', 'power') else 2 * hidden, 1)
            for t in GLOBAL_TASKS
        })
        self.slack_head = nn.Linear(hidden, 1)

    def forward(self, g, mean, maximum, local):
        ng = len(g)
        vectors = torch.cat([g, mean, maximum, local], dim=0)
        projected = self.shared(vectors)
        hg = projected[:ng]
        hm = projected[ng:2 * ng]
        hx = projected[2 * ng:3 * ng]
        hl = projected[3 * ng:]
        path = torch.cat([hm, hx], dim=-1)
        out = {t: self.circuit_heads[t](hg if t in ('area', 'power') else path).flatten()
               for t in GLOBAL_TASKS}
        out['slack'] = self.slack_head(hl).flatten()
        return out


def compute_target_stats(train):
    """Fit target normalizers on TRAIN ONLY, using streaming sums per task."""
    sums = {t: [0.0, 0.0, 0] for t in TASKS}
    for i in tqdm(range(len(train)), desc='Train-only target statistics', leave=False):
        *_, glob_y, slack, _name = train[i]
        for j, t in enumerate(GLOBAL_TASKS):
            val = signed_log1p(float(glob_y[j]))
            if np.isfinite(val):
                sums[t][0] += float(val)
                sums[t][1] += float(val) ** 2
                sums[t][2] += 1
        valid = slack[torch.isfinite(slack)].numpy().astype(np.float64)
        val = signed_log1p(valid)
        sums['slack'][0] += float(np.sum(val))
        sums['slack'][1] += float(np.sum(val * val))
        sums['slack'][2] += int(val.size)
    stats = {}
    for t, (sm, squares, count) in sums.items():
        if count == 0:
            raise ValueError(f'No finite TRAIN labels for {t}')
        mu = sm / count
        std = max(float(np.sqrt(max(squares / count - mu * mu, 0.0))), 1e-3)
        stats[t] = {'mean': mu, 'std': std, 'count': count}
    return stats


def standard_target(y, task, stats, enabled):
    # Signed log1p aligns with official TopoRTL PPA convention.
    value = torch.sign(y) * torch.log1p(y.abs())
    if enabled:
        value = (value - stats[task]['mean']) / stats[task]['std']
    return value


def destandard_prediction(p, task, stats, enabled):
    if enabled:
        return p * stats[task]['std'] + stats[task]['mean']
    return p


def joint_loss(pred, global_y, slack, owner, stats, standardize, weights):
    individual = {}
    losses = []
    for i, task in enumerate(GLOBAL_TASKS):
        y = global_y[:, i]
        good = torch.isfinite(y) & torch.isfinite(pred[task])
        if not bool(good.any()):
            continue
        tgt = standard_target(y[good], task, stats, standardize)
        loss = F.mse_loss(pred[task][good], tgt)
        individual[task] = float(loss.detach())
        losses.append(weights[task] * loss)
    good = torch.isfinite(slack) & torch.isfinite(pred['slack'])
    if bool(good.any()):
        p = pred['slack'][good]
        y = standard_target(slack[good], 'slack', stats, standardize)
        group = owner[good]
        counts = torch.bincount(group, minlength=global_y.size(0))
        sums = p.new_zeros(global_y.size(0))
        sums.index_add_(0, group, (p - y).square())
        # One circuit = one slack contribution, so large circuits don't
        # overpower all other designs in the joint objective.
        slack_loss = (sums[counts > 0] / counts[counts > 0]).mean()
        individual['slack'] = float(slack_loss.detach())
        losses.append(weights['slack'] * slack_loss)
    if not losses:
        raise RuntimeError('Entire minibatch contains no finite PPA targets')
    return sum(losses) / sum(weights[t] for t in individual), individual


def joint_device_batch(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


@torch.inference_mode()
def joint_evaluate(head, loader, device, stats, standardize):
    """Evaluate all five tasks using one forward pass per minibatch."""
    head.eval()
    truth = {t: [] for t in TASKS}
    pred = {t: [] for t in TASKS}
    for batch in loader:
        *tensors, = joint_device_batch(batch, device)
        g, mean, maximum, local, global_y, slack, owner = tensors
        outputs = head(g, mean, maximum, local)
        for i, t in enumerate(GLOBAL_TASKS):
            y, p = global_y[:, i], outputs[t]
            good = torch.isfinite(y) & torch.isfinite(p)
            if bool(good.any()):
                actual = standard_target(y[good], t, stats, False)
                predicted = destandard_prediction(p[good], t, stats, standardize)
                truth[t].append(actual.float().cpu().numpy())
                pred[t].append(predicted.float().cpu().numpy())
        good = torch.isfinite(slack) & torch.isfinite(outputs['slack'])
        if bool(good.any()):
            actual = standard_target(slack[good], 'slack', stats, False)
            predicted = destandard_prediction(outputs['slack'][good], 'slack', stats, standardize)
            truth['slack'].append(actual.float().cpu().numpy())
            pred['slack'].append(predicted.float().cpu().numpy())
    evaluations = {}
    for t in TASKS:
        if not truth[t]:
            raise ValueError(f'No valid {t} targets in evaluation split')
        y = np.concatenate(truth[t]).astype(np.float64)
        p = np.concatenate(pred[t]).astype(np.float64)
        log_metrics = metrics(y, p)
        # Protect expm1 against huge invalid early MLP predictions.
        original_y = signed_expm1(np.clip(y, -60, 60))
        original_p = signed_expm1(np.clip(p, -60, 60))
        raw_metrics = metrics(original_y, original_p)
        evaluations[t] = {'log': log_metrics, 'raw': raw_metrics,
                          'y_log': y, 'p_log': p}
    return evaluations


def joint_macro_score(evaluation):
    """Validation-only model selection: equal weight for all 5 target types."""
    return float(np.mean([evaluation[t]['log']['custom_score'] for t in TASKS]))


def make_joint_loaders(root, epoch, args, device):
    base = root / 'embeddings' / f'epoch_{epoch:03d}'
    train, valid, test = (JointPPADataset(base / split)
                          for split in ('train', 'valid', 'test'))
    opts = {'collate_fn': collate_joint, 'num_workers': args.mlp_workers,
            'pin_memory': device.type == 'cuda'}
    rng = torch.Generator().manual_seed(args.seed + 100)
    tr = DataLoader(train, batch_size=args.mlp_batch_size, shuffle=True,
                    generator=rng, **opts)
    va = DataLoader(valid, batch_size=args.mlp_batch_size, shuffle=False, **opts)
    te = DataLoader(test, batch_size=args.mlp_batch_size, shuffle=False, **opts)
    return train, tr, va, te


def train_joint_one_ssl(args, root, device, epoch):
    """Cold-start a SINGLE five-task MLP for this particular SSL checkpoint."""
    directory = root / 'joint_mlp' / f'ssl_epoch_{epoch:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    train_ds, train_loader, valid_loader, test_loader = make_joint_loaders(root, epoch, args, device)
    stats = compute_target_stats(train_ds)
    save_json(directory / 'train_target_stats.json', stats)
    # Initial weights and DataLoader shuffle are identical across SSL epochs.
    seed_everything(args.seed + 7109)
    head = JointMLP(hidden=args.mlp_hidden, dropout=args.mlp_dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.mlp_lr,
                                  weight_decay=args.mlp_weight_decay)
    weights = {t: 1.0 for t in TASKS}
    best_score = -float('inf')
    best_mlp_epoch = 0
    history = []
    for me in range(1, args.mlp_epochs + 1):
        head.train()
        running, n = 0.0, 0
        for batch in train_loader:
            g, mean, maximum, local, gy, slack, owner = joint_device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            p = head(g, mean, maximum, local)
            loss, per_task = joint_loss(p, gy, slack, owner, stats,
                                        not args.no_standardize, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite joint MLP loss at ssl={epoch} mlp={me}')
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.detach())
            n += 1
        if me == 1 or me % args.mlp_val_every == 0 or me == args.mlp_epochs:
            val = joint_evaluate(head, valid_loader, device, stats, not args.no_standardize)
            macro = joint_macro_score(val)
            row = {'ssl_epoch': epoch, 'mlp_epoch': me,
                   'train_joint_loss': running / max(1, n), 'val_macro_score': macro}
            for t in TASKS:
                row[f'{t}_val_pcc'] = val[t]['log']['pcc']
                row[f'{t}_val_r2'] = val[t]['log']['r2']
                row[f'{t}_val_score'] = val[t]['log']['custom_score']
            history.append(row)
            if macro > best_score and np.isfinite(macro):
                best_score = macro
                best_mlp_epoch = me
                torch.save({'model': head.state_dict(), 'hidden': args.mlp_hidden,
                            'mlp_epoch': me, 'ssl_epoch': epoch, 'target_stats': stats,
                            'standardize': not args.no_standardize}, directory / 'best.pt')
            if me == 1 or me % args.mlp_log_every == 0 or me == args.mlp_epochs:
                scores = ' '.join(f'{t}:{val[t]["log"]["pcc"]:+.3f}' for t in TASKS)
                print(f'[SSL {epoch:03d} | JointMLP {me:03d}/{args.mlp_epochs}] '
                      f'loss={running/max(1,n):.5f} val_macro={macro:.4f} '
                      f'best={best_score:.4f}@{best_mlp_epoch} val_PCC({scores})',
                      flush=True)
    if not (directory / 'best.pt').exists():
        raise RuntimeError(f'No usable validation MLP checkpoint at ssl={epoch}')
    save_csv(directory / 'mlp_history.csv', history)
    ck = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    head.load_state_dict(ck['model'])
    val = joint_evaluate(head, valid_loader, device, stats, not args.no_standardize)
    test = joint_evaluate(head, test_loader, device, stats, not args.no_standardize)
    rows = []
    for task in TASKS:
        row = {'ssl_epoch': epoch, 'task': task,
               'selected_mlp_epoch': best_mlp_epoch,
               'val_joint_score': best_score,
               'selection': 'val_joint_macro_only'}
        for prefix, ev in (('val', val[task]), ('test', test[task])):
            for metric, value in ev['log'].items():
                row[f'{prefix}_{metric}'] = value
            for metric, value in ev['raw'].items():
                row[f'{prefix}_raw_{metric}'] = value
        rows.append(row)
    save_csv(directory / 'five_tasks_val_test.csv', rows)
    save_json(directory / 'summary.json', {
        'ssl_epoch': epoch, 'selected_mlp_epoch': best_mlp_epoch,
        'selection': 'best validation macro score of 5 tasks',
        'val_joint_score': best_score,
        'task_metrics': rows,
    })
    return rows


def existing_csv_rows(path):
    if not path.exists():
        return []
    with path.open('r', newline='') as f:
        return list(csv.DictReader(f))


def update_live_outputs(args, root, new_rows):
    """Rewrite result tables each SSL epoch: readable immediately while running."""
    summary = root / 'test_summary.csv'       # 5 rows * completed SSL epochs
    all_rows = [r for r in existing_csv_rows(summary)
                if int(r['ssl_epoch']) != int(new_rows[0]['ssl_epoch'])] + new_rows
    all_rows.sort(key=lambda r: (int(r['ssl_epoch']), TASKS.index(r['task'])))
    save_csv(summary, all_rows)
    best_rows = []
    for task in TASKS:
        # Use validation score to select; test values are reported, NEVER ranked.
        choices = [r for r in all_rows if r['task'] == task]
        best = max(choices, key=lambda r: float(r['val_custom_score']))
        best_rows.append(dict(best))
    save_csv(root / 'best_summary.csv', best_rows)
    epoch_scores = []
    epochs = sorted(set(int(r['ssl_epoch']) for r in all_rows))
    for epoch in epochs:
        group = [r for r in all_rows if int(r['ssl_epoch']) == epoch]
        if len(group) != len(TASKS):
            continue
        epoch_scores.append({'ssl_epoch': epoch,
                             'val_joint_score': float(group[0]['val_joint_score']),
                             'selected_mlp_epoch': int(group[0]['selected_mlp_epoch']),
                             'mean_val_pcc': float(np.mean([float(x['val_pcc']) for x in group])),
                             'mean_test_pcc_DIAGNOSTIC': float(np.mean([float(x['test_pcc']) for x in group]))})
    save_csv(root / 'epoch_joint_summary.csv', epoch_scores)
    winner = max(epoch_scores, key=lambda r: float(r['val_joint_score']))
    save_json(root / 'best_joint_epoch.json', {
        **winner, 'selection': 'validation_joint_macro_only',
        'warning': 'test was evaluated for monitoring; not used for selection'})
    return best_rows, winner


def show_live_best(epoch, rows, best_rows, winner, root):
    print('\n' + '=' * 108, flush=True)
    print(f' SSL EPOCH {epoch} FINISHED | THIS EPOCH VAL -> TEST, AND HISTORY BEST (VAL-SELECTED)', flush=True)
    print(f' {"Task":<7} {"ValPCC":>9} {"ValR2":>9} {"TestPCC":>9} '
          f'{"TestR2":>9} {"BestSSL":>9} {"BestValPCC":>11} {"BestTestPCC":>12}', flush=True)
    best_by_task = {r['task']: r for r in best_rows}
    for r in rows:
        b = best_by_task[r['task']]
        print(f' {r["task"]:<7} {float(r["val_pcc"]):>9.4f} {float(r["val_r2"]):>9.4f} '
              f'{float(r["test_pcc"]):>9.4f} {float(r["test_r2"]):>9.4f} '
              f'{int(b["ssl_epoch"]):>9} {float(b["val_pcc"]):>11.4f} '
              f'{float(b["test_pcc"]):>12.4f}', flush=True)
    print(f' BEST JOINT SSL: {winner["ssl_epoch"]}, val_macro={winner["val_joint_score"]:.5f} '
          f'| CSV updated: {root / "test_summary.csv"}', flush=True)
    print(' NOTE: BEST is selected with validation only. Test is diagnostic per epoch.', flush=True)
    print('=' * 108 + '\n', flush=True)


def evaluate_ssl_epoch(args, root, device, epoch, encoder=None, train_dataset=None):
    """Current SSL weights -> 3 embedding splits -> one 5-task MLP -> val/test."""
    output = root / 'joint_mlp' / f'ssl_epoch_{epoch:03d}' / 'summary.json'
    if args.resume and output.exists():
        old = json.loads(output.read_text())
        rows = old['task_metrics']
        best_rows, winner = update_live_outputs(args, root, rows)
        show_live_best(epoch, rows, best_rows, winner, root)
        return
    with preserve_rng():
        own = encoder is None
        m = load_model(root, epoch, args, device) if own else encoder
        previous_predict = m.predict
        previous_mode = m.training
        try:
            m.predict = True
            m.eval()
            embed_dir = root / 'embeddings' / f'epoch_{epoch:03d}'
            for split in ('train', 'valid', 'test'):
                ds = (train_dataset if split == 'train' and train_dataset is not None
                      else get_dataset(args, split))
                # An epoch is always freshly extracted if its MLP is not done.
                extract_for_split(m, ds, split, embed_dir, device, skip_existing=False)
            # Free model forward memory before creating the five-task MLP graph.
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            rows = train_joint_one_ssl(args, root, device, epoch)
            best_rows, winner = update_live_outputs(args, root, rows)
            show_live_best(epoch, rows, best_rows, winner, root)
        finally:
            m.predict = previous_predict
            if train_dataset is not None and not own:
                attach_feats(m, train_dataset, device)
            m.train(previous_mode)
            if own:
                del m
            if not args.keep_embeddings:
                shutil.rmtree(root / 'embeddings' / f'epoch_{epoch:03d}', ignore_errors=True)
            if device.type == 'cuda':
                torch.cuda.empty_cache()


def run_ssl_interleaved(args, root, device, evaluate_every_epoch):
    ds = get_dataset(args, 'train')
    if len(ds) < 2 or args.batch_size < 2:
        raise ValueError('SSL triplet negatives require at least 2 designs and --batch-size >=2')
    model = new_model(args, predict=False).to(device)
    attach_feats(model, ds, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.ssl_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.95)
    start = 1
    history = []
    if args.resume:
        ckpts = sorted((root / 'checkpoints').glob('ssl_epoch_*.pt'))
        if ckpts:
            ck = torch.load(ckpts[-1], map_location='cpu', weights_only=False)
            if ck['scheme'] != args.scheme or ck.get('gnn_layers') != args.gnn_layers:
                raise ValueError('Checkpoint architecture differs from --scheme/--gnn-layers')
            model.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            scheduler.load_state_dict(ck['scheduler'])
            start = int(ck['epoch']) + 1
            hp = root / 'ssl_history.json'
            history = json.loads(hp.read_text()) if hp.exists() else []
            if 'torch_rng' in ck:
                torch.set_rng_state(ck['torch_rng'].cpu())
            if device.type == 'cuda' and ck.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all(ck['cuda_rng'])
            if 'numpy_rng' in ck:
                np.random.set_state(ck['numpy_rng'])
            if 'python_rng' in ck:
                random.setstate(ck['python_rng'])
            print(f'Resume SSL at epoch {start}', flush=True)
            if evaluate_every_epoch:
                # Re-evaluate the saved final epoch if interrupted mid-MLP.
                previous = start - 1
                last = root / 'joint_mlp' / f'ssl_epoch_{previous:03d}' / 'summary.json'
                if previous > 0 and not last.exists():
                    evaluate_ssl_epoch(args, root, device, previous, encoder=model,
                                       train_dataset=ds)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        pin_memory=device.type == 'cuda')
    for epoch in range(start, args.epochs + 1):
        model.train()
        attach_feats(model, ds, device)
        running, steps = 0.0, 0
        progress = tqdm(loader, desc=f'{args.scheme} SSL {epoch}/{args.epochs}')
        for batch in progress:
            idx, edges, types, width, path, density, _ = unpack(batch, device)
            optimizer.zero_grad(set_to_none=True)
            *_, y, z = model(idx, edges, types, width, path, density)
            n = len(y)
            if n < 2:
                continue
            neg_y = torch.roll(y, random.randrange(1, n), 0)
            neg_z = torch.roll(z, random.randrange(1, n), 0)
            loss = (F.triplet_margin_loss(y, z, neg_z, margin=1.) +
                    F.triplet_margin_loss(z, y, neg_y, margin=1.))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SSL loss')
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.detach())
            steps += 1
            progress.set_postfix(ssl_loss=f'{running / steps:.4f}')
        scheduler.step()
        row = {'epoch': epoch, 'ssl_loss': running / max(steps, 1),
               'lr': optimizer.param_groups[0]['lr']}
        history.append(row)
        save_json(root / 'ssl_history.json', history)
        checkpoint = model_path(root, epoch)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        # Save BEFORE running the MLP so --resume can recover if interrupted.
        torch.save({'epoch': epoch, 'scheme': args.scheme,
                    'gnn_layers': args.gnn_layers, 'dropout': args.dropout,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                    'numpy_rng': np.random.get_state(),
                    'python_rng': random.getstate()}, checkpoint)
        print(f'[SSL {epoch:03d}/{args.epochs}] loss={row["ssl_loss"]:.5f}; '
              f'checkpoint={checkpoint}', flush=True)
        if evaluate_every_epoch:
            evaluate_ssl_epoch(args, root, device, epoch, encoder=model, train_dataset=ds)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stage', choices=('all', 'ssl', 'evaluate'), default='all',
                   help='all=interleaved SSL+MLP+Val+Test; evaluate=existing SSL checkpoints')
    p.add_argument('--scheme', choices=tuple(BRANCHES), required=True)
    p.add_argument('--data-root', default='data')
    p.add_argument('--split-ratio', default='0.3-0.3-0.4')
    p.add_argument('--output', default='runs/multi_gnn_joint')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=615)
    p.add_argument('--gnn-layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=.2)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--ssl-lr', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--eval-epochs', default='all',
                   help='Used only by --stage evaluate; all, last, or 1,2,3,...')
    p.add_argument('--mlp-epochs', type=int, default=50,
                   help='Joint MLP epochs FROM SCRATCH after EACH SSL epoch')
    p.add_argument('--mlp-batch-size', type=int, default=8)
    p.add_argument('--mlp-workers', type=int, default=0)
    p.add_argument('--mlp-lr', type=float, default=1e-3)
    p.add_argument('--mlp-hidden', type=int, default=512)
    p.add_argument('--mlp-dropout', type=float, default=0.0)
    p.add_argument('--mlp-weight-decay', type=float, default=0.0)
    p.add_argument('--mlp-val-every', type=int, default=1)
    p.add_argument('--mlp-log-every', type=int, default=10)
    p.add_argument('--no-standardize', action='store_true',
                   help='Disable train-only per-task standardization after signed_log1p')
    p.add_argument('--keep-embeddings', action='store_true',
                   help='Preserve large per-epoch embedding folders on disk')
    p.add_argument('--resume', action='store_true',
                   help='Restore SSL and unfinished MLP evaluation')
    return p.parse_args()


def main():
    args = parse_args()
    if args.mlp_epochs < 1 or args.mlp_val_every < 1 or args.mlp_log_every < 1:
        raise ValueError('MLP epochs and reporting intervals must be >=1')
    seed_everything(args.seed)
    device = resolve_device(args.device)
    root = Path(args.output) / args.scheme
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / 'settings.json', vars(args))
    print(f'[{args.scheme}] {BRANCHES[args.scheme]} | 384 graph + 384 summary = 768 | '
          f'SSL epochs={args.epochs}; ONE JOINT 5-task MLP every SSL epoch; device={device}',
          flush=True)
    if args.stage in ('all', 'ssl'):
        run_ssl_interleaved(args, root, device, evaluate_every_epoch=args.stage == 'all')
    else:
        for epoch in selected_epochs(args, root):
            evaluate_ssl_epoch(args, root, device, epoch)
    print(f'COMPLETE: {root / "test_summary.csv"}' if args.stage != 'ssl'
          else 'COMPLETE: SSL checkpoints saved; PPA evaluation not requested', flush=True)


if __name__ == '__main__':
    main()
