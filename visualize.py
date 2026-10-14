#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PolyRTL branch visualization for a FINISHED Scheme-2 run.

What this script does
---------------------
1) Finds the validation-selected best joint SSL epoch from best_joint_epoch.json.
2) Loads the corresponding SSL checkpoint and best joint PPA MLP.
3) Extracts the 8 x 48-D branch representations without modifying the model.
4) Evaluates/plots:
   A. Branch representation similarity (Linear CKA), using TEST designs.
   B. Structural specialization (train-fitted linear probes, TEST R^2).
   C. Branch -> PPA contribution by inference-time leave-one-branch-out,
      zeroing one 48-D graph branch while keeping the trained MLP fixed.
   D1. Per-branch PCA/UMAP manifolds on TEST only.
   D2. Per-branch PCA/UMAP manifolds on ALL designs (TRAIN + VALID + TEST).
   E. Sample-wise branch sensitivity heatmap on TEST designs.

Important evaluation policy
---------------------------
- CKA: TEST only.
- Structural linear probe: fit on TRAIN, evaluate on TEST.
- Branch ablation: TEST only, fixed trained model, no retraining.
- TEST manifold: TEST only.
- ALL manifold: TRAIN + VALID + TEST, for qualitative visualization only.
- The main PolyRTL encoder and joint PPA head are NEVER retrained here.
- Only the lightweight ridge linear probes in (B) are fit.

Run this script from the TopoRTL repository root.

Example
-------
python visualize.py \
  --run-dir results/scheme2 \
  --train-script train.py \
  --manifold-figsize 22 12 \
  --manifold-font-size 30 \
  --manifold-panel-aspect 1.0 \
  --dpi 600
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


BRANCHES = (
    "global", "plain", "fanout", "fanin",
    "chain", "junction", "depth", "reverse",
)

TASKS = ("area", "power", "wns", "tns", "slack")
GLOBAL_TASKS = ("area", "power", "wns", "tns")

STRUCT_ATTRS = (
    "register_count",
    "edge_density",
    "mean_degree",
    "fanout_ratio",
    "fanin_ratio",
    "chain_ratio",
    "junction_ratio",
    "mean_path_depth",
    "max_path_depth",
    "directional_asymmetry",
)

BRANCH_COLOR_ATTR = {
    "global": "edge_density",
    "plain": "mean_degree",
    "fanout": "fanout_ratio",
    "fanin": "fanin_ratio",
    "chain": "chain_ratio",
    "junction": "junction_ratio",
    "depth": "max_path_depth",
    "reverse": "directional_asymmetry",
}


# ============================================================
# Arguments / utilities
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        required=True,
        help="Finished run root, e.g. results/multignn_full-v3/scheme2",
    )
    p.add_argument(
        "--train-script",
        default="03-train-v3.py",
        help="Training entry file used to define/load the exact model",
    )
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Override data root saved in settings.json",
    )
    p.add_argument(
        "--split-ratio",
        default=None,
        help="Override split ratio saved in settings.json",
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Override best_joint_epoch.json and visualize this SSL epoch",
    )
    p.add_argument(
        "--probe-alpha",
        type=float,
        default=1e-2,
        help="Ridge strength for structural linear probes",
    )
    p.add_argument(
        "--top-k-cases",
        type=int,
        default=20,
        help="Number of most branch-sensitive TEST designs in case-study heatmap",
    )
    p.add_argument("--seed", type=int, default=615)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--manifold-figsize", "--pca-figsize",
        type=float, nargs=2, metavar=("WIDTH", "HEIGHT"),
        default=(22.0, 12.0),
        help="Full 2x4 PCA/UMAP figure dimensions in inches; default: 22 12",
    )
    p.add_argument(
        "--manifold-font-size", "--pca-font-size",
        type=float, default=40.0,
        help="One shared font size (pt) for ALL manifold text: figure title, "
             "(a)-(h), colorbar caption and colorbar tick labels; default: 40",
    )
    p.add_argument(
        "--manifold-panel-aspect", "--pca-panel-aspect",
        type=float, default=1.0,
        help="Individual PCA/UMAP box aspect (height/width); 1.0 = square",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Plot directory; default=<run-dir>/visualizations/best_ssl_XXX",
    )
    return p.parse_args()


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str):
    d = torch.device(name)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({name}) but CUDA is unavailable.")
    return d


def resolve_run_dir(path: str) -> Path:
    p = Path(path)

    if (p / "best_joint_epoch.json").exists() or (p / "checkpoints").exists():
        return p

    q = p / "scheme2"
    if (q / "best_joint_epoch.json").exists() or (q / "checkpoints").exists():
        return q

    raise FileNotFoundError(
        f"Cannot recognize run directory: {p}\n"
        f"Expected best_joint_epoch.json/checkpoints under {p} or {q}."
    )


def load_training_module(path: str):
    path = str(Path(path).resolve())
    if not Path(path).exists():
        raise FileNotFoundError(f"Training script not found: {path}")

    spec = importlib.util.spec_from_file_location(
        "polyrtl_training_for_viz", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import training script: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_json(path: Path):
    return json.loads(path.read_text())


def get_best_epoch(run_dir: Path, override=None) -> int:
    if override is not None:
        return int(override)

    f = run_dir / "best_joint_epoch.json"
    if f.exists():
        x = read_json(f)
        return int(x["ssl_epoch"])

    f = run_dir / "epoch_joint_summary.csv"
    if f.exists():
        with f.open(newline="") as h:
            rows = list(csv.DictReader(h))
        if rows:
            row = max(rows, key=lambda r: float(r["val_joint_score"]))
            return int(row["ssl_epoch"])

    f = run_dir / "test_summary.csv"
    if f.exists():
        with f.open(newline="") as h:
            rows = list(csv.DictReader(h))

        if rows:
            by_epoch = {}
            for r in rows:
                e = int(r["ssl_epoch"])
                by_epoch.setdefault(e, float(r["val_joint_score"]))
            return max(by_epoch, key=by_epoch.get)

    raise FileNotFoundError(
        "Could not infer the best SSL epoch. Missing best_joint_epoch.json, "
        "epoch_joint_summary.csv, and usable test_summary.csv."
    )


def build_cfg(run_dir: Path, ckpt, cli):
    settings_path = run_dir / "settings.json"
    settings = read_json(settings_path) if settings_path.exists() else {}

    settings.setdefault("scheme", "scheme2")
    settings["gnn_layers"] = int(
        ckpt.get("gnn_layers", settings.get("gnn_layers", 3))
    )
    settings["dropout"] = float(
        ckpt.get("dropout", settings.get("dropout", 0.2))
    )
    settings["data_root"] = (
        cli.data_root or settings.get("data_root", "data")
    )
    settings["split_ratio"] = (
        cli.split_ratio or settings.get("split_ratio", "0.3-0.3-0.4")
    )
    settings.setdefault("mlp_hidden", 512)
    settings.setdefault("mlp_dropout", 0.0)

    return SimpleNamespace(**settings)


def instantiate_model(train_mod, cfg, ckpt, device):
    if hasattr(train_mod, "new_model"):
        try:
            model = train_mod.new_model(cfg, predict=True)
        except TypeError:
            model = train_mod.new_model(cfg)
            model.predict = True

    elif hasattr(train_mod, "MultiGNNTopoRTL"):
        cls = train_mod.MultiGNNTopoRTL

        try:
            model = cls(cfg.scheme, cfg.gnn_layers, cfg)
        except TypeError:
            model = cls(cfg.gnn_layers, cfg)

        model.predict = True

    else:
        raise AttributeError(
            "Training module exposes neither new_model nor MultiGNNTopoRTL."
        )

    model = model.to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.predict = True
    model.eval()

    return model


def get_dataset(train_mod, cfg, split):
    if hasattr(train_mod, "get_dataset"):
        return train_mod.get_dataset(cfg, split)

    if not hasattr(train_mod, "CircuitDataset"):
        raise AttributeError(
            "Could not find get_dataset or CircuitDataset in training module."
        )

    path = (
        Path(cfg.data_root)
        / "processed"
        / cfg.split_ratio
        / split
        / f"{split}.pkl"
    )

    return train_mod.CircuitDataset(str(path))


def attach_feats(train_mod, model, ds, device):
    if hasattr(train_mod, "attach_feats"):
        train_mod.attach_feats(model, ds, device)
        return

    model.set_graph_features(ds.graph_features.to(device))
    model.set_summary_features(ds.summary_features.to(device))


def unpack_batch(train_mod, batch, device):
    if hasattr(train_mod, "unpack"):
        return train_mod.unpack(batch, device)

    idx, edges, types, width, density, path, labels = batch

    return (
        idx.to(device),
        edges.to(device),
        types.to(device),
        width.to(device),
        path.to(device),
        density.to(device),
        labels,
    )


def detect_branch_names(model):
    gm = model.graph_model
    names = tuple(getattr(gm, "names", BRANCHES))

    if tuple(names) != BRANCHES:
        raise ValueError(
            "This visualization script is for Scheme 2 only.\n"
            f"Expected branches {BRANCHES}, got {names}"
        )

    return names


# ============================================================
# Structural statistics
# ============================================================

def valid_rig_degrees(index, edges, types):
    """
    Replicate the trained encoder's valid-RIG definition for one design.
    """
    n = int(index.numel())
    real = index.reshape(-1) > 1

    valid = (types >= 0) & (types <= 1)
    valid = (
        valid
        & (edges[:, 0] >= 1)
        & (edges[:, 0] < n)
        & (edges[:, 1] >= 1)
        & (edges[:, 1] < n)
    )

    ep = edges[valid].long()

    if ep.numel() == 0:
        src = torch.empty(
            0, dtype=torch.long, device=index.device
        )
        dst = torch.empty(
            0, dtype=torch.long, device=index.device
        )
    else:
        src, dst = ep[:, 0], ep[:, 1]
        keep = real[src] & real[dst]
        src, dst = src[keep], dst[keep]

    ind = torch.bincount(dst, minlength=n)
    outd = torch.bincount(src, minlength=n)

    return real, src, dst, ind, outd


def structural_statistics(index, edges, types, path):
    real, src, dst, ind, outd = valid_rig_degrees(
        index, edges, types
    )

    nodes = real.nonzero(as_tuple=True)[0]
    nr = int(nodes.numel())
    ne = int(src.numel())

    if nr:
        deg = (ind + outd)[nodes].float()

        fanout = (outd[nodes] >= 2).float()
        fanin = (ind[nodes] >= 2).float()
        chain = (
            (outd[nodes] == 1)
            & (ind[nodes] == 1)
        ).float()
        junction = (
            (outd[nodes] >= 2)
            & (ind[nodes] >= 2)
        ).float()

        asym = (
            (ind[nodes] - outd[nodes]).abs().float()
            / (ind[nodes] + outd[nodes])
            .float()
            .clamp_min(1.0)
        )

        padded_path = torch.nn.functional.pad(
            path.float().reshape(-1), (1, 0)
        )

        if padded_path.numel() >= index.numel():
            p = padded_path[nodes]
        else:
            p = path.float().reshape(-1)

        if p.numel() > nr:
            p = p[:nr]

        mean_degree = float(deg.mean())
        fanout_ratio = float(fanout.mean())
        fanin_ratio = float(fanin.mean())
        chain_ratio = float(chain.mean())
        junction_ratio = float(junction.mean())
        directional_asymmetry = float(asym.mean())
        mean_path_depth = float(p.mean()) if p.numel() else 0.0
        max_path_depth = float(p.max()) if p.numel() else 0.0

    else:
        mean_degree = 0.0
        fanout_ratio = 0.0
        fanin_ratio = 0.0
        chain_ratio = 0.0
        junction_ratio = 0.0
        directional_asymmetry = 0.0
        mean_path_depth = 0.0
        max_path_depth = 0.0

    possible = max(
        nr * max(nr - 1, 1),
        1,
    )

    edge_density = ne / possible

    return {
        "register_count": float(nr),
        "edge_density": float(edge_density),
        "mean_degree": mean_degree,
        "fanout_ratio": fanout_ratio,
        "fanin_ratio": fanin_ratio,
        "chain_ratio": chain_ratio,
        "junction_ratio": junction_ratio,
        "mean_path_depth": mean_path_depth,
        "max_path_depth": max_path_depth,
        "directional_asymmetry": directional_asymmetry,
    }


# ============================================================
# Feature extraction
# ============================================================

@torch.inference_mode()
def extract_design_representations(
    train_mod,
    model,
    ds,
    device,
    split_name,
    branch_names,
):
    """
    Returns design-level branch embeddings and structural attributes.

    cls_repr  : [num_designs, 8, 48]
    mean_repr : [num_designs, 8, 48]
    attrs     : [num_designs, num_attrs]
    """
    attach_feats(train_mod, model, ds, device)
    model.eval()

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
    )

    cls_rows = []
    mean_rows = []
    attr_rows = []
    design_names = []

    branch_dim_seen = None

    for i, batch in enumerate(
        tqdm(loader, desc=f"Extract {split_name} branches")
    ):
        (
            idx,
            edges,
            types,
            width,
            path,
            density,
            labels,
        ) = unpack_batch(
            train_mod,
            batch,
            device,
        )

        out = model(
            idx,
            edges,
            types,
            width,
            path,
            density,
        )

        if len(out) < 4:
            raise RuntimeError(
                "predict=True model did not return graph_emb."
            )

        # model(..., predict=True) returns:
        # tokens, tokens.mean(1), graph, graph_emb, summary, summary_emb
        graph_emb = out[3]

        branch_dim = (
            graph_emb.shape[-1]
            // len(branch_names)
        )

        if branch_dim_seen is None:
            branch_dim_seen = branch_dim

        if (
            branch_dim * len(branch_names)
            != graph_emb.shape[-1]
        ):
            raise ValueError(
                "Graph embedding cannot be evenly split into 8 branches."
            )

        real = idx[0] > 1

        cls_parts = []
        mean_parts = []

        for bi in range(len(branch_names)):
            s = bi * branch_dim
            e = (bi + 1) * branch_dim

            h = graph_emb[0, :, s:e]

            cls_parts.append(
                h[0]
                .float()
                .cpu()
                .numpy()
            )

            if bool(real.any()):
                mean_parts.append(
                    h[real]
                    .mean(0)
                    .float()
                    .cpu()
                    .numpy()
                )
            else:
                mean_parts.append(
                    np.zeros(
                        branch_dim,
                        dtype=np.float32,
                    )
                )

        cls_rows.append(
            np.stack(cls_parts)
        )

        mean_rows.append(
            np.stack(mean_parts)
        )

        stats = structural_statistics(
            idx[0],
            edges[0],
            types[0],
            path[0],
        )

        attr_rows.append(
            [
                stats[k]
                for k in STRUCT_ATTRS
            ]
        )

        if hasattr(ds, "design_list"):
            design_names.append(
                str(ds.design_list[i])
            )
        else:
            design_names.append(
                str(i)
            )

    return {
        "designs": np.asarray(
            design_names,
            dtype=object,
        ),
        "cls": np.asarray(
            cls_rows,
            dtype=np.float64,
        ),
        "mean": np.asarray(
            mean_rows,
            dtype=np.float64,
        ),
        "attrs": np.asarray(
            attr_rows,
            dtype=np.float64,
        ),
        "branch_dim": int(
            branch_dim_seen or 48
        ),
    }


# ============================================================
# CKA
# ============================================================

def linear_cka(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    )
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    x = (
        x
        - x.mean(
            0,
            keepdims=True,
        )
    )

    y = (
        y
        - y.mean(
            0,
            keepdims=True,
        )
    )

    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y

    num = np.sum(
        xy * xy
    )

    den = math.sqrt(
        np.sum(xx * xx)
        * np.sum(yy * yy)
    )

    if den <= 1e-20:
        return 0.0

    return float(
        num / den
    )


def cka_matrix(repr_3d):
    b = repr_3d.shape[1]

    out = np.zeros(
        (b, b),
        dtype=np.float64,
    )

    for i in range(b):
        for j in range(b):
            out[i, j] = linear_cka(
                repr_3d[:, i],
                repr_3d[:, j],
            )

    return out


# ============================================================
# Linear probe
# ============================================================

def safe_standardize_fit(x):
    mu = np.mean(
        x,
        axis=0,
        keepdims=True,
    )

    sd = np.std(
        x,
        axis=0,
        keepdims=True,
    )

    sd = np.where(
        sd < 1e-8,
        1.0,
        sd,
    )

    return mu, sd


def ridge_probe(
    train_x,
    train_y,
    test_x,
    alpha,
):
    train_x = np.asarray(
        train_x,
        dtype=np.float64,
    )
    test_x = np.asarray(
        test_x,
        dtype=np.float64,
    )
    train_y = np.asarray(
        train_y,
        dtype=np.float64,
    ).reshape(-1)

    xmu, xsd = safe_standardize_fit(
        train_x
    )

    xm = (
        train_x - xmu
    ) / xsd

    xt = (
        test_x - xmu
    ) / xsd

    ymu = float(
        np.mean(train_y)
    )

    ysd = float(
        np.std(train_y)
    )

    if ysd < 1e-8:
        return np.full(
            len(test_x),
            ymu,
            dtype=np.float64,
        )

    ym = (
        train_y - ymu
    ) / ysd

    gram = xm.T @ xm

    eye = np.eye(
        gram.shape[0],
        dtype=np.float64,
    )

    beta = np.linalg.solve(
        gram + alpha * eye,
        xm.T @ ym,
    )

    return (
        xt @ beta
    ) * ysd + ymu


def r2_score_np(y, p):
    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    p = np.asarray(
        p,
        dtype=np.float64,
    ).reshape(-1)

    keep = (
        np.isfinite(y)
        & np.isfinite(p)
    )

    y = y[keep]
    p = p[keep]

    if len(y) < 2:
        return 0.0

    den = np.sum(
        (y - y.mean()) ** 2
    )

    if den <= 1e-12:
        return 0.0

    return float(
        1.0
        - np.sum(
            (y - p) ** 2
        )
        / den
    )


def pcc_np(y, p):
    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    p = np.asarray(
        p,
        dtype=np.float64,
    ).reshape(-1)

    keep = (
        np.isfinite(y)
        & np.isfinite(p)
    )

    y = y[keep]
    p = p[keep]

    if (
        len(y) < 2
        or np.std(y) < 1e-12
        or np.std(p) < 1e-12
    ):
        return 0.0

    return float(
        np.corrcoef(
            y,
            p,
        )[0, 1]
    )


def structural_probe_matrix(
    train_repr,
    test_repr,
    train_attrs,
    test_attrs,
    alpha,
):
    b = train_repr.shape[1]
    a = train_attrs.shape[1]

    scores = np.zeros(
        (b, a),
        dtype=np.float64,
    )

    for bi in range(b):
        for ai in range(a):
            pred = ridge_probe(
                train_repr[:, bi],
                train_attrs[:, ai],
                test_repr[:, bi],
                alpha,
            )

            scores[bi, ai] = r2_score_np(
                test_attrs[:, ai],
                pred,
            )

    return scores


# ============================================================
# CSV / plots
# ============================================================

def save_matrix_csv(
    path: Path,
    matrix,
    row_names,
    col_names,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        w = csv.writer(f)

        w.writerow(
            [""]
            + list(col_names)
        )

        for name, row in zip(
            row_names,
            matrix,
        ):
            w.writerow(
                [name]
                + [
                    float(x)
                    for x in row
                ]
            )


def make_strong_rwb_cmap():
    """High-contrast red/white/blue colormap with pure extremes.

    Low values -> pure blue; middle -> white; high values -> pure red.
    Extra control points near both ends provide more visible gradation while
    preserving the most saturated colors at the two extremes.
    """
    return LinearSegmentedColormap.from_list(
        "polyrtl_pure_rwb",
        [
            (0.000, "#0518EA"),
            (0.100, "#0a52c6"),
            (0.260, "#6491cb"),
            (0.360, "#aac3fb"),
            (0.450, "#e2e6ea"),
            (0.460, "#dfe0e2"),
            (0.500, "#f9e7e7"),
            (0.640, "#ed8d8d"),
            (0.740, "#e73e44"),
            (0.830, "#d42533"),
            (1.000, "#c0001d"),
        ],
        N=1024,
    )


def _density_scores_on_grid(xy, mask, bins=34):
    """Return a [0,1] density score for the selected points, per sample.

    The density is estimated with a 2D histogram over the embedding plane.
    Points in denser same-color regions receive larger scores.
    """
    xy = np.asarray(xy, dtype=np.float64)
    out = np.zeros(len(xy), dtype=np.float64)
    if mask.sum() < 3:
        return out

    pts = xy[mask]
    x = pts[:, 0]
    y = pts[:, 1]
    x0, x1 = float(np.min(x)), float(np.max(x))
    y0, y1 = float(np.min(y)), float(np.max(y))
    if abs(x1 - x0) < 1e-12 or abs(y1 - y0) < 1e-12:
        out[mask] = 1.0
        return out

    # Slight padding keeps edge points inside the outermost bins.
    px = 0.02 * (x1 - x0)
    py = 0.02 * (y1 - y0)
    x_edges = np.linspace(x0 - px, x1 + px, bins + 1)
    y_edges = np.linspace(y0 - py, y1 + py, bins + 1)
    hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    bx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1)
    by = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1)
    dens = hist[bx, by].astype(np.float64)

    lo = float(np.min(dens))
    hi = float(np.percentile(dens, 95.0))
    if hi <= lo + 1e-12:
        score = np.ones_like(dens, dtype=np.float64)
    else:
        score = np.clip((dens - lo) / (hi - lo), 0.0, 1.0)

    out[mask] = score
    return out


def _enhance_extreme_colors_by_local_density(xy, normalized):
    """Push blue/red points closer to pure extremes in dense same-color regions.

    - Blue-side points in blue-dense areas become purer blue.
    - Red-side points in red-dense areas become purer red.
    - Values close to the middle remain closer to white.
    """
    vals = np.asarray(normalized, dtype=np.float64).copy()
    blue = vals < 0.5
    red = vals > 0.5

    blue_density = _density_scores_on_grid(xy, blue)
    red_density = _density_scores_on_grid(xy, red)

    if blue.any():
        dist = (0.5 - vals[blue]) / 0.5  # 0..1
        gain = 0.90 + 0.90 * blue_density[blue]
        pushed = np.clip(dist * gain, 0.0, 1.0)
        vals[blue] = 0.5 - 0.5 * pushed

    if red.any():
        dist = (vals[red] - 0.5) / 0.5  # 0..1
        gain = 0.90 + 0.90 * red_density[red]
        pushed = np.clip(dist * gain, 0.0, 1.0)
        vals[red] = 0.5 + 0.5 * pushed

    # Neutral points remain around white; clip to valid range.
    vals = np.clip(vals, 0.0, 1.0)
    local_density = np.maximum(blue_density, red_density)
    return vals, local_density


def plot_heatmap(
    matrix,
    row_names,
    col_names,
    title,
    path,
    dpi,
    value_fmt=".2f",
    center_zero=False,
    cmap=None,
    annot_fontsize=8,
    tick_fontsize=10,
    title_fontsize=12,
    cbar_tick_fontsize=10,
    color_min=None,
    color_max=None,
):
    """Generic heatmap, with optional 4x typography for the CLS CKA plot."""
    four_x_cls = (annot_fontsize >= 32 and tick_fontsize >= 40)
    if four_x_cls:
        # Large figure avoids overlapping the 32pt numbers and 40pt labels.
        fig_w, fig_h = 23.0, 21.0
    else:
        fig_w = max(7.0, 0.82 * len(col_names) + 2.4)
        fig_h = max(5.2, 0.58 * len(row_names) + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    array = np.asarray(matrix, dtype=np.float64)
    kwargs = dict(aspect="auto", cmap=cmap if cmap is not None else "viridis")
    if center_zero:
        good = np.abs(array[np.isfinite(array)])
        lim = max(float(good.max()) if good.size else 0.0, 1e-8)
        kwargs.update(vmin=-lim, vmax=lim)
    else:
        if color_min is not None:
            kwargs["vmin"] = color_min
        if color_max is not None:
            kwargs["vmax"] = color_max

    im = ax.imshow(array, **kwargs)
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_xticklabels(col_names, rotation=40, ha="right", fontsize=tick_fontsize)
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=tick_fontsize)
    if four_x_cls:
        title = title.replace(" — ", "\n")
    ax.set_title(title, fontsize=title_fontsize, pad=28 if four_x_cls else 10)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            v = array[i, j]
            if np.isfinite(v):
                # Keep labels readable even on saturated blue/red cells.
                if cmap is not None and (v < 0.25 or v > 0.84):
                    text_color = "white"
                else:
                    text_color = "#1a1c25"
                ax.text(
                    j, i, format(v, value_fmt),
                    ha="center", va="center", color=text_color,
                    fontsize=annot_fontsize,
                )

    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    bar.ax.tick_params(labelsize=cbar_tick_fontsize)
    if four_x_cls:
        # Reserve enough room for enlarged 40pt ticks and 48pt title.
        fig.subplots_adjust(left=0.14, right=0.82, bottom=0.21, top=0.85)
    else:
        fig.tight_layout()

    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PPA branch ablation
# ============================================================

def signed_log1p_torch(y):
    return (
        torch.sign(y)
        * torch.log1p(
            y.abs()
        )
    )


def destandard_log_prediction(
    pred,
    task,
    stats,
    standardize,
):
    if standardize:
        return (
            pred
            * stats[task]["std"]
            + stats[task]["mean"]
        )

    return pred


def load_joint_head(
    train_mod,
    run_dir,
    epoch,
    device,
):
    file = (
        run_dir
        / "joint_mlp"
        / f"ssl_epoch_{epoch:03d}"
        / "best.pt"
    )

    if not file.exists():
        raise FileNotFoundError(
            f"Best JointMLP checkpoint missing: {file}\n"
            "The branch->PPA ablation plots require this file."
        )

    ck = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    if not hasattr(
        train_mod,
        "JointMLP",
    ):
        raise AttributeError(
            "Training module does not expose JointMLP."
        )

    hidden = int(
        ck.get(
            "hidden",
            512,
        )
    )

    head = train_mod.JointMLP(
        hidden=hidden
    ).to(device)

    head.load_state_dict(
        ck["model"],
        strict=True,
    )

    head.eval()

    return (
        head,
        ck["target_stats"],
        bool(
            ck.get(
                "standardize",
                True,
            )
        ),
        ck,
    )


@torch.inference_mode()
def ablation_evaluation(
    train_mod,
    model,
    head,
    ds,
    device,
    branch_names,
    target_stats,
    standardize,
    top_k_cases,
):
    attach_feats(
        train_mod,
        model,
        ds,
        device,
    )

    model.eval()
    head.eval()

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
    )

    conditions = (
        "full",
    ) + tuple(branch_names)

    truth = {
        t: []
        for t in TASKS
    }

    preds = {
        c: {
            t: []
            for t in TASKS
        }
        for c in conditions
    }

    design_names = []
    design_sensitivity = []

    for i, batch in enumerate(
        tqdm(
            loader,
            desc="TEST branch ablation",
        )
    ):
        (
            idx,
            edges,
            types,
            width,
            path,
            density,
            labels,
        ) = unpack_batch(
            train_mod,
            batch,
            device,
        )

        out = model(
            idx,
            edges,
            types,
            width,
            path,
            density,
        )

        tokens = out[0]

        real = (
            idx[0] > 1
        )

        local_full = (
            tokens[0, real]
        )

        g_full = (
            tokens[0, 0]
        )

        for t in GLOBAL_TASKS:
            y = (
                labels[t]
                .flatten()[0]
                .to(device)
                .float()
            )

            if torch.isfinite(y):
                truth[t].append(
                    float(
                        signed_log1p_torch(y)
                        .cpu()
                    )
                )

        slack_y = (
            labels["slack"][0, real.cpu()]
            .to(device)
            .float()
        )

        slack_good = torch.isfinite(
            slack_y
        )

        if bool(
            slack_good.any()
        ):
            truth["slack"].extend(
                signed_log1p_torch(
                    slack_y[
                        slack_good
                    ]
                )
                .cpu()
                .numpy()
                .tolist()
            )

        branch_dim = (
            384
            // len(branch_names)
        )

        per_design = []

        full_standard_outputs = None

        for condition in conditions:
            g = g_full.clone()
            local = local_full.clone()

            if condition != "full":
                bi = branch_names.index(
                    condition
                )

                s = (
                    bi
                    * branch_dim
                )

                e = (
                    (bi + 1)
                    * branch_dim
                )

                g[s:e] = 0

                if local.numel():
                    local[:, s:e] = 0

            if len(local):
                mean = (
                    local
                    .mean(
                        0,
                        keepdim=True,
                    )
                )

                maximum = (
                    local
                    .max(
                        0,
                        keepdim=True,
                    )
                    .values
                )
            else:
                mean = (
                    torch.zeros_like(g)
                    .unsqueeze(0)
                )

                maximum = (
                    torch.zeros_like(g)
                    .unsqueeze(0)
                )

            o = head(
                g.unsqueeze(0),
                mean,
                maximum,
                local.reshape(
                    -1,
                    768,
                ),
            )

            standard_out = {}

            for t in GLOBAL_TASKS:
                standard_out[t] = (
                    o[t]
                    .detach()
                    .flatten()
                )

                p_log = destandard_log_prediction(
                    o[t]
                    .detach()
                    .flatten(),
                    t,
                    target_stats,
                    standardize,
                )

                preds[
                    condition
                ][t].append(
                    float(
                        p_log
                        .cpu()[0]
                    )
                )

            standard_out["slack"] = (
                o["slack"]
                .detach()
                .flatten()
            )

            p_log = destandard_log_prediction(
                o["slack"]
                .detach()
                .flatten(),
                "slack",
                target_stats,
                standardize,
            )

            if len(p_log):
                preds[
                    condition
                ]["slack"].extend(
                    p_log
                    .cpu()
                    .numpy()
                    .tolist()
                )

            if condition == "full":
                full_standard_outputs = (
                    standard_out
                )

            else:
                vals = []

                for t in GLOBAL_TASKS:
                    vals.append(
                        float(
                            (
                                standard_out[t]
                                - full_standard_outputs[t]
                            )
                            .abs()
                            .mean()
                            .cpu()
                        )
                    )

                if len(
                    standard_out["slack"]
                ):
                    vals.append(
                        float(
                            (
                                standard_out["slack"]
                                - full_standard_outputs["slack"]
                            )
                            .abs()
                            .mean()
                            .cpu()
                        )
                    )
                else:
                    vals.append(0.0)

                per_design.append(
                    float(
                        np.mean(vals)
                    )
                )

        if hasattr(
            ds,
            "design_list",
        ):
            name = str(
                ds.design_list[i]
            )
        else:
            name = str(i)

        design_names.append(
            name
        )

        design_sensitivity.append(
            per_design
        )

    r2 = np.zeros(
        (
            len(branch_names),
            len(TASKS),
        ),
        dtype=np.float64,
    )

    pcc = np.zeros_like(
        r2
    )

    baseline_r2 = {}
    baseline_pcc = {}

    for t in TASKS:
        y = np.asarray(
            truth[t],
            dtype=np.float64,
        )

        p = np.asarray(
            preds["full"][t],
            dtype=np.float64,
        )

        if len(y) != len(p):
            raise RuntimeError(
                f"Prediction/label count mismatch for {t}: "
                f"y={len(y)}, p={len(p)}"
            )

        baseline_r2[t] = r2_score_np(
            y,
            p,
        )

        baseline_pcc[t] = pcc_np(
            y,
            p,
        )

    for bi, branch in enumerate(
        branch_names
    ):
        for ti, t in enumerate(
            TASKS
        ):
            y = np.asarray(
                truth[t],
                dtype=np.float64,
            )

            p = np.asarray(
                preds[branch][t],
                dtype=np.float64,
            )

            r2[
                bi,
                ti,
            ] = (
                baseline_r2[t]
                - r2_score_np(
                    y,
                    p,
                )
            )

            pcc[
                bi,
                ti,
            ] = (
                baseline_pcc[t]
                - pcc_np(
                    y,
                    p,
                )
            )

    sens = np.asarray(
        design_sensitivity,
        dtype=np.float64,
    )

    if len(sens):
        order = np.argsort(
            -sens.mean(
                axis=1
            )
        )

        k = min(
            int(top_k_cases),
            len(order),
        )

        order = order[:k]

    else:
        order = np.asarray(
            [],
            dtype=int,
        )

    return {
        "delta_r2": r2,
        "delta_pcc": pcc,
        "baseline_r2": baseline_r2,
        "baseline_pcc": baseline_pcc,
        "design_names": np.asarray(
            design_names,
            dtype=object,
        ),
        "sensitivity": sens,
        "top_indices": order,
    }


# ============================================================
# PCA / UMAP manifolds
# ============================================================

def pca_2d(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = (
        x
        - x.mean(
            0,
            keepdims=True,
        )
    )

    u, s, vt = np.linalg.svd(
        x,
        full_matrices=False,
    )

    if vt.shape[0] < 2:
        z = np.zeros(
            (
                len(x),
                2,
            ),
            dtype=np.float64,
        )

        if vt.shape[0] == 1:
            z[:, 0] = (
                x @ vt[0]
            )

        return z

    return (
        x
        @ vt[:2].T
    )


def embed_2d(
    x,
    seed,
):
    try:
        import umap

        if len(x) >= 4:
            n_neighbors = max(
                2,
                min(
                    15,
                    len(x) - 1,
                ),
            )

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=seed,
            )

            return (
                reducer.fit_transform(x),
                "UMAP",
            )

    except Exception:
        pass

    return (
        pca_2d(x),
        "PCA",
    )


def plot_branch_manifolds(
    repr_data,
    attrs_data,
    branch_names,
    out,
    dpi,
    seed,
    dataset_label,
    figsize=(22.0, 12.0),
    font_size=40.0,
    panel_aspect=1.0,
):
    """Render 8 near-square manifold panels in a 2x4 grid.

    Adjustable visualization hyperparameters:
      figsize:      (width, height) of the complete figure, in inches.
      font_size:    one shared font size, in points, for every visible text item.
      panel_aspect: height/width of each panel (1.0 = a perfect square).

    The per-branch structural attributes differ, so each attribute is
    normalized separately by its 2nd--98th percentiles. The shared colorbar
    displays that relative range; density-based enhancement subsequently
    pushes dense same-color regions toward stronger reds or blues.
    """
    width, height = map(float, figsize)
    font_size = float(font_size)
    panel_aspect = float(panel_aspect)
    if width <= 0 or height <= 0 or font_size <= 0 or panel_aspect <= 0:
        raise ValueError("Manifold figure dimensions, fonts and aspect must be positive")
    if len(branch_names) != 8:
        raise ValueError("This plot requires exactly 8 branches")

    attr_index = {name: i for i, name in enumerate(STRUCT_ATTRS)}
    fig, axes = plt.subplots(2, 4, figsize=(width, height))

    # Keep panels closely spaced, but reserve a separate right column for
    # the single, large-type colorbar.  Independent box aspects ensure the
    # panels stay square even when the whole figure size is changed.
    fig.subplots_adjust(
        left=0.035, right=0.805, top=0.875, bottom=0.075,
        wspace=0.065, hspace=0.16,
    )
    method_used = set()
    cmap = make_strong_rwb_cmap()
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)

    for bi, (ax, branch) in enumerate(zip(axes.flat, branch_names)):
        z, method = embed_2d(repr_data[:, bi], seed + bi)
        method_used.add(method)
        attr = BRANCH_COLOR_ATTR[branch]
        raw = np.asarray(attrs_data[:, attr_index[attr]], dtype=np.float64)
        finite = np.isfinite(raw)
        normalized = np.full(raw.shape, 0.5, dtype=np.float64)
        if finite.any():
            lo, hi = np.percentile(raw[finite], [2, 98])
            if hi > lo + 1e-12:
                normalized[finite] = np.clip(
                    (raw[finite] - lo) / (hi - lo), 0, 1
                )
            else:
                normalized[finite] = 0.5

        enhanced = normalized.copy()
        local_density = np.zeros_like(normalized, dtype=np.float64)
        if finite.any():
            enhanced_finite, dens_finite = _enhance_extreme_colors_by_local_density(
                z[finite], normalized[finite]
            )
            enhanced[finite] = enhanced_finite
            local_density[finite] = dens_finite

        dense = len(repr_data) > 100
        size = 80 if dense else 95
        base_face_alpha = 0.86 if dense else 0.92
        line_width = 0.75 if dense else 0.79

        faces = cmap(norm(enhanced))
        faces[:, 3] = base_face_alpha * (0.78 + 0.22 * local_density)
        if not finite.all():
            faces[~finite] = (0.78, 0.78, 0.78, base_face_alpha)
        ax.scatter(
            z[:, 0], z[:, 1], s=size,
            facecolors=faces,
            edgecolors=(0.12, 0.13, 0.15, 0.60),
            linewidths=line_width,
        )

        # Uniform type scale across figure, unlike V3's small panel titles.
        ax.set_title(f"({chr(ord('a') + bi)})", fontsize=font_size, pad=7)
        ax.set_box_aspect(panel_aspect)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.15)
            spine.set_edgecolor((0.18, 0.19, 0.22, 0.95))

    label = (
        "UMAP" if method_used == {"UMAP"}
        else "PCA" if method_used == {"PCA"}
        else "UMAP/PCA"
    )
    # Shorten only the main heading so it remains legible at 40+ pt.
    split_name = (
        "TRAIN+VALID+TEST" if "ALL" in dataset_label.upper()
        else dataset_label
    )
    fig.suptitle(
        f"PolyRTL | {split_name} | {label}",
        fontsize=font_size, y=0.995,
    )

    # One shared colorbar. Text sizes all equal to font_size.
    mapper = ScalarMappable(cmap=cmap, norm=norm)
    mapper.set_array([])
    cax = fig.add_axes([0.845, 0.175, 0.020, 0.640])
    bar = fig.colorbar(mapper, cax=cax)
    bar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    bar.set_label("Relative value", rotation=90, labelpad=13,
                  fontsize=font_size)
    bar.ax.tick_params(labelsize=font_size, pad=6)

    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return label



# ============================================================
# Sample sensitivity
# ============================================================

def plot_sensitivity_case_study(
    result,
    branch_names,
    out,
    dpi,
):
    idx = result[
        "top_indices"
    ]

    if len(idx) == 0:
        return

    m = result[
        "sensitivity"
    ][idx]

    names = result[
        "design_names"
    ][idx]

    denom = m.sum(
        axis=1,
        keepdims=True,
    )

    norm = np.divide(
        m,
        denom,
        out=np.zeros_like(m),
        where=denom > 1e-12,
    )

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            max(
                5.0,
                0.34 * len(idx) + 2.0,
            ),
        )
    )

    im = ax.imshow(
        norm,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(branch_names)
        )
    )

    ax.set_xticklabels(
        branch_names,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(names)
        )
    )

    ax.set_yticklabels(
        names,
        fontsize=8,
    )

    ax.set_title(
        "TEST case study: normalized prediction sensitivity to each branch\n"
        "(higher = masking this branch changes the trained 5-task predictor more)"
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.03,
        pad=0.03,
    )

    fig.tight_layout()

    fig.savefig(
        out.with_suffix(
            ".png"
        ),
        dpi=dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        out.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    seed_everything(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    run_dir = resolve_run_dir(
        args.run_dir
    )

    epoch = get_best_epoch(
        run_dir,
        args.epoch,
    )

    ckpt_file = (
        run_dir
        / "checkpoints"
        / f"ssl_epoch_{epoch:03d}.pt"
    )

    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Best SSL checkpoint does not exist: {ckpt_file}"
        )

    train_mod = load_training_module(
        args.train_script
    )

    ckpt = torch.load(
        ckpt_file,
        map_location="cpu",
        weights_only=False,
    )

    cfg = build_cfg(
        run_dir,
        ckpt,
        args,
    )

    model = instantiate_model(
        train_mod,
        cfg,
        ckpt,
        device,
    )

    branch_names = detect_branch_names(
        model
    )

    if args.output:
        out_dir = Path(
            args.output
        )
    else:
        out_dir = (
            run_dir
            / "visualizations"
            / f"best_ssl_{epoch:03d}"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        f"Run directory : {run_dir}"
    )
    print(
        f"Best SSL epoch: {epoch}"
    )
    print(
        f"Checkpoint    : {ckpt_file}"
    )
    print(
        f"Branches      : {branch_names}"
    )
    print(
        f"Device        : {device}"
    )
    print(
        f"Output        : {out_dir}"
    )
    print("=" * 88)

    # --------------------------------------------------------
    # Load all three splits.
    # --------------------------------------------------------

    train_ds = get_dataset(
        train_mod,
        cfg,
        "train",
    )

    valid_ds = get_dataset(
        train_mod,
        cfg,
        "valid",
    )

    test_ds = get_dataset(
        train_mod,
        cfg,
        "test",
    )

    # --------------------------------------------------------
    # Extract all three splits.
    # --------------------------------------------------------

    train_feat = extract_design_representations(
        train_mod,
        model,
        train_ds,
        device,
        "train",
        branch_names,
    )

    valid_feat = extract_design_representations(
        train_mod,
        model,
        valid_ds,
        device,
        "valid",
        branch_names,
    )

    test_feat = extract_design_representations(
        train_mod,
        model,
        test_ds,
        device,
        "test",
        branch_names,
    )

    # --------------------------------------------------------
    # Save extracted representations.
    # --------------------------------------------------------

    np.savez_compressed(
        out_dir
        / "branch_design_features.npz",

        train_designs=train_feat[
            "designs"
        ],

        valid_designs=valid_feat[
            "designs"
        ],

        test_designs=test_feat[
            "designs"
        ],

        train_cls=train_feat[
            "cls"
        ],

        valid_cls=valid_feat[
            "cls"
        ],

        test_cls=test_feat[
            "cls"
        ],

        train_mean=train_feat[
            "mean"
        ],

        valid_mean=valid_feat[
            "mean"
        ],

        test_mean=test_feat[
            "mean"
        ],

        train_attrs=train_feat[
            "attrs"
        ],

        valid_attrs=valid_feat[
            "attrs"
        ],

        test_attrs=test_feat[
            "attrs"
        ],

        branch_names=np.asarray(
            branch_names,
            dtype=object,
        ),

        structure_attributes=np.asarray(
            STRUCT_ATTRS,
            dtype=object,
        ),
    )

    # --------------------------------------------------------
    # 1) CKA — TEST ONLY
    # --------------------------------------------------------

    cka_cls = cka_matrix(
        test_feat[
            "cls"
        ]
    )

    cka_mean = cka_matrix(
        test_feat[
            "mean"
        ]
    )

    save_matrix_csv(
        out_dir
        / "01_cka_cls.csv",
        cka_cls,
        branch_names,
        branch_names,
    )

    save_matrix_csv(
        out_dir
        / "01b_cka_mean.csv",
        cka_mean,
        branch_names,
        branch_names,
    )

    plot_heatmap(
        cka_cls,
        branch_names,
        branch_names,
        "Branch representation similarity on TEST — Linear CKA (CLS)",
        out_dir
        / "01_cka_cls",
        args.dpi,
        cmap=make_strong_rwb_cmap(),
        annot_fontsize=32,     # Original 8pt -> 32pt (4x)
        tick_fontsize=40,      # Original 10pt -> 40pt (4x)
        title_fontsize=48,     # Original 12pt -> 48pt (4x)
        cbar_tick_fontsize=40, # Original 10pt -> 40pt (4x)
        color_min=0.0,
        color_max=1.0,
    )

    plot_heatmap(
        cka_mean,
        branch_names,
        branch_names,
        "Branch representation similarity on TEST — Linear CKA (mean real nodes)",
        out_dir
        / "01b_cka_mean",
        args.dpi,
    )

    # --------------------------------------------------------
    # 2) Structural probe — TRAIN -> TEST
    # --------------------------------------------------------

    probe = structural_probe_matrix(
        train_feat[
            "cls"
        ],
        test_feat[
            "cls"
        ],
        train_feat[
            "attrs"
        ],
        test_feat[
            "attrs"
        ],
        args.probe_alpha,
    )

    save_matrix_csv(
        out_dir
        / "02_structural_probe_test_r2.csv",
        probe,
        branch_names,
        STRUCT_ATTRS,
    )

    plot_heatmap(
        probe,
        branch_names,
        STRUCT_ATTRS,
        "Structural specialization — linear probe TEST R² (probe fit on TRAIN)",
        out_dir
        / "02_structural_probe_test_r2",
        args.dpi,
        center_zero=True,
    )

    # --------------------------------------------------------
    # 3) Branch ablation — TEST ONLY
    # --------------------------------------------------------

    (
        head,
        target_stats,
        standardize,
        head_ckpt,
    ) = load_joint_head(
        train_mod,
        run_dir,
        epoch,
        device,
    )

    ab = ablation_evaluation(
        train_mod,
        model,
        head,
        test_ds,
        device,
        branch_names,
        target_stats,
        standardize,
        args.top_k_cases,
    )

    save_matrix_csv(
        out_dir
        / "03_ablation_delta_r2.csv",
        ab[
            "delta_r2"
        ],
        branch_names,
        TASKS,
    )

    save_matrix_csv(
        out_dir
        / "03b_ablation_delta_pcc.csv",
        ab[
            "delta_pcc"
        ],
        branch_names,
        TASKS,
    )

    plot_heatmap(
        ab[
            "delta_r2"
        ],
        branch_names,
        TASKS,
        "TEST branch contribution — ΔR² after masking one trained branch\n"
        "(positive = branch removal hurts performance)",
        out_dir
        / "03_ablation_delta_r2",
        args.dpi,
        center_zero=True,
    )

    plot_heatmap(
        ab[
            "delta_pcc"
        ],
        branch_names,
        TASKS,
        "TEST branch contribution — ΔPCC after masking one trained branch\n"
        "(positive = branch removal hurts performance)",
        out_dir
        / "03b_ablation_delta_pcc",
        args.dpi,
        center_zero=True,
    )

    with (
        out_dir
        / "03_baseline_metrics.json"
    ).open(
        "w"
    ) as f:
        json.dump(
            {
                "ssl_epoch": epoch,
                "mlp_epoch": int(
                    head_ckpt.get(
                        "mlp_epoch",
                        -1,
                    )
                ),
                "baseline_test_log_r2": ab[
                    "baseline_r2"
                ],
                "baseline_test_log_pcc": ab[
                    "baseline_pcc"
                ],
                "note": (
                    "Branch ablation is inference-time masking of the "
                    "corresponding 48-D graph slice; neither encoder nor "
                    "MLP is retrained."
                ),
            },
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # 4A) Branch manifold — TEST ONLY
    # --------------------------------------------------------

    test_method = plot_branch_manifolds(
        test_feat[
            "cls"
        ],
        test_feat[
            "attrs"
        ],
        branch_names,
        out_dir
        / "04_branch_manifolds_test",
        args.dpi,
        args.seed,
        "TEST",
        figsize=args.manifold_figsize,
        font_size=args.manifold_font_size,
        panel_aspect=args.manifold_panel_aspect,
    )

    # --------------------------------------------------------
    # 4B) Branch manifold — ALL DESIGNS
    #     TRAIN + VALID + TEST
    # --------------------------------------------------------

    all_cls = np.concatenate(
        [
            train_feat[
                "cls"
            ],
            valid_feat[
                "cls"
            ],
            test_feat[
                "cls"
            ],
        ],
        axis=0,
    )

    all_attrs = np.concatenate(
        [
            train_feat[
                "attrs"
            ],
            valid_feat[
                "attrs"
            ],
            test_feat[
                "attrs"
            ],
        ],
        axis=0,
    )

    all_designs = np.concatenate(
        [
            train_feat[
                "designs"
            ],
            valid_feat[
                "designs"
            ],
            test_feat[
                "designs"
            ],
        ],
        axis=0,
    )

    all_method = plot_branch_manifolds(
        all_cls,
        all_attrs,
        branch_names,
        out_dir
        / "04b_branch_manifolds_all",
        args.dpi,
        args.seed,
        "ALL DESIGNS (TRAIN+VALID+TEST)",
        figsize=args.manifold_figsize,
        font_size=args.manifold_font_size,
        panel_aspect=args.manifold_panel_aspect,
    )

    # Save which design belongs to which split.
    with (
        out_dir
        / "04b_all_designs_index.csv"
    ).open(
        "w",
        newline="",
    ) as f:
        w = csv.writer(f)

        w.writerow(
            [
                "design",
                "split",
            ]
        )

        for name in train_feat[
            "designs"
        ]:
            w.writerow(
                [
                    name,
                    "train",
                ]
            )

        for name in valid_feat[
            "designs"
        ]:
            w.writerow(
                [
                    name,
                    "valid",
                ]
            )

        for name in test_feat[
            "designs"
        ]:
            w.writerow(
                [
                    name,
                    "test",
                ]
            )

    # --------------------------------------------------------
    # 5) Sample-wise sensitivity — TEST ONLY
    # --------------------------------------------------------

    with (
        out_dir
        / "05_samplewise_branch_sensitivity.csv"
    ).open(
        "w",
        newline="",
    ) as f:
        w = csv.writer(f)

        w.writerow(
            [
                "design"
            ]
            + list(
                branch_names
            )
        )

        for name, row in zip(
            ab[
                "design_names"
            ],
            ab[
                "sensitivity"
            ],
        ):
            w.writerow(
                [
                    name
                ]
                + [
                    float(x)
                    for x in row
                ]
            )

    plot_sensitivity_case_study(
        ab,
        branch_names,
        out_dir
        / "05_samplewise_branch_sensitivity",
        args.dpi,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "ssl_epoch": epoch,
        "ssl_checkpoint": str(
            ckpt_file
        ),
        "joint_mlp_checkpoint": str(
            run_dir
            / "joint_mlp"
            / f"ssl_epoch_{epoch:03d}"
            / "best.pt"
        ),
        "branch_names": list(
            branch_names
        ),
        "branch_dim": int(
            test_feat[
                "branch_dim"
            ]
        ),
        "num_train_designs": int(
            len(
                train_feat[
                    "designs"
                ]
            )
        ),
        "num_valid_designs": int(
            len(
                valid_feat[
                    "designs"
                ]
            )
        ),
        "num_test_designs": int(
            len(
                test_feat[
                    "designs"
                ]
            )
        ),
        "num_all_designs_for_visualization": int(
            len(
                all_designs
            )
        ),
        "structural_probe": (
            "Ridge probe fit on TRAIN branch CLS embeddings "
            "and evaluated on TEST."
        ),
        "cka": (
            "Computed only from TEST branch embeddings."
        ),
        "branch_ablation": (
            "Computed only on TEST by zeroing one trained 48-D "
            "graph branch at inference and feeding the fixed "
            "trained JointMLP."
        ),
        "test_manifold_method": test_method,
        "all_manifold_method": all_method,
        "all_manifold_note": (
            "ALL manifold concatenates TRAIN+VALID+TEST only for "
            "qualitative representation visualization. It is not "
            "used for model selection or quantitative evaluation."
        ),
        "outputs": [
            "01_cka_cls.{png,pdf,csv}",
            "01b_cka_mean.{png,pdf,csv}",
            "02_structural_probe_test_r2.{png,pdf,csv}",
            "03_ablation_delta_r2.{png,pdf,csv}",
            "03b_ablation_delta_pcc.{png,pdf,csv}",
            "04_branch_manifolds_test.{png,pdf}",
            "04b_branch_manifolds_all.{png,pdf}",
            "04b_all_designs_index.csv",
            "05_samplewise_branch_sensitivity.{png,pdf,csv}",
            "branch_design_features.npz",
        ],
    }

    with (
        out_dir
        / "analysis_summary.json"
    ).open(
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("Generated:")

    for x in summary[
        "outputs"
    ]:
        print(
            "  -",
            x,
        )

    print()
    print(
        f"TEST manifold designs : {len(test_feat['designs'])}"
    )

    print(
        f"ALL manifold designs  : {len(all_designs)} "
        f"(train={len(train_feat['designs'])}, "
        f"valid={len(valid_feat['designs'])}, "
        f"test={len(test_feat['designs'])})"
    )

    print()
    print(
        f"Done: {out_dir}"
    )


if __name__ == "__main__":
    main()
