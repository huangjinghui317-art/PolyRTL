# PolyRTL

**Different Structures Through Different Lenses in RTL Representation Learning**

PolyRTL learns register-transfer-level (RTL) circuit representations for five PPA prediction tasks: Area, Power, register Slack, Worst Negative Slack (WNS), and Total Negative Slack (TNS). It builds on [TopoRTL](https://github.com/BUPT-GAMMA/TopoRTL)'s register-cone decomposition and dual-modal graph/functional-summary inputs. Eight specialized graph branches model complementary views of the register-interaction graph: **global, plain, fanout, fanin, chain, junction, depth, and reverse**. Their 384-dimensional structural output is combined with a 384-dimensional summary representation, and the encoder is trained with cross-modal self-supervision before joint PPA prediction.

The paper evaluates 115 RTL designs and 7,576 register cones on the TopoRTL benchmark. The released `data/processed/` files contain the precomputed graph and summary features needed to train PolyRTL; raw RTL preprocessing and tokenizer training are outside the scope of this repository.

## Repository contents

```text
data/processed/<split-ratio>/{train,valid,test}/  Preprocessed benchmark splits
models/                                         Model components
utils/                                          Dataset and training utilities
train.py                                        SSL and five-task PPA training
visualize.py                                    Analysis of a completed Scheme-2 run
results/scheme2/                                Saved metrics and downstream MLP heads
reference_environment.yml                       Reference Linux/CUDA environment
```

The paper uses the `0.3-0.3-0.4` circuit split (34 train, 34 validation, and 47 test designs). Its inputs are `data/processed/0.3-0.3-0.4/train/train.pkl`, `valid/valid.pkl`, and `test/test.pkl` under the same ratio directory. Other split ratios are also included. Dataset provenance and checksums are recorded in `data/audit.json`; the original benchmark is available from the [TopoRTL dataset](https://huggingface.co/datasets/ZhengDu/TopoRTL_Dataset).

## Checkpoint availability

**SSL encoder checkpoints are not included.** In particular, `results/scheme2/checkpoints/ssl_epoch_*.pt` is absent. The `results/scheme2/joint_mlp/ssl_epoch_*/best.pt` files in this repository are saved *downstream MLP heads*, not complete PolyRTL models. They require the matching SSL encoder checkpoint and cannot, by themselves, reproduce inference or the figures from `visualize.py`.

You can train a new encoder and predictor from the included preprocessed data. `train.py --stage all` creates the missing SSL checkpoints and fits a fresh five-task MLP after each SSL epoch. Commands that only evaluate or visualize existing checkpoints require you to generate those checkpoints first.

## Environment

The reference environment is a Python 3.10 Linux/CUDA 11.8 snapshot. On a compatible machine:

```bash
conda env create -f reference_environment.yml
conda activate polyrtl
```

The exported environment contains platform-specific package builds, so adapt it to your OS and CUDA installation if necessary. The inherited TopoRTL modules import both DGL and PyTorch Geometric, even though PolyRTL's new branch message passing uses native PyTorch; both dependencies must be available. `train.py` selects CUDA when available and otherwise runs on CPU; a GPU is recommended for the full experiment.

## Train from scratch

Run commands from the repository root. This example matches the paper's Scheme-2 training budget: 50 SSL epochs at `1e-4`, with a newly initialized joint predictor trained for 100 epochs at `1e-3` after each SSL epoch.

```bash
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
  --output runs/polyrtl
```

`--output` is a parent directory; the command writes to `runs/polyrtl/scheme2/`. Key outputs are:

- `checkpoints/ssl_epoch_XXX.pt`: newly trained SSL encoder checkpoints.
- `joint_mlp/ssl_epoch_XXX/best.pt`: validation-selected downstream heads for each SSL epoch.
- `best_summary.csv`: per-task results selected using validation data.
- `best_joint_epoch.json`: the epoch selected by the joint validation score.

For separate stages, first run `--stage ssl` to generate encoder checkpoints, then run `--stage evaluate` with the **same** `--output`, `--scheme`, `--data-root`, `--split-ratio`, and architecture settings. Set `--mlp-epochs 100` during evaluation to match the paper. `--stage evaluate` cannot run on a fresh checkout because the SSL checkpoints are not distributed. To continue an interrupted run, repeat the original training command with `--resume`.

Once a complete Scheme-2 run exists, generate the branch analyses with:

```bash
python visualize.py \
  --run-dir runs/polyrtl/scheme2 \
  --train-script train.py
```

## Paper results

The paper reports the following test-set results in signed-log space after reversing target standardization. Each task uses its validation-selected SSL epoch; see `results/scheme2/best_summary.csv` for the saved metrics.

| Task | PCC | R² |
| --- | ---: | ---: |
| Area | 0.878 | 0.757 |
| Power | 0.887 | 0.748 |
| Slack | 0.871 | 0.741 |
| WNS | 0.858 | 0.726 |
| TNS | 0.892 | 0.788 |

## Citation and license

If you use this work, please cite *PolyRTL: Different Structures Through Different Lenses in RTL Representation Learning* by Jinghui Huang, Chang Han, Sining Wang, Shiqi Zhao, Jie Yang, and Mohamad Sawan. The repository is released under the [MIT License](LICENSE).
