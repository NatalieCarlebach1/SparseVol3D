# SparseVol3D

**Sparse Axial Supervision for 3D Medical Image Segmentation
with Volumetric Interpolation Consistency**

A clean, reproducible research prototype for annotation-efficient 3D CT segmentation.
Tested on [KiTS19](https://kits19.grand-challenge.org/) (kidney & tumor segmentation).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NatalieCarlebach1/SparseVol3D/blob/master/colab_setup.ipynb)

---

## The Idea in One Paragraph

Annotating 3D CT volumes is expensive — a single scan can have 400+ axial slices.
**SparseVol3D** trains a 3D U-Net with labels on only a fraction of those slices
(e.g., every 5th or 10th). The key innovation is a
**Volumetric Interpolation Consistency (VIC) loss**: the network's predictions at
unlabeled slices should lie on the linear interpolation between the nearest labeled
neighbors. This is inspired by 3D Gaussian Splatting's insight that a good
volumetric representation is a *smooth, continuous field* between observations —
not just accurate at sparse anchor points.

---

## Method

```
 Input volume (D × H × W)
        │
   3D U-Net (encoder-decoder, skip connections)
        │
   Logits  (C × D × H × W)
        │
   ┌──────────────┬─────────────────────┐
   │              │                     │
Labeled        Unlabeled              VIC loss
slices         slices          (interpolation target)
   │                                    │
CE + Dice loss  ←──── L_total ──────────┘
(supervised)      = L_sup + λ · L_VIC
```

**Volumetric Interpolation Consistency (VIC):**

For an unlabeled slice at depth `z`, bounded by labeled slices `z0` and `z1`:

```
L_VIC(z) = || p(z) - [ (1-α)·p(z0) + α·p(z1) ] ||²    where α = (z-z0)/(z1-z0)
```

`p(·)` is the softmax probability at that depth position.
Anchors `p(z0)` and `p(z1)` are **detached** — we regularize unlabeled slices only.

---

## Repository Structure

```
SparseVol3D/
├── config.py                  # All hyperparameters in one dataclass
├── train.py                   # Training script (argparse CLI)
├── evaluate.py                # Sliding-window inference + Dice evaluation
├── data/
│   └── kits_dataset.py        # Lazy-loading dataset; generates sparse label masks
├── models/
│   └── unet3d.py              # 3D U-Net (4 levels + bottleneck, ~19 M params)
├── losses/
│   └── sparse_supervision.py  # Sparse CE+Dice loss and VIC loss
├── utils/
│   └── metrics.py             # Per-class Dice coefficient
├── scripts/
│   └── prepare_splits.py      # Generate train/val/test splits JSON
├── requirements.txt
└── research_writeup.md        # 3-page paper draft
```

---

## Setup

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate sparsevol3d
```

Installs PyTorch with CUDA 12.1. If you have **no NVIDIA GPU**, remove the
`pytorch-cuda=12.1` line from `environment.yml` first.

### Option B — pip

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.0+.

### Verify

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. Get data

#### Option A — Synthetic data (quick test, no download needed)
```bash
python scripts/make_synthetic_data.py --n_cases 10 --data_dir data/kits19/data
```
Generates 10 fake NIfTI volumes in seconds. Good for verifying the pipeline.

#### Option B — Real KiTS19 data (~50 GB)

```bash
git clone https://github.com/neheller/kits19
cd kits19
pip install -r requirements.txt
python starter_code/get_imaging.py     # downloads ~50 GB
cd ..
```

Point the scripts at the data directory:

```bash
# Either symlink:
mkdir -p data/kits19
ln -s $(pwd)/kits19/data data/kits19/data

# Or pass --data_dir to each script directly.
```

### 3. Generate train/val/test splits

```bash
mkdir -p outputs
python scripts/prepare_splits.py --data_dir data/kits19/data
# → outputs/splits.json  (160 train / 25 val / 25 test)
```

---

## Training

### CPU smoke test (no GPU needed)
```bash
python train.py --data_dir data/kits19/data --output_dir outputs/debug --debug
```
`--debug` uses a tiny model (base_channels=8, patch 16×64×64) and runs 3 epochs
in ~2 minutes on CPU. Triggered automatically if no GPU is detected.

---

All GPU experiments use the same 3D U-Net. Only `--label_stride` and `--lambda_vic` vary.

### Dense supervision (oracle baseline)
```bash
python train.py \
  --data_dir data/kits19/data \
  --label_stride 1 \
  --lambda_vic 0.0 \
  --output_dir outputs/dense
```

### Sparse supervision only (no VIC)
```bash
python train.py \
  --data_dir data/kits19/data \
  --label_stride 5 \
  --lambda_vic 0.0 \
  --output_dir outputs/sparse_stride5
```

### SparseVol3D — sparse + VIC (the proposed method)
```bash
python train.py \
  --data_dir data/kits19/data \
  --label_stride 5 \
  --lambda_vic 0.1 \
  --output_dir outputs/sparsevol3d_stride5
```

### Run all ablation experiments
```bash
for stride in 1 2 5 10 20; do
  for vic in 0.0 0.1; do
    python train.py \
      --data_dir data/kits19/data \
      --label_stride $stride \
      --lambda_vic $vic \
      --output_dir outputs/s${stride}_vic${vic}
  done
done
```

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--label_stride` | 1 | Annotate every N-th axial slice |
| `--lambda_vic` | 0.1 | VIC loss weight (0 = disable) |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 2 | Batch size |
| `--base_channels` | 32 | Model width |
| `--no_amp` | — | Disable mixed precision |

---

## Evaluation

```bash
python evaluate.py \
  --checkpoint  outputs/sparsevol3d_stride5/best_model.pt \
  --data_dir    data/kits19/data \
  --splits_file outputs/splits.json \
  --split       test
```

Save predicted segmentation volumes:
```bash
python evaluate.py \
  --checkpoint  outputs/sparsevol3d_stride5/best_model.pt \
  --data_dir    data/kits19/data \
  --splits_file outputs/splits.json \
  --split       test \
  --save_predictions \
  --output_dir  outputs/predictions
```

---

## Expected Results

| Method | Label stride | Annot. % | Kidney Dice | Tumor Dice |
|--------|:---:|:---:|:---:|:---:|
| Dense U-Net (oracle) | 1 | 100% | ~0.95 | ~0.75 |
| Sparse (no VIC) | 5 | 20% | ~0.88 | ~0.60 |
| **SparseVol3D** | 5 | 20% | **~0.91** | **~0.65** |
| Sparse (no VIC) | 10 | 10% | ~0.80 | ~0.50 |
| **SparseVol3D** | 10 | 10% | **~0.85** | **~0.57** |

*Expected ranges from comparable literature. Your numbers will vary by hardware, split, and training duration.*

---

## Hardware Requirements

| Component | Recommended |
|---|---|
| GPU VRAM | ≥ 8 GB (tested on RTX 3090 / A100) |
| Disk | ~50 GB for KiTS19 |
| RAM | 16 GB |

With `--no_amp` disabled (default), mixed precision halves VRAM usage.
To reduce VRAM further, lower `--base_channels 16` or `--batch_size 1`.

---

## Extending This Work

The codebase is intentionally modular:

- **Swap the model**: replace `models/unet3d.py` with any (B, 1, D, H, W) → (B, C, D, H, W) network.
- **Try other consistency losses**: modify `losses/sparse_supervision.py` — e.g., replace linear interpolation with a learned interpolation network.
- **Add pseudo-labeling**: after initial training, generate pseudo-labels on unlabeled slices and retrain.
- **Different annotation patterns**: the `label_stride` logic in `data/kits_dataset.py` is easy to replace (e.g., random sparse selection instead of uniform stride).

---

## Citation

If you use this code, please cite KiTS19:

```bibtex
@article{heller2021state,
  title   = {The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging},
  author  = {Heller, Nicholas and others},
  journal = {Medical Image Analysis},
  year    = {2021}
}
```

---

## License

MIT
