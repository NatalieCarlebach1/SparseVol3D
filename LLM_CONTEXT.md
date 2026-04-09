# LLM_CONTEXT.md — SparseVol3D

This file gives a complete picture of the project so any LLM or developer
can pick it up and run it from scratch without prior context.

---

## What this project does

**SparseVol3D** trains a 3D U-Net on CT kidney scans (KiTS19 dataset) using
only a fraction of annotated axial slices. The innovation is the
**Volumetric Interpolation Consistency (VIC) loss**: predictions at unlabeled
slices must match the linear interpolation between nearby labeled neighbors —
inspired by 3D Gaussian Splatting's continuous field principle.

**Task**: 3D voxel segmentation. Classes: 0=background, 1=kidney, 2=tumor.
**Metric**: Dice coefficient per class (kidney, tumor).

---

## Environment setup

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate sparsevol3d
```

This installs PyTorch with CUDA 12.1. If you have no NVIDIA GPU, remove the
`pytorch-cuda=12.1` line from `environment.yml` before running.

### Option B — pip

```bash
pip install -r requirements.txt
```

### Verify install

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Data setup

### Real data (KiTS19 — recommended for full training)

```bash
git clone https://github.com/neheller/kits19.git
cd kits19
pip install -r requirements.txt
python starter_code/get_imaging.py    # downloads ~50 GB, takes ~1 hour
cd ..
mkdir -p data/kits19
ln -s $(pwd)/kits19/data data/kits19/data   # or copy
```

### Synthetic data (for smoke tests / no internet)

```bash
python scripts/make_synthetic_data.py --n_cases 10 --data_dir data/kits19/data
```

Generates 10 NIfTI volumes (96×192×192) with fake kidney/tumor blobs.
Enough to verify the pipeline end-to-end in minutes.

### Generate splits

```bash
python scripts/prepare_splits.py --data_dir data/kits19/data
# → outputs/splits.json  (train / val / test indices)
```

---

## Running the code

### CPU (debug/smoke test — runs in ~2 minutes)

```bash
python train.py \
  --data_dir data/kits19/data \
  --output_dir outputs/debug \
  --debug
```

`--debug` is set automatically when no GPU is detected. It uses:
- patch size 16×64×64
- base_channels=8
- 3 epochs, 2 training cases

### GPU (full training)

```bash
# Dense baseline (oracle):
python train.py --label_stride 1 --lambda_vic 0.0 --output_dir outputs/dense

# SparseVol3D — every 5th slice + VIC loss:
python train.py --label_stride 5 --lambda_vic 0.1 --output_dir outputs/sparsevol3d_5

# Sparse only, no VIC (ablation):
python train.py --label_stride 5 --lambda_vic 0.0 --output_dir outputs/sparse_5
```

### All ablation experiments

```bash
for stride in 1 2 5 10 20; do
  python train.py --label_stride $stride --lambda_vic 0.1 \
                  --output_dir outputs/s${stride}_vic
  python train.py --label_stride $stride --lambda_vic 0.0 \
                  --output_dir outputs/s${stride}_novic
done
```

### Evaluation

```bash
python evaluate.py \
  --checkpoint  outputs/sparsevol3d_5/best_model.pt \
  --data_dir    data/kits19/data \
  --splits_file outputs/splits.json \
  --split       test

# Save prediction .nii.gz files:
python evaluate.py \
  --checkpoint  outputs/sparsevol3d_5/best_model.pt \
  --splits_file outputs/splits.json \
  --split       test --save_predictions --output_dir outputs/predictions
```

---

## Key files and what they do

| File | Role |
|------|------|
| `config.py` | Single dataclass with all hyperparameters |
| `train.py` | Training loop. Auto-detects CUDA; falls back to CPU debug mode |
| `evaluate.py` | Sliding-window inference + per-class Dice |
| `data/kits_dataset.py` | Lazy-loads NIfTI files, random 3D crop, generates sparse label mask |
| `models/unet3d.py` | 3D U-Net (4 encoder levels + bottleneck, skip connections) |
| `losses/sparse_supervision.py` | CE+Dice on labeled slices; VIC loss on unlabeled slices |
| `utils/metrics.py` | Dice coefficient for torch tensors and numpy arrays |
| `scripts/make_synthetic_data.py` | Generate fake NIfTI data for testing |
| `scripts/prepare_splits.py` | Create train/val/test split JSON |

---

## Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `label_stride` | 1 | Annotate every N-th axial slice (1=dense, 5=20%) |
| `lambda_vic` | 0.1 | Weight for VIC loss. 0 = plain sparse supervision |
| `patch_size` | (64,128,128) | 3D crop size during training |
| `base_channels` | 32 | U-Net width (~19M params) |
| `epochs` | 100 | Training epochs |
| `batch_size` | 2 | — |
| `lr` | 1e-3 | AdamW learning rate |

---

## The VIC loss — core innovation

```python
# losses/sparse_supervision.py: volumetric_consistency_loss()

# For each pair of consecutive labeled slices z0, z1:
for z in range(z0 + 1, z1):
    alpha  = (z - z0) / (z1 - z0)
    target = ((1 - alpha) * p(z0) + alpha * p(z1)).detach()
    loss  += MSE(p(z), target)
```

- `p(z)` = softmax probabilities at depth z
- Anchors `p(z0)` and `p(z1)` are detached — only unlabeled slices are regularized
- Zero overhead at inference

---

## Expected outputs

After training `outputs/sparsevol3d_5/` will contain:
```
best_model.pt          # best checkpoint by val mean Dice
ckpt_epoch010.pt       # periodic checkpoints
config.json            # full config used for this run
train_log.json         # loss + Dice per epoch
```

After evaluation `outputs/predictions/` will contain:
```
results_test.json              # per-case Dice scores
case_00185_pred.nii.gz         # predicted segmentation volumes (if --save_predictions)
...
```

---

## Hardware requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | 8 GB VRAM | 24 GB (A100/RTX 3090) |
| RAM | 16 GB | 32 GB |
| Disk | 60 GB | 100 GB |
| Python | 3.9+ | 3.11 |

---

## Common issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: nibabel` | `pip install nibabel` |
| `RuntimeError: No valid cases found` | Check `--data_dir` path; run `make_synthetic_data.py` |
| OOM on GPU | Reduce `--batch_size 1` or `--base_channels 16` |
| Slow on CPU | Use `--debug` flag; real training needs a GPU |
| `autocast` error | Add `--no_amp` flag |
