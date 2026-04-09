# SparseVol3D — Claude Context

## Project
Medical imaging thesis prototype. 3D CT segmentation with sparse axial supervision.
Dataset: KiTS19. Target venue: MIDL short paper or MICCAI workshop.

## Stack
- Python 3.9+, PyTorch 2.0+, nibabel, CUDA GPU required
- No unnecessary abstractions — keep code clean and minimal

## Key ideas
1. **Sparse axial supervision**: train with labels on only every K-th axial slice.
2. **Volumetric Interpolation Consistency (VIC) loss** — predictions at unlabeled
   slices should match linear interpolation between nearest labeled neighbors.
   Inspired by 3D Gaussian Splatting's continuous field principle.
3. **NeRF-inspired CoordMLP** (optional, `--use_coord_mlp`) — fuses sinusoidal
   positional encoding of (x,y,z) voxel coordinates into U-Net decoder features.
   Adds ~8K parameters; gives the network explicit position awareness.

## Structure
- `config.py` — all hyperparameters (includes use_coord_mlp, coord_features, coord_freq_bands)
- `train.py` — training loop
- `evaluate.py` — sliding-window inference + Dice
- `data/kits_dataset.py` — lazy-loading dataset, sparse mask generation
- `models/unet3d.py` — 3D U-Net (~19M params, base_channels=32)
- `models/coord_mlp.py` — NeRF positional encoding + CoordMLP module
- `losses/sparse_supervision.py` — sparse CE+Dice + VIC loss
- `utils/metrics.py` — Dice coefficient
- `scripts/prepare_splits.py` — train/val/test split

## Run commands
```bash
pip install -r requirements.txt
python scripts/prepare_splits.py --data_dir data/kits19/data
# Baseline
python train.py --label_stride 5 --lambda_vic 0.1 --output_dir outputs/sparsevol3d_5
# With CoordMLP
python train.py --label_stride 5 --lambda_vic 0.1 --use_coord_mlp --output_dir outputs/sparsevol3d_5_coord
# Evaluate
python evaluate.py --checkpoint outputs/sparsevol3d_5/best_model.pt --splits_file outputs/splits.json --split test
# All ablations (add --coord_mlp to also run CoordMLP variants)
python run_experiments.py --data_dir data/kits19/data --coord_mlp
```

## Preferences
- Simple, modular, readable code over clever code
- No docstrings or comments on unchanged code
- Short responses are fine
