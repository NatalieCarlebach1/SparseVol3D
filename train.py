"""
train.py — SparseVol3D training script

Usage examples:

  # Dense baseline (full supervision):
  python train.py --label_stride 1 --lambda_vic 0.0 --output_dir outputs/dense

  # Sparse supervision only (no VIC):
  python train.py --label_stride 5 --lambda_vic 0.0 --output_dir outputs/sparse_5

  # SparseVol3D (sparse + VIC):
  python train.py --label_stride 5 --lambda_vic 0.1 --output_dir outputs/sparsevol3d_5
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data import KiTS19Dataset
from losses import combined_loss
from models import UNet3D
from utils import compute_dice


# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config_from_args(args) -> Config:
    cfg = Config()

    # Override any field supplied on the command line
    for key, val in vars(args).items():
        if hasattr(cfg, key) and val is not None:
            setattr(cfg, key, val)

    # Load case splits from file if provided
    if args.splits_file and os.path.exists(args.splits_file):
        with open(args.splits_file) as f:
            splits = json.load(f)
        cfg.train_cases = splits["train"]
        cfg.val_cases   = splits["val"]
        cfg.test_cases  = splits["test"]

    return cfg


# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, lambda_vic):
    model.train()
    epoch_loss = 0.0

    for img, seg, mask in tqdm(loader, desc="  train", leave=False, unit="batch"):
        img  = img.to(device)
        seg  = seg.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=scaler is not None):
            logits = model(img)
            loss   = combined_loss(logits, seg, mask, lambda_vic=lambda_vic)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, num_classes=3):
    model.eval()
    all_dice = {c: [] for c in range(1, num_classes)}   # skip background

    for img, seg, _ in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
        img = img.to(device)
        seg = seg.to(device)

        logits = model(img)
        pred   = logits.argmax(dim=1)     # (B, D, H, W)

        for b in range(pred.shape[0]):
            d = compute_dice(pred[b], seg[b], num_classes=num_classes)
            for c in range(1, num_classes):
                all_dice[c].append(d[c])

    return {c: float(np.mean(v)) for c, v in all_dice.items()}


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SparseVol3D training")
    parser.add_argument("--data_dir",     type=str,   default=None)
    parser.add_argument("--output_dir",   type=str,   default=None)
    parser.add_argument("--splits_file",  type=str,   default="outputs/splits.json",
                        help="Path to splits JSON produced by scripts/prepare_splits.py")
    parser.add_argument("--label_stride", type=int,   default=None,
                        help="Label every N-th axial slice. 1=dense, 5=20%%, 10=10%%")
    parser.add_argument("--lambda_vic",   type=float, default=None,
                        help="VIC loss weight. 0 = no consistency regularization")
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--base_channels",type=int,   default=None)
    parser.add_argument("--seed",         type=int,   default=None)
    parser.add_argument("--no_amp",       action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--debug",        action="store_true",
                        help="CPU debug mode: tiny model + 2 cases + 3 epochs")
    parser.add_argument("--use_coord_mlp", action="store_true",
                        help="Add NeRF-inspired CoordMLP to U-Net decoder")
    parser.add_argument("--coord_features",   type=int, default=None,
                        help="Output channels from CoordMLP (default 16)")
    parser.add_argument("--coord_freq_bands", type=int, default=None,
                        help="Sinusoidal frequency bands in CoordMLP (default 6)")
    args = parser.parse_args()

    cfg = build_config_from_args(args)
    if args.no_amp:
        cfg.amp = False
    if args.use_coord_mlp:
        cfg.use_coord_mlp = True
    if args.coord_features is not None:
        cfg.coord_features = args.coord_features
    if args.coord_freq_bands is not None:
        cfg.coord_freq_bands = args.coord_freq_bands

    # ── Auto-detect device ────────────────────────────────────────────────────
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    if not cuda_available:
        print("WARNING: CUDA not available — running on CPU.")
        print("         For full training use a CUDA-enabled GPU.")
        print("         Switching to CPU debug mode (small patch/model).\n")
        args.debug = True

    # ── CPU debug mode overrides ──────────────────────────────────────────────
    if args.debug:
        cfg.patch_size    = (16, 64, 64)
        cfg.base_channels = 8
        cfg.batch_size    = 1
        cfg.epochs        = 3
        cfg.num_workers   = 0           # no multiprocessing on CPU
        cfg.amp           = False       # AMP requires CUDA
        cfg.log_interval  = 1
        cfg.save_interval = 3
        # Use only first 2 train cases and 1 val case for speed
        cfg.train_cases   = cfg.train_cases[:2]
        cfg.val_cases     = cfg.val_cases[:1]
        print("DEBUG MODE: patch=(16,64,64), base_channels=8, epochs=3, 2 train cases\n")

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Device       : {device}")
    print(f"Label stride : {cfg.label_stride}  "
          f"({100 / cfg.label_stride:.0f}% annotation budget)")
    print(f"VIC lambda   : {cfg.lambda_vic}")
    print(f"AMP          : {cfg.amp and cuda_available}")
    print()

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = KiTS19Dataset(
        cfg.data_dir, cfg.train_cases,
        patch_size=cfg.patch_size,
        label_stride=cfg.label_stride,
        mode="train",
        crops_per_case=cfg.crops_per_case,
    )
    val_ds = KiTS19Dataset(
        cfg.data_dir, cfg.val_cases,
        patch_size=cfg.patch_size,
        label_stride=1,           # always evaluate with dense labels
        mode="val",
    )

    pin = cuda_available
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=pin,
    )

    print(f"Train cases  : {len(cfg.train_cases)}  ({len(train_ds)} patch samples)")
    print(f"Val cases    : {len(cfg.val_cases)}")
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels=1,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        use_coord_mlp=cfg.use_coord_mlp,
        coord_features=cfg.coord_features,
        coord_freq_bands=cfg.coord_freq_bands,
    ).to(device)
    if cfg.use_coord_mlp:
        print(f"CoordMLP      : ON  (features={cfg.coord_features}, freq_bands={cfg.coord_freq_bands})")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params : {n_params / 1e6:.2f} M")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler    = GradScaler("cuda") if cfg.amp and cuda_available else None

    # ── Save config ───────────────────────────────────────────────────────────
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        # Config is a dataclass; convert to dict manually
        import dataclasses
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_dice = 0.0
    log_rows  = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg.lambda_vic
        )
        scheduler.step()

        row = {"epoch": epoch, "train_loss": round(train_loss, 6)}

        if epoch % cfg.log_interval == 0 or epoch == cfg.epochs:
            val_dice = validate(model, val_loader, device, cfg.num_classes)
            # val_dice keys are class indices 1, 2
            kidney_dice = val_dice.get(1, 0.0)
            tumor_dice  = val_dice.get(2, 0.0)
            mean_dice   = (kidney_dice + tumor_dice) / 2.0

            row.update({
                "kidney_dice": round(kidney_dice, 4),
                "tumor_dice":  round(tumor_dice,  4),
                "mean_dice":   round(mean_dice,   4),
            })

            print(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"loss {train_loss:.4f} | "
                f"kidney {kidney_dice:.4f} | "
                f"tumor {tumor_dice:.4f} | "
                f"mean {mean_dice:.4f}"
            )

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_dice": best_dice,
                        "config": dataclasses.asdict(cfg),
                    },
                    os.path.join(cfg.output_dir, "best_model.pt"),
                )
        else:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | loss {train_loss:.4f}")

        log_rows.append(row)

        if epoch % cfg.save_interval == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "config": dataclasses.asdict(cfg)},
                os.path.join(cfg.output_dir, f"ckpt_epoch{epoch:03d}.pt"),
            )

    # ── Save training log ─────────────────────────────────────────────────────
    with open(os.path.join(cfg.output_dir, "train_log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)

    print(f"\nDone. Best mean Dice: {best_dice:.4f}")
    print(f"Best checkpoint saved to {cfg.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
