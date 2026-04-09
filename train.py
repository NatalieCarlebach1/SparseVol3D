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
from torch.cuda.amp import GradScaler, autocast
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

        with autocast(enabled=scaler is not None):
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
    args = parser.parse_args()

    cfg = build_config_from_args(args)
    if args.no_amp:
        cfg.amp = False

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Label stride : {cfg.label_stride}  "
          f"({100 / cfg.label_stride:.0f}% annotation budget)")
    print(f"VIC lambda   : {cfg.lambda_vic}")
    print(f"AMP          : {cfg.amp}")
    print()

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = KiTS19Dataset(
        cfg.data_dir, cfg.train_cases,
        patch_size=cfg.patch_size,
        label_stride=cfg.label_stride,
        mode="train",
    )
    val_ds = KiTS19Dataset(
        cfg.data_dir, cfg.val_cases,
        patch_size=cfg.patch_size,
        label_stride=1,           # always evaluate with dense labels
        mode="val",
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
    )

    print(f"Train cases  : {len(cfg.train_cases)}  "
          f"→  {len(train_ds)} patch samples")
    print(f"Val cases    : {len(cfg.val_cases)}")
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels=1,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params : {n_params / 1e6:.2f} M")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler    = GradScaler() if cfg.amp and device.type == "cuda" else None

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
    print(f"Best checkpoint → {cfg.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
