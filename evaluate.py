"""
evaluate.py — Sliding-window inference + Dice evaluation on val or test set.

Usage:
  python evaluate.py --checkpoint outputs/sparsevol3d_5/best_model.pt --splits_file outputs/splits.json --split test
  python evaluate.py --checkpoint outputs/dense/best_model.pt --splits_file outputs/splits.json --split val --save_predictions
"""

import argparse
import dataclasses
import json
import os

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from models import UNet3D
from utils import compute_dice_volume


# ──────────────────────────────────────────────────────────────────────────────

def preprocess(img: np.ndarray) -> np.ndarray:
    """Match the preprocessing used in KiTS19Dataset."""
    img = np.clip(img, -200, 300)
    return ((img + 200.0) / 500.0).astype(np.float32)


def sliding_window_inference(
    model:       torch.nn.Module,
    volume:      np.ndarray,
    patch_size:  tuple,
    stride:      tuple,
    device:      torch.device,
    num_classes: int = 3,
) -> np.ndarray:
    """
    Aggregate class probability predictions over overlapping patches.

    Patches that extend past the boundary are shifted back so they stay
    inside the volume (no zero-padding of incomplete patches).

    Args:
        volume:     (D, H, W) float32 numpy array, preprocessed
        patch_size: (pd, ph, pw)
        stride:     (sd, sh, sw) — use patch_size // 2 for 50% overlap

    Returns:
        pred: (D, H, W) uint8 numpy array of predicted labels
    """
    D, H, W      = volume.shape
    pd, ph, pw   = patch_size
    sd, sh, sw   = stride

    pred_acc = np.zeros((num_classes, D, H, W), dtype=np.float32)
    count    = np.zeros((D, H, W), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for d in _sliding_range(D, pd, sd):
            for h in _sliding_range(H, ph, sh):
                for w in _sliding_range(W, pw, sw):
                    patch = volume[d:d+pd, h:h+ph, w:w+pw]
                    t = torch.from_numpy(patch[None, None]).to(device)
                    logits = model(t)                          # (1, C, pd, ph, pw)
                    probs  = torch.softmax(logits, dim=1)
                    probs  = probs.cpu().numpy()[0]            # (C, pd, ph, pw)

                    pred_acc[:, d:d+pd, h:h+ph, w:w+pw] += probs
                    count[d:d+pd, h:h+ph, w:w+pw]        += 1

    count = np.maximum(count, 1)
    return (pred_acc / count[None]).argmax(axis=0).astype(np.uint8)


def _sliding_range(size: int, patch: int, stride: int):
    """Yield start positions so that the last patch always fits inside size."""
    positions = list(range(0, size - patch + 1, stride))
    if not positions or positions[-1] + patch < size:
        positions.append(max(0, size - patch))
    return positions


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SparseVol3D evaluation")
    parser.add_argument("--checkpoint",       type=str, required=True)
    parser.add_argument("--data_dir",         type=str, default=None,
                        help="Override data_dir from checkpoint config")
    parser.add_argument("--output_dir",       type=str, default="outputs/predictions")
    parser.add_argument("--split",            type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--save_predictions", action="store_true",
                        help="Write predicted .nii.gz files to output_dir")
    parser.add_argument("--stride_factor",    type=float, default=0.5,
                        help="Sliding window stride as fraction of patch size (default 0.5)")
    parser.add_argument("--splits_file",      type=str, default="outputs/splits.json",
                        help="Path to splits JSON (produced by scripts/prepare_splits.py)")
    args = parser.parse_args()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt     = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNet3D(
        in_channels=1,
        num_classes=cfg_dict.get("num_classes", 3),
        base_channels=cfg_dict.get("base_channels", 32),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}  "
          f"Best mean Dice (val): {ckpt.get('best_dice', '?'):.4f}")

    # ── Resolve paths / splits ────────────────────────────────────────────────
    data_dir   = args.data_dir or cfg_dict.get("data_dir", "data/kits19/data")
    patch_size = tuple(cfg_dict.get("patch_size", [64, 128, 128]))
    stride     = tuple(max(1, int(p * args.stride_factor)) for p in patch_size)
    num_classes = cfg_dict.get("num_classes", 3)

    # Load splits from file if available, else fall back to Config defaults
    if os.path.exists(args.splits_file):
        with open(args.splits_file) as f:
            splits = json.load(f)
        cases = splits["val"] if args.split == "val" else splits["test"]
    else:
        cfg = Config()
        cases = cfg.val_cases if args.split == "val" else cfg.test_cases

    if args.save_predictions:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results     = []
    dice_per_class = {c: [] for c in range(1, num_classes)}

    for case_id in tqdm(cases, desc=f"Evaluating ({args.split})"):
        case_dir = os.path.join(data_dir, f"case_{case_id:05d}")
        img_path = os.path.join(case_dir, "imaging.nii.gz")
        seg_path = os.path.join(case_dir, "segmentation.nii.gz")

        if not os.path.exists(img_path):
            print(f"  [skip] case_{case_id:05d}: imaging.nii.gz not found")
            continue

        nii = nib.load(img_path)
        img = nii.get_fdata(dtype=np.float32)
        img = preprocess(img)

        pred = sliding_window_inference(
            model, img, patch_size, stride, device, num_classes
        )

        row = {"case_id": case_id}

        if os.path.exists(seg_path):
            seg  = nib.load(seg_path).get_fdata().astype(np.int64)
            dice = compute_dice_volume(pred, seg, num_classes=num_classes)
            for c in range(1, num_classes):
                dice_per_class[c].append(dice[c])
            row.update({f"dice_class_{c}": round(dice[c], 4)
                        for c in range(1, num_classes)})
            tqdm.write(
                f"  case_{case_id:05d}: kidney={dice.get(1, 0):.4f}  "
                f"tumor={dice.get(2, 0):.4f}"
            )

        if args.save_predictions:
            out_path = os.path.join(args.output_dir, f"case_{case_id:05d}_pred.nii.gz")
            pred_nii = nib.Nifti1Image(pred, nii.affine, nii.header)
            nib.save(pred_nii, out_path)

        results.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'-'*50}")
    print(f"Results on {args.split} split ({len(results)} cases)")
    print(f"{'-'*50}")

    label_names = {1: "Kidney", 2: "Tumor"}
    mean_dices  = []
    for c in range(1, num_classes):
        if dice_per_class[c]:
            m = float(np.mean(dice_per_class[c]))
            s = float(np.std(dice_per_class[c]))
            name = label_names.get(c, f"Class {c}")
            print(f"  {name:10s} Dice: {m:.4f} ± {s:.4f}")
            mean_dices.append(m)

    if mean_dices:
        print(f"  {'Mean':10s} Dice: {np.mean(mean_dices):.4f}")

    # Save results JSON
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "per_case": results,
        "summary": {
            f"class_{c}_mean_dice": float(np.mean(v))
            for c, v in dice_per_class.items() if v
        },
    }
    out_json = os.path.join(args.output_dir, f"results_{args.split}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to {out_json}")


if __name__ == "__main__":
    main()
