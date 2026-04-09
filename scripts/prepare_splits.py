"""
scripts/prepare_splits.py

Scan the KiTS19 data directory, find all cases with both imaging and
segmentation files, shuffle deterministically, and write a train/val/test
split JSON to outputs/splits.json.

Usage:
  python scripts/prepare_splits.py --data_dir data/kits19/data
  python scripts/prepare_splits.py --data_dir data/kits19/data --seed 0
"""

import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/kits19/data")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   type=str, default="outputs/splits.json")
    parser.add_argument("--train_frac", type=float, default=0.76)
    parser.add_argument("--val_frac",   type=float, default=0.12)
    # test_frac = 1 - train_frac - val_frac
    args = parser.parse_args()

    random.seed(args.seed)

    # Collect annotated cases
    available = []
    for i in range(210):                       # KiTS19 has 210 labeled cases
        case_dir = os.path.join(args.data_dir, f"case_{i:05d}")
        has_img = os.path.exists(os.path.join(case_dir, "imaging.nii.gz"))
        has_seg = os.path.exists(os.path.join(case_dir, "segmentation.nii.gz"))
        if has_img and has_seg:
            available.append(i)

    if not available:
        print(f"No cases found in {args.data_dir}.")
        print("Download KiTS19 data first (see README).")
        return

    print(f"Found {len(available)} annotated cases")

    random.shuffle(available)
    n       = len(available)
    n_train = int(args.train_frac * n)
    n_val   = int(args.val_frac   * n)

    splits = {
        "train": sorted(available[:n_train]),
        "val":   sorted(available[n_train : n_train + n_val]),
        "test":  sorted(available[n_train + n_val :]),
    }

    print(f"Split  :  {len(splits['train'])} train | "
          f"{len(splits['val'])} val | {len(splits['test'])} test")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved to {args.output}")


if __name__ == "__main__":
    main()
