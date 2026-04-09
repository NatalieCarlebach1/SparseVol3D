"""
run_experiments.py — Run all ablation experiments in sequence.

Trains one model per (label_stride, lambda_vic, use_coord_mlp) combination,
saves checkpoints, then prints a summary results table.

Usage:
  python run_experiments.py --data_dir data/kits19/data
  python run_experiments.py --data_dir data/kits19/data --epochs 50 --quick
  python run_experiments.py --data_dir data/kits19/data --coord_mlp   # also run CoordMLP variants
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime


EXPERIMENTS = [
    # (label_stride, lambda_vic, use_coord_mlp, name)
    (1,  0.0, False, "dense_baseline"),
    (2,  0.0, False, "sparse_stride2_novic"),
    (2,  0.1, False, "sparsevol3d_stride2"),
    (5,  0.0, False, "sparse_stride5_novic"),
    (5,  0.1, False, "sparsevol3d_stride5"),
    (10, 0.0, False, "sparse_stride10_novic"),
    (10, 0.1, False, "sparsevol3d_stride10"),
    (20, 0.0, False, "sparse_stride20_novic"),
    (20, 0.1, False, "sparsevol3d_stride20"),
]

# CoordMLP variants: SparseVol3D + NeRF-inspired coordinate field
COORD_MLP_EXPERIMENTS = [
    # (label_stride, lambda_vic, use_coord_mlp, name)
    (1,  0.0, True, "dense_coord"),
    (5,  0.0, True, "sparse_stride5_coord"),
    (5,  0.1, True, "sparsevol3d_stride5_coord"),
    (10, 0.1, True, "sparsevol3d_stride10_coord"),
]


def run_experiment(name, label_stride, lambda_vic, use_coord_mlp, data_dir, output_root, epochs, extra_args):
    output_dir = os.path.join(output_root, name)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  label_stride={label_stride}  lambda_vic={lambda_vic}  coord_mlp={use_coord_mlp}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "train.py",
        "--data_dir",     data_dir,
        "--output_dir",   output_dir,
        "--label_stride", str(label_stride),
        "--lambda_vic",   str(lambda_vic),
        "--epochs",       str(epochs),
    ] + extra_args

    if use_coord_mlp:
        cmd.append("--use_coord_mlp")

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0, output_dir


def read_best_dice(output_dir):
    log_path = os.path.join(output_dir, "train_log.json")
    if not os.path.exists(log_path):
        return None, None, None

    with open(log_path) as f:
        log = json.load(f)

    best = max(
        (row for row in log if "mean_dice" in row),
        key=lambda r: r["mean_dice"],
        default=None,
    )
    if best is None:
        return None, None, None
    return best.get("kidney_dice"), best.get("tumor_dice"), best.get("mean_dice")


def main():
    parser = argparse.ArgumentParser(description="SparseVol3D ablation runner")
    parser.add_argument("--data_dir",    type=str, default="data/kits19/data")
    parser.add_argument("--output_root", type=str, default="outputs/experiments")
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--quick",       action="store_true",
                        help="Quick run: 10 epochs, base_channels=16 (for testing)")
    parser.add_argument("--debug",       action="store_true",
                        help="CPU debug mode (tiny model, 3 epochs)")
    parser.add_argument("--strides",     type=int, nargs="+", default=None,
                        help="Only run experiments with these label strides, e.g. --strides 1 5 10")
    parser.add_argument("--coord_mlp",   action="store_true",
                        help="Also run CoordMLP variants (NeRF-inspired coordinate field)")
    args = parser.parse_args()

    extra_args = []
    if args.quick:
        args.epochs = 10
        extra_args += ["--base_channels", "16"]
    if args.debug:
        extra_args += ["--debug"]

    # Build experiment list
    experiments = list(EXPERIMENTS)
    if args.coord_mlp:
        experiments += COORD_MLP_EXPERIMENTS

    # Filter by stride if requested
    if args.strides:
        experiments = [e for e in experiments if e[0] in args.strides]

    os.makedirs(args.output_root, exist_ok=True)
    start_time = datetime.now()
    print(f"Starting {len(experiments)} experiments")
    print(f"Output root: {args.output_root}")
    print(f"Epochs: {args.epochs}")
    if args.coord_mlp:
        print("CoordMLP variants: ON")

    results = []
    for label_stride, lambda_vic, use_coord_mlp, name in experiments:
        success, output_dir = run_experiment(
            name, label_stride, lambda_vic, use_coord_mlp,
            args.data_dir, args.output_root,
            args.epochs, extra_args,
        )
        kidney, tumor, mean = read_best_dice(output_dir)
        results.append({
            "name":          name,
            "label_stride":  label_stride,
            "lambda_vic":    lambda_vic,
            "coord_mlp":     use_coord_mlp,
            "annot_pct":     f"{100 // label_stride}%",
            "kidney_dice":   f"{kidney:.4f}" if kidney is not None else "-",
            "tumor_dice":    f"{tumor:.4f}"  if tumor  is not None else "-",
            "mean_dice":     f"{mean:.4f}"   if mean   is not None else "-",
            "success":       success,
        })

    # Print results table
    print(f"\n\n{'='*82}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*82}")
    print(f"{'Name':<34} {'Annot%':>6} {'VIC':>5} {'CoordMLP':>9} {'Kidney':>8} {'Tumor':>8} {'Mean':>8}")
    print(f"{'-'*82}")
    for r in results:
        coord_str = "yes" if r["coord_mlp"] else "no"
        print(
            f"{r['name']:<34} {r['annot_pct']:>6} "
            f"{r['lambda_vic']:>5} {coord_str:>9} "
            f"{r['kidney_dice']:>8} {r['tumor_dice']:>8} {r['mean_dice']:>8}"
        )
    print(f"{'='*82}")

    # Save CSV
    csv_path = os.path.join(args.output_root, "results_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    elapsed = datetime.now() - start_time
    print(f"\nDone in {elapsed}. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
