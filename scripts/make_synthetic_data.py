"""
scripts/make_synthetic_data.py

Generate synthetic KiTS19-style NIfTI volumes for local testing.
Useful when you don't have the real dataset yet, or for CI smoke tests.

Usage:
  python scripts/make_synthetic_data.py                  # 10 cases
  python scripts/make_synthetic_data.py --n_cases 5 --data_dir data/kits19/data
"""

import argparse
import os
import numpy as np
import nibabel as nib


def make_case(case_dir: str, seed: int):
    rng = np.random.default_rng(seed)

    D, H, W = 96, 192, 192

    # CT-like image: background HU noise + bright kidney blob
    img = rng.normal(-50, 80, (D, H, W)).astype(np.float32)

    # Kidney region (class 1)
    seg = np.zeros((D, H, W), dtype=np.uint8)
    d0, h0, w0 = D // 4, H // 4, W // 4
    seg[d0 : d0 + D // 2, h0 : h0 + H // 2, w0 : w0 + W // 2] = 1
    img[seg == 1] += rng.normal(150, 30, (seg == 1).sum()).astype(np.float32)

    # Tumor region (class 2) — smaller blob inside kidney
    t0 = (d0 + D // 6, h0 + H // 6, w0 + W // 6)
    seg[t0[0] : t0[0] + D // 6, t0[1] : t0[1] + H // 6, t0[2] : t0[2] + W // 6] = 2
    img[seg == 2] += rng.normal(80, 20, (seg == 2).sum()).astype(np.float32)

    affine = np.diag([0.8, 0.8, 3.0, 1.0])  # realistic CT spacing

    os.makedirs(case_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(img,                    affine), f"{case_dir}/imaging.nii.gz")
    nib.save(nib.Nifti1Image(seg.astype(np.float32), affine), f"{case_dir}/segmentation.nii.gz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/kits19/data")
    parser.add_argument("--n_cases",  type=int, default=10)
    args = parser.parse_args()

    print(f"Generating {args.n_cases} synthetic cases in {args.data_dir} ...")
    for i in range(args.n_cases):
        case_dir = os.path.join(args.data_dir, f"case_{i:05d}")
        make_case(case_dir, seed=i)
        print(f"  case_{i:05d} done")

    print(f"\nDone. Run prepare_splits.py next:")
    print(f"  python scripts/prepare_splits.py --data_dir {args.data_dir}")


if __name__ == "__main__":
    main()
