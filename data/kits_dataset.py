import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from typing import List, Tuple, Optional


class KiTS19Dataset(Dataset):
    """
    KiTS19 kidney tumor segmentation dataset with sparse axial supervision.

    For each volume, a binary 'label mask' of shape (D,) marks which axial
    slices are considered labeled.  label_stride=1 → all slices labeled (dense),
    label_stride=5 → every 5th slice labeled (20% annotation budget).

    Volumes are loaded lazily (one per __getitem__ call) and a random 3D patch
    is cropped on-the-fly during training.

    Args:
        data_dir:     path to KiTS19 `data/` folder (contains case_XXXXX/)
        case_ids:     list of integer case IDs to include
        patch_size:   (D, H, W) size of the 3D patch to extract
        label_stride: K — every K-th axial slice is labeled (1 = dense)
        mode:         'train' (random crops, augment) | 'val' | 'test' (center crop)
    """

    def __init__(
        self,
        data_dir: str,
        case_ids: List[int],
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        label_stride: int = 1,
        mode: str = "train",
        crops_per_case: int = 4,
    ):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.label_stride = label_stride
        self.mode = mode
        self.crops_per_case = crops_per_case if mode == "train" else 1

        self.cases = self._find_cases(case_ids)
        if not self.cases:
            raise RuntimeError(
                f"No valid cases found in {data_dir}. "
                "Check that imaging.nii.gz files exist."
            )

    # ------------------------------------------------------------------
    def _find_cases(self, case_ids: List[int]):
        found = []
        for cid in case_ids:
            case_dir = os.path.join(self.data_dir, f"case_{cid:05d}")
            img_p = os.path.join(case_dir, "imaging.nii.gz")
            seg_p = os.path.join(case_dir, "segmentation.nii.gz")
            if os.path.exists(img_p):
                found.append({
                    "img": img_p,
                    "seg": seg_p if os.path.exists(seg_p) else None,
                    "id":  cid,
                })
        return found

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.cases) * self.crops_per_case

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        case = self.cases[idx // self.crops_per_case]

        img = nib.load(case["img"]).get_fdata(dtype=np.float32)
        img = self._preprocess(img)

        if case["seg"] is not None:
            seg = nib.load(case["seg"]).get_fdata().astype(np.int64)
        else:
            seg = np.zeros(img.shape, dtype=np.int64)

        img_patch, seg_patch, d_start = self._crop(img, seg)

        if self.mode == "train":
            img_patch, seg_patch = self._augment(img_patch, seg_patch)

        # Sparse label mask: 1.0 for labeled axial slices, 0.0 otherwise
        D = img_patch.shape[0]
        mask = np.array(
            [(d_start + i) % self.label_stride == 0 for i in range(D)],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(img_patch[None]),   # (1, D, H, W)  float32
            torch.from_numpy(seg_patch),          # (D, H, W)     int64
            torch.from_numpy(mask),               # (D,)          float32
        )

    # ------------------------------------------------------------------
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """HU windowing then [0, 1] normalization."""
        img = np.clip(img, -200, 300)
        img = (img + 200.0) / 500.0          # maps [-200, 300] → [0, 1]
        return img.astype(np.float32)

    def _augment(
        self,
        img: np.ndarray,
        seg: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Random flip along each axis
        for axis in range(3):
            if np.random.random() > 0.5:
                img = np.flip(img, axis=axis).copy()
                seg = np.flip(seg, axis=axis).copy()
        # Mild intensity jitter (scale + shift, then re-clip to [0, 1])
        scale = np.random.uniform(0.9, 1.1)
        shift = np.random.uniform(-0.05, 0.05)
        img = np.clip(img * scale + shift, 0.0, 1.0).astype(np.float32)
        return img, seg

    def _crop(
        self,
        img: np.ndarray,
        seg: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        D, H, W = img.shape
        pd, ph, pw = self.patch_size

        if self.mode == "train":
            d0 = np.random.randint(0, max(1, D - pd + 1))
            h0 = np.random.randint(0, max(1, H - ph + 1))
            w0 = np.random.randint(0, max(1, W - pw + 1))
        else:
            d0 = max(0, (D - pd) // 2)
            h0 = max(0, (H - ph) // 2)
            w0 = max(0, (W - pw) // 2)

        img_p = img[d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw]
        seg_p = seg[d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw]

        # Pad if the patch extends past the volume boundary
        img_p = _pad3d(img_p, (pd, ph, pw), pad_val=0.0)
        seg_p = _pad3d(seg_p, (pd, ph, pw), pad_val=0)

        return img_p, seg_p, d0


# ──────────────────────────────────────────────────────────────────────────────

def _pad3d(arr: np.ndarray, target: Tuple[int, int, int], pad_val=0) -> np.ndarray:
    """Zero-pad arr to at least `target` shape."""
    pads = [(0, max(0, t - s)) for s, t in zip(arr.shape, target)]
    if any(p[1] > 0 for p in pads):
        arr = np.pad(arr, pads, mode="constant", constant_values=pad_val)
    return arr
