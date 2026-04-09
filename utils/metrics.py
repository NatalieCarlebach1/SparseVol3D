"""
Dice coefficient utilities for 3D segmentation evaluation.
"""

import numpy as np
import torch
from typing import Dict


def compute_dice(
    pred:        torch.Tensor,
    target:      torch.Tensor,
    num_classes: int = 3,
    smooth:      float = 1e-5,
) -> Dict[int, float]:
    """
    Per-class Dice coefficient on torch tensors.

    Args:
        pred:   (D, H, W)  predicted class labels (argmax of logits)
        target: (D, H, W)  ground-truth labels
        num_classes: number of classes (including background)

    Returns:
        dict {class_id: dice_score}
    """
    dice = {}
    for c in range(num_classes):
        p = (pred   == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        union        = p.sum() + t.sum()
        if union < smooth:
            dice[c] = 1.0          # both empty — perfect score
        else:
            dice[c] = float((2.0 * intersection + smooth) / (union + smooth))
    return dice


def compute_dice_volume(
    pred:        np.ndarray,
    target:      np.ndarray,
    num_classes: int = 3,
    smooth:      float = 1e-5,
) -> Dict[int, float]:
    """
    Per-class Dice on numpy arrays (full-volume evaluation).

    Args:
        pred:   (D, H, W)  predicted labels
        target: (D, H, W)  ground-truth labels
    """
    dice = {}
    for c in range(num_classes):
        p = (pred   == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        intersection = float((p * t).sum())
        union        = float(p.sum() + t.sum())
        if union < smooth:
            dice[c] = 1.0
        else:
            dice[c] = (2.0 * intersection + smooth) / (union + smooth)
    return dice
