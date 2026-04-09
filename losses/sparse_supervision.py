"""
Losses for sparse axial supervision.

Two components:
1. sparse_supervised_loss — CE + Dice computed on labeled slices only.
2. volumetric_consistency_loss — VIC loss: predictions at unlabeled slices
   should lie on the linear interpolation between neighboring labeled slices.

The combined_loss function wraps both.
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _soft_dice(probs: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """
    Multi-class soft Dice loss, averaged over foreground classes.

    probs:  (N, C, H, W)  softmax probabilities
    target: (N, H, W)     integer class labels
    """
    num_classes = probs.shape[1]
    loss = torch.tensor(0.0, device=probs.device)
    for c in range(1, num_classes):           # skip background class 0
        p = probs[:, c]                        # (N, H, W)
        t = (target == c).float()
        inter = (p * t).sum()
        loss = loss + 1.0 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)
    return loss / (num_classes - 1)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Sparse supervised loss
# ──────────────────────────────────────────────────────────────────────────────

def sparse_supervised_loss(
    logits: torch.Tensor,
    seg:    torch.Tensor,
    mask:   torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy + soft Dice loss, computed on labeled axial slices only.

    Args:
        logits: (B, C, D, H, W)
        seg:    (B, D, H, W)   integer ground-truth labels
        mask:   (B, D)         float mask, 1.0 = labeled slice

    Returns:
        Scalar loss.  Returns 0 (with grad) if no labeled slices in batch.
    """
    B, C, D, H, W = logits.shape

    labeled_logits = []
    labeled_seg    = []

    for b in range(B):
        idx = mask[b].bool()                                  # (D,)
        if idx.sum() == 0:
            continue
        # logits[b]: (C, D, H, W) → select along D → (C, K, H, W) → (K, C, H, W)
        labeled_logits.append(logits[b, :, idx].permute(1, 0, 2, 3))
        labeled_seg.append(seg[b, idx])                       # (K, H, W)

    if not labeled_logits:
        return logits.sum() * 0.0                             # zero with gradient

    ll = torch.cat(labeled_logits, dim=0)   # (N_labeled, C, H, W)
    ls = torch.cat(labeled_seg,    dim=0)   # (N_labeled, H, W)

    ce    = F.cross_entropy(ll, ls)
    probs = F.softmax(ll, dim=1)
    dice  = _soft_dice(probs, ls)

    return ce + dice


# ──────────────────────────────────────────────────────────────────────────────
# 2. Volumetric Interpolation Consistency (VIC) loss
# ──────────────────────────────────────────────────────────────────────────────

def volumetric_consistency_loss(
    logits: torch.Tensor,
    mask:   torch.Tensor,
) -> torch.Tensor:
    """
    Volumetric Interpolation Consistency (VIC) loss.

    For each unlabeled slice at depth z (bounded by labeled slices z0 < z < z1):

        target(z) = (1 - α) · p(z0) + α · p(z1),   α = (z - z0) / (z1 - z0)

    where p(·) = softmax(logits[:, :, ·, :, :]).

    The target is detached — we regularize the unlabeled slice predictions,
    not the labeled anchors.

    Motivation (from 3DGS): a good volumetric representation should be a
    smooth, continuous field between observations, not just accurate at
    sparse annotation planes.

    Args:
        logits: (B, C, D, H, W)
        mask:   (B, D)  float mask, 1.0 = labeled

    Returns:
        Scalar VIC loss averaged over all unlabeled slices in the batch.

    Note:
        The inner loop runs over pairs of consecutive labeled slices.
        For label_stride=K it iterates O(D/K) pairs per sample.
        Vectorizable if needed, but clear as written for a prototype.
    """
    B, C, D, H, W = logits.shape
    probs = F.softmax(logits, dim=1)          # (B, C, D, H, W)

    total = torch.tensor(0.0, device=logits.device)
    count = 0

    for b in range(B):
        labeled_idx = mask[b].nonzero(as_tuple=False).view(-1)  # sorted 1-D
        if labeled_idx.numel() < 2:
            continue

        for k in range(len(labeled_idx) - 1):
            z0  = labeled_idx[k].item()
            z1  = labeled_idx[k + 1].item()
            gap = z1 - z0

            if gap <= 1:
                continue                       # no unlabeled slices in between

            p0 = probs[b, :, z0]              # (C, H, W) — anchor at z0
            p1 = probs[b, :, z1]              # (C, H, W) — anchor at z1

            for z in range(z0 + 1, z1):
                alpha  = (z - z0) / gap
                target = ((1.0 - alpha) * p0 + alpha * p1).detach()
                total  = total + F.mse_loss(probs[b, :, z], target)
                count += 1

    return total / max(count, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Combined loss
# ──────────────────────────────────────────────────────────────────────────────

def combined_loss(
    logits:     torch.Tensor,
    seg:        torch.Tensor,
    mask:       torch.Tensor,
    lambda_vic: float = 0.1,
) -> torch.Tensor:
    """
    L_total = L_sup + lambda_vic * L_VIC

    Args:
        logits:     (B, C, D, H, W)
        seg:        (B, D, H, W)
        mask:       (B, D)  1.0 = labeled slice
        lambda_vic: weight for VIC loss (0 = supervised-only baseline)
    """
    sup = sparse_supervised_loss(logits, seg, mask)
    vic = volumetric_consistency_loss(logits, mask) if lambda_vic > 0 else 0.0
    return sup + lambda_vic * vic
