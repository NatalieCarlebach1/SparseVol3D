from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str = "data/kits19/data"
    output_dir: str = "outputs"

    # Default splits (case indices 0–209 in KiTS19)
    train_cases: List[int] = field(default_factory=lambda: list(range(160)))
    val_cases:   List[int] = field(default_factory=lambda: list(range(160, 185)))
    test_cases:  List[int] = field(default_factory=lambda: list(range(185, 210)))

    # ── Patch sampling ────────────────────────────────────────────────────────
    patch_size: Tuple[int, int, int] = (64, 128, 128)   # (D, H, W)
    crops_per_case: int = 4                              # random crops per epoch per case

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes: int = 3          # 0=background, 1=kidney, 2=tumor
    base_channels: int = 32       # channels at first encoder level

    # ── Sparse supervision ────────────────────────────────────────────────────
    label_stride: int = 1         # 1=dense, N=every N-th axial slice labeled
    lambda_vic: float = 0.1       # weight for VIC loss (0 = disable)

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True              # mixed precision (recommended)

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 4
    log_interval: int = 10        # validate every N epochs
    save_interval: int = 10       # checkpoint every N epochs
    device: str = "cuda"
