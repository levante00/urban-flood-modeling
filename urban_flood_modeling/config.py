from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(slots=True)
class Settings:
    """Runtime configuration for the urban flood pipeline."""

    project_root: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_path: Path = field(default_factory=lambda: Path("submission.csv"))
    checkpoint_path: Path = field(default_factory=lambda: Path("artifacts/flood_model.ckpt"))
    seq_len: int = 10
    batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 1e-3
    num_workers: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
