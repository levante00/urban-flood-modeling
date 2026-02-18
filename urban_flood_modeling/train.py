from pathlib import Path

import pytorch_lightning as pl
import torch

from .config import Settings
from .modules import FloodDataModule
from .modules.lightning_module import FloodLightningModule


def run_training(settings: Settings) -> Path:
    """Train model and save Lightning checkpoint."""
    datamodule = FloodDataModule(
        data_dir=settings.data_dir,
        seq_len=settings.seq_len,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )
    model = FloodLightningModule(learning_rate=settings.learning_rate)

    accelerator = "gpu" if settings.device == "cuda" and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=settings.epochs,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=datamodule)

    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(settings.checkpoint_path))
    print(f"Training finished. Checkpoint saved to: {settings.checkpoint_path}")
    return settings.checkpoint_path


def main() -> None:
    settings = Settings()
    run_training(settings)


if __name__ == "__main__":
    main()
