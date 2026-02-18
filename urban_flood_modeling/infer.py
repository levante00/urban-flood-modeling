import pytorch_lightning as pl
import torch

from .config import Settings
from .modules import FloodDataModule
from .modules.lightning_module import FloodLightningModule


def run_predict(settings: Settings, pred_node_type: int):
    datamodule = FloodDataModule(
        project_root=settings.project_root,
        data_dir=settings.data_dir,
        seq_len=settings.seq_len,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pred_node_type=pred_node_type,
    )
    model = FloodLightningModule.load_from_checkpoint(
        settings.checkpoint_path, learning_rate=settings.learning_rate
    )

    accelerator = "gpu" if settings.device == "cuda" and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=settings.epochs,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    batches = trainer.predict(model, datamodule=datamodule)

    out = {}
    for b in batches:
        ids = b["node_id"].detach().cpu().tolist()
        preds = b["pred"].detach().cpu().tolist()
        for nid, p in zip(ids, preds, strict=False):
            out[str(int(nid))] = float(p)

    print(f"Prediction finished. Batches: {len(batches)}")
    return out


def main() -> None:
    settings = Settings()
    run_predict(settings, pred_node_type=1)


if __name__ == "__main__":
    main()
