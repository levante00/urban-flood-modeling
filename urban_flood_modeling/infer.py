from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from .modules import FloodDataModule
from .modules.lightning_module import FloodLightningModule


def run_infer(cfg: DictConfig):
    project_root = Path(cfg.paths.project_root).resolve()
    data_dir = (project_root / cfg.paths.data_dir).resolve()
    checkpoint_path = (project_root / cfg.paths.checkpoint_path).resolve()

    datamodule = FloodDataModule(
        project_root=project_root,
        data_dir=data_dir,
        seq_len=int(cfg.data.seq_len),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        pred_node_type=int(cfg.inference.pred_node_type),
        preprocess_fillna_value=float(cfg.preprocessing.fillna_value),
        preprocess_sort_columns=tuple(cfg.preprocessing.sort_columns),
        dvc_pull_targets=tuple(cfg.dvc.pull_targets),
    )
    model = FloodLightningModule.load_from_checkpoint(str(checkpoint_path))

    accelerator = str(cfg.inference.trainer.accelerator)
    if accelerator == "gpu" and not torch.cuda.is_available():
        accelerator = "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=cfg.inference.trainer.devices,
        logger=bool(cfg.inference.trainer.logger),
        enable_checkpointing=bool(cfg.inference.trainer.enable_checkpointing),
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


@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    run_infer(cfg)


if __name__ == "__main__":
    main()
