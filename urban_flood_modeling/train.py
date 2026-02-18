import subprocess
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

from .modules import FloodDataModule
from .modules.lightning_module import FloodLightningModule


def _get_git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_training(cfg: DictConfig) -> Path:
    """Train model and save Lightning checkpoint."""
    project_root = Path(cfg.paths.project_root).resolve()
    data_dir = (project_root / cfg.paths.data_dir).resolve()
    checkpoint_path = (project_root / cfg.paths.checkpoint_path).resolve()

    datamodule = FloodDataModule(
        project_root=project_root,
        data_dir=data_dir,
        seq_len=int(cfg.data.seq_len),
        batch_size=int(cfg.data.batch_size),
        pred_node_type=int(cfg.inference.pred_node_type),
        preprocess_fillna_value=float(cfg.preprocessing.fillna_value),
        preprocess_sort_columns=tuple(cfg.preprocessing.sort_columns),
        dvc_pull_targets=tuple(cfg.dvc.pull_targets),
        num_workers=int(cfg.data.num_workers),
    )
    model = FloodLightningModule(
        learning_rate=float(cfg.training.learning_rate),
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        postprocess_cfg=OmegaConf.to_container(cfg.postprocessing, resolve=True),
    )

    mlflow_logger = None
    if bool(cfg.logging.enabled):
        mlflow_logger = MLFlowLogger(
            experiment_name=str(cfg.logging.experiment_name),
            tracking_uri=str(cfg.logging.tracking_uri),
            run_name=str(cfg.logging.run_name),
            log_model=bool(cfg.logging.log_model),
        )

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow_logger.log_hyperparams(cfg_dict)

        git_commit = _get_git_commit(project_root)
        mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "git_commit", git_commit)

    accelerator = str(cfg.training.trainer.accelerator)
    if accelerator == "gpu" and not torch.cuda.is_available():
        accelerator = "cpu"

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.epochs),
        accelerator=accelerator,
        devices=cfg.training.trainer.devices,
        logger=mlflow_logger if mlflow_logger is not None else bool(cfg.training.trainer.logger),
        enable_checkpointing=bool(cfg.training.trainer.enable_checkpointing),
    )
    trainer.fit(model, datamodule=datamodule)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(checkpoint_path))
    print(f"Training finished. Checkpoint saved to: {checkpoint_path}")
    return checkpoint_path


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
