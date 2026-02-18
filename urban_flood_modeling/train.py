import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

from .callbacks import LocalMetricsPlotCallback
from .modules import FloodDataModule
from .modules.lightning_module import FloodLightningModule
from .utils.get_git_commit import get_git_commit


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
        data_files={k: str(v) for k, v in cfg.data.files.items()},
    )
    model = FloodLightningModule(
        learning_rate=float(cfg.training.learning_rate),
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        postprocess_cfg=OmegaConf.to_container(cfg.postprocessing, resolve=True),
    )

    mlflow_logger = None
    callbacks: list[pl.Callback] = []

    if bool(cfg.logging.save_local_artifacts):
        callbacks.append(
            LocalMetricsPlotCallback(
                project_root=project_root,
                local_plots_dir=str(cfg.logging.local_plots_dir),
            )
        )

    if bool(cfg.logging.enabled):
        mlflow_logger = MLFlowLogger(
            experiment_name=str(cfg.logging.experiment_name),
            tracking_uri=str(cfg.logging.tracking_uri),
            run_name=str(cfg.logging.run_name),
            log_model=bool(cfg.logging.log_model),
        )

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow_logger.log_hyperparams(cfg_dict)

        git_commit = get_git_commit(project_root)
        mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "git_commit", git_commit)

        mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "stage", "train")

    accelerator = str(cfg.training.trainer.accelerator)
    if accelerator == "gpu" and not torch.cuda.is_available():
        accelerator = "cpu"

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.epochs),
        accelerator=accelerator,
        devices=cfg.training.trainer.devices,
        logger=mlflow_logger if mlflow_logger is not None else bool(cfg.training.trainer.logger),
        callbacks=callbacks,
        enable_checkpointing=bool(cfg.training.trainer.enable_checkpointing),
    )
    trainer.fit(model, datamodule=datamodule)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(checkpoint_path))
    print(f"Training finished. Checkpoint saved to: {checkpoint_path}")

    return checkpoint_path


def compose_train_config(overrides: list[str] | None = None) -> DictConfig:
    config_dir = (Path(__file__).resolve().parents[1] / "configs").resolve()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="train"):
        return compose(config_name="train", overrides=overrides or [])


def main(overrides: list[str] | None = None) -> None:
    cfg = compose_train_config(overrides=overrides)
    run_training(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
