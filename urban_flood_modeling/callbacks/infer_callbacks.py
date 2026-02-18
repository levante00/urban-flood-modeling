from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch


class LocalInferenceMetricsCallback(pl.Callback):
    """Save final inference metrics locally at predict end."""

    def __init__(self, project_root: Path, local_plots_dir: str) -> None:
        super().__init__()
        self.project_root = project_root
        self.local_plots_dir = local_plots_dir
        self.pred_values: list[float] = []

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del (
            trainer,
            pl_module,
            batch,
            batch_idx,
            dataloader_idx,
        )  # workround to avoid unused argument warnings
        if outputs is None:
            return

        preds = outputs.get("pred")
        if preds is None:
            return

        pred_batch = preds.detach().float().cpu().view(-1).tolist()
        self.pred_values.extend(float(v) for v in pred_batch)

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module  # workround to avoid unused argument warnings
        if not trainer.is_global_zero:
            return
        if not self.pred_values:
            return

        pred_tensor = torch.tensor(self.pred_values, dtype=torch.float32)
        final_metrics = {
            "pred_count": float(pred_tensor.numel()),
            "pred_mean": float(pred_tensor.mean()),
            "pred_std": float(pred_tensor.std(unbiased=False)),
            "pred_min": float(pred_tensor.min()),
            "pred_max": float(pred_tensor.max()),
        }

        mlflow_logger = trainer.loggers[0] if trainer.loggers else None
        run_id = mlflow_logger.run_id if mlflow_logger is not None else "local_run"

        out_dir = (self.project_root / self.local_plots_dir / "infer" / run_id).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "final_metrics.csv"
        pd.DataFrame([final_metrics]).to_csv(metrics_path, index=False)

        if mlflow_logger is not None:
            for key, value in final_metrics.items():
                mlflow_logger.experiment.log_metric(mlflow_logger.run_id, key, value)
