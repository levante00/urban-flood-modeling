from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch


class LocalMetricsPlotCallback(pl.Callback):
    """Save train metrics and plots locally at train end."""

    def __init__(self, project_root: Path, local_plots_dir: str) -> None:
        super().__init__()
        self.project_root = project_root
        self.local_plots_dir = local_plots_dir
        self.history: list[dict[str, float]] = []

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:  # workround to avoid unused argument warnings
        del pl_module
        metrics = trainer.callback_metrics
        row: dict[str, float] = {"epoch": float(trainer.current_epoch)}

        for metric_name in ["train_loss", "train_rmse", "train_mae"]:
            value = metrics.get(metric_name)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                row[metric_name] = float(value.detach().cpu())
            else:
                row[metric_name] = float(value)

        self.history.append(row)

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module  # workround to avoid unused argument warnings
        if not self.history:
            return

        mlflow_logger = trainer.loggers[0] if trainer.loggers else None
        run_id = mlflow_logger.run_id if mlflow_logger is not None else "local_run"

        plot_dir = (self.project_root / self.local_plots_dir / "train" / run_id).resolve()
        plot_dir.mkdir(parents=True, exist_ok=True)

        metrics_df = pd.DataFrame(self.history).drop_duplicates(subset=["epoch"])
        metrics_csv = plot_dir / "metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)

        for metric in ["train_loss", "train_rmse", "train_mae"]:
            if metric not in metrics_df.columns:
                continue
            subset = metrics_df[["epoch", metric]].dropna()
            if subset.empty:
                continue

            output_path = plot_dir / f"{metric}.png"
            plt.figure(figsize=(7, 4))
            plt.plot(subset["epoch"], subset[metric])
            plt.title(metric)
            plt.xlabel("epoch")
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
