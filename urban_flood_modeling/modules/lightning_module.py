import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..model.model import FloodModel


class FloodLightningModule(pl.LightningModule):
    """LightningModule wrapper for training FloodModel."""

    def __init__(
        self,
        learning_rate: float,
        model_cfg: dict[str, int | list[int]],
        postprocess_cfg: dict[str, float],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        fc_hidden_sizes = tuple(int(v) for v in model_cfg["fc_hidden_sizes"])
        self.model = FloodModel(
            input_size=int(model_cfg["input_size"]),
            hidden_size=int(model_cfg["hidden_size"]),
            node_embed_dim=int(model_cfg["node_embed_dim"]),
            fc_hidden_sizes=(fc_hidden_sizes[0], fc_hidden_sizes[1]),
        )
        self.loss_fn = nn.MSELoss()
        self.smoothing_alpha = float(postprocess_cfg["smoothing_alpha"])
        self.smoothing_beta = float(postprocess_cfg["smoothing_beta"])

    def forward(self, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        return self.model(x, node_type)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y, t = batch
        pred = self.model(x, t)
        loss = self.loss_fn(pred, y)
        rmse = torch.sqrt(loss)
        mae = torch.mean(torch.abs(pred - y))

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_mae", mae, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x, t, node_id = batch
        residual = self.model(x, t).view(-1)

        last_val = x[:, -1, 0]
        pred = last_val + residual
        pred = self.smoothing_alpha * pred + self.smoothing_beta * last_val

        pred = torch.where(torch.isfinite(pred), pred, last_val)

        return {"node_id": node_id, "pred": pred}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
