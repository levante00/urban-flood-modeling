import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..model.model import FloodModel


class FloodLightningModule(pl.LightningModule):
    """LightningModule wrapper for training FloodModel."""

    def __init__(self, learning_rate: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = FloodModel()
        self.loss_fn = nn.MSELoss()

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
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x, t, node_id = batch
        residual = self.model(x, t).view(-1)

        last_val = x[:, -1, 0]
        pred = last_val + residual
        pred = 0.8 * pred + 0.2 * last_val

        pred = torch.where(torch.isfinite(pred), pred, last_val)

        return {"node_id": node_id, "pred": pred}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
