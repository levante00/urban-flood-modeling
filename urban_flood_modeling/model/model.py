import torch
import torch.nn as nn


class FloodModel(nn.Module):
    """LSTM-based sequence model with node-type embedding."""

    def __init__(self) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_size=5, hidden_size=64, batch_first=True)
        self.node_embed = nn.Linear(1, 8)
        self.fc = nn.Sequential(
            nn.Linear(64 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = out[:, -1]
        n = self.node_embed(node_type.unsqueeze(1))
        cat = torch.cat([h, n], dim=1)
        return self.fc(cat).squeeze(-1)
