import torch
import torch.nn as nn


class FloodModel(nn.Module):
    """LSTM-based sequence model with node-type embedding."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        node_embed_dim: int,
        fc_hidden_sizes: tuple[int, int],
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.node_embed = nn.Linear(1, node_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + node_embed_dim, fc_hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(fc_hidden_sizes[0], fc_hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(fc_hidden_sizes[1], 1),
        )

    def forward(self, input_features: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(input_features)
        last_hidden_state = out[:, -1]
        node_type_embedding = self.node_embed(node_type.unsqueeze(1))
        cat = torch.cat([last_hidden_state, node_type_embedding], dim=1)
        return self.fc(cat).squeeze(-1)
