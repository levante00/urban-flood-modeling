import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FloodTrainDataset(Dataset):
    """Sequence dataset for node-level flood forecasting."""

    def __init__(
        self,
        df: pd.DataFrame,
        node_type: int,
        neighbors: dict[int, list[int]],
        seq_len: int,
    ) -> None:
        self.samples: list[tuple[np.ndarray, np.float32, np.float32]] = []

        node_series: dict[int, np.ndarray] = {}
        for nid, group in df.groupby("node_id"):
            group = group.sort_values("timestep")
            node_series[int(nid)] = group["water_level"].to_numpy()

        for node_id, group in df.groupby("node_id"):
            node_id_int = int(node_id)
            group = group.sort_values("timestep")

            vals = group["water_level"].to_numpy()
            rain = (
                group["rainfall"].to_numpy()
                if "rainfall" in group.columns
                else np.zeros(len(group), dtype=np.float32)
            )
            neigh_ids = neighbors.get(node_id_int, [])

            for i in range(len(group) - seq_len):
                seq = vals[i : i + seq_len]
                rain_seq = rain[i : i + seq_len]

                mean = seq.mean()
                std = seq.std() if seq.std() > 0 else 1.0

                neigh_vals: list[float] = []
                for t in range(seq_len):
                    idx = i + t
                    vals_list = [
                        node_series[nb][idx]
                        for nb in neigh_ids
                        if nb in node_series and idx < len(node_series[nb])
                    ]
                    neigh_vals.append(float(np.mean(vals_list)) if vals_list else float(seq[t]))

                x = np.stack(
                    [
                        seq,
                        rain_seq,
                        np.full(seq_len, mean),
                        np.full(seq_len, std),
                        np.array(neigh_vals),
                    ],
                    axis=1,
                ).astype(np.float32)
                y = np.float32(vals[i + seq_len])

                self.samples.append((x, y, np.float32(node_type)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, t = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(t, dtype=torch.float32),
        )


class FloodPredictDataset(Dataset):
    """
    One item per node_id, features built exactly like predict_df.
    Assumes each node has at least SEQ_LEN timesteps (like predict_df).
    """

    def __init__(
        self, df: pd.DataFrame, node_type: int, neighbors: dict[int, list[int]], seq_len: int
    ):
        self.df = df
        self.node_type = np.float32(node_type)
        self.neighbors = neighbors
        self.seq_len = seq_len

        self.node_series = {}
        for nid, group in df.groupby("node_id"):
            group = group.sort_values("timestep")
            self.node_series[int(nid)] = group["water_level"].values

        self.node_ids = []
        self.groups = {}
        for nid, group in df.groupby("node_id"):
            nid = int(nid)
            self.node_ids.append(nid)
            self.groups[nid] = group.sort_values("timestep")

        self.has_rain = "rainfall" in df.columns

    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, idx):
        nid = self.node_ids[idx]
        group = self.groups[nid]

        vals = group["water_level"].values[-self.seq_len :]
        rain = (
            group["rainfall"].to_numpy()[-self.seq_len :]
            if self.has_rain
            else np.zeros(self.seq_len, dtype=np.float32)
        )
        mean = vals.mean()
        std = vals.std() if vals.std() > 0 else 1.0

        neigh_ids = self.neighbors.get(nid, [])
        neigh_vals = []
        for t in range(self.seq_len):
            vals_list = []
            for nb in neigh_ids:
                if nb in self.node_series and t < len(self.node_series[nb]):
                    vals_list.append(self.node_series[nb][-self.seq_len + t])
            neigh_vals.append(np.mean(vals_list) if len(vals_list) > 0 else vals[t])

        x = np.stack(
            [
                vals,
                rain,
                np.full(self.seq_len, mean),
                np.full(self.seq_len, std),
                np.array(neigh_vals),
            ],
            axis=1,
        ).astype(np.float32)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.node_type, dtype=torch.float32),
            torch.tensor(nid, dtype=torch.long),
        )
