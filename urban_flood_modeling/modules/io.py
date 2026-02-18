from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all required CSV files for training and inference."""
    return {
        "train_1d": pd.read_csv(data_dir / "1d_nodes_dynamic_all.csv"),
        "train_2d": pd.read_csv(data_dir / "2d_nodes_dynamic_all.csv"),
        "test_1d": pd.read_csv(data_dir / "test_1d_nodes_dynamic_all.csv"),
        "test_2d": pd.read_csv(data_dir / "test_2d_nodes_dynamic_all.csv"),
        "sample": pd.read_csv(data_dir / "sample_submission.csv"),
        "edges_1d": pd.read_csv(data_dir / "1d_edge_index.csv"),
        "edges_2d": pd.read_csv(data_dir / "2d_edge_index.csv"),
    }


def build_neighbors(edges_1d: pd.DataFrame, edges_2d: pd.DataFrame) -> dict[int, list[int]]:
    """Build an undirected node neighborhood map from edge lists."""
    neighbors: dict[int, list[int]] = defaultdict(list)
    for df in (edges_1d, edges_2d):
        a, b = df.columns[:2]
        for u, v in df[[a, b]].values:
            u_i, v_i = int(u), int(v)
            neighbors[u_i].append(v_i)
            neighbors[v_i].append(u_i)
    return dict(neighbors)
