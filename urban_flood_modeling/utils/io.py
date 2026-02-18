from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_data(data_dir: Path, data_files: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Load required CSV files for training and inference from configured filenames."""
    return {key: pd.read_csv(data_dir / file_name) for key, file_name in data_files.items()}


def build_neighbors(edges_1d: pd.DataFrame, edges_2d: pd.DataFrame) -> dict[int, list[int]]:
    """Build an undirected node neighborhood map from edge lists."""
    neighbors: dict[int, list[int]] = defaultdict(list)
    for df in (edges_1d, edges_2d):
        source_col, target_col = df.columns[:2]
        for source_node_id, target_node_id in df[[source_col, target_col]].values:
            source_node_id_int, target_node_id_int = int(source_node_id), int(target_node_id)
            neighbors[source_node_id_int].append(target_node_id_int)
            neighbors[target_node_id_int].append(source_node_id_int)

    return dict(neighbors)
