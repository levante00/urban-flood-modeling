from pathlib import Path

import pytorch_lightning as pl
from dvc.repo import Repo
from torch.utils.data import ConcatDataset, DataLoader

from ..utils.io import build_neighbors, load_data
from ..utils.preprocessing import preprocess_dynamic_df
from .dataset import FloodPredictDataset, FloodTrainDataset


class FloodDataModule(pl.LightningDataModule):
    """Lightning DataModule for flood training data."""

    def __init__(
        self,
        project_root: Path,
        data_dir: Path,
        seq_len: int,
        batch_size: int,
        pred_node_type: int,
        preprocess_fillna_value: float,
        preprocess_sort_columns: tuple[str, str],
        dvc_pull_targets: tuple[str, ...],
        num_workers: int,
        data_files: dict[str, str],
    ) -> None:
        super().__init__()
        self.project_root = project_root
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocess_fillna_value = preprocess_fillna_value
        self.preprocess_sort_columns = preprocess_sort_columns
        self.dvc_pull_targets = dvc_pull_targets
        self.data_files = data_files

        self.neighbors: dict[int, list[int]] | None = None
        self.pred_node_type = pred_node_type

        self._predict_ds: FloodPredictDataset | None = None
        self._train_ds: ConcatDataset | None = None

    def prepare_data(self) -> None:
        """Ensure data files are available locally via DVC before setup."""

        if not (self.project_root / ".dvc").is_dir():
            msg = f"Provided project_root is not a DVC repository: {self.project_root}"
            raise RuntimeError(msg)

        with Repo(str(self.project_root)) as repo:
            targets = list(self.dvc_pull_targets) if self.dvc_pull_targets else None
            repo.pull(targets=targets)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" and self._train_ds is not None:
            return
        if stage == "predict" and self._predict_ds is not None:
            return

        data = load_data(self.data_dir, self.data_files)

        self.neighbors = build_neighbors(data["edges_1d"], data["edges_2d"])

        if stage == "fit":
            train_1d = preprocess_dynamic_df(
                data["train_1d"],
                fillna_value=self.preprocess_fillna_value,
                sort_columns=self.preprocess_sort_columns,
            )
            train_2d = preprocess_dynamic_df(
                data["train_2d"],
                fillna_value=self.preprocess_fillna_value,
                sort_columns=self.preprocess_sort_columns,
            )
            ds1 = FloodTrainDataset(
                train_1d,
                node_type=1,
                neighbors=self.neighbors,
                seq_len=self.seq_len,
            )
            ds2 = FloodTrainDataset(
                train_2d,
                node_type=2,
                neighbors=self.neighbors,
                seq_len=self.seq_len,
            )
            self._train_ds = ConcatDataset([ds1, ds2])
        elif stage == "predict":
            pred_data = data["test_1d"] if self.pred_node_type == 1 else data["test_2d"]
            predict = preprocess_dynamic_df(
                pred_data,
                fillna_value=self.preprocess_fillna_value,
                sort_columns=self.preprocess_sort_columns,
            )

            self._predict_ds = FloodPredictDataset(
                predict,
                node_type=self.pred_node_type,
                neighbors=self.neighbors,
                seq_len=self.seq_len,
            )

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            self.setup(stage="fit")

        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        if self._predict_ds is None:
            self.setup(stage="predict")

        return DataLoader(
            self._predict_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
