import ast
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_EPS = 1e-6


class ConditionalDataset(Dataset):
    """TSFragment-600K-style conditional time-series dataset.

    Each CSV row is one (text, time-series) pair: column ``OT`` holds the
    series as a stringified list, column ``TextEmbedding`` holds the
    pre-computed text embedding used as the conditioning vector. The series
    length equals the file's window size.

    Train / test split: a single random permutation (seeded) over all rows.
    The first ``split_ratio`` fraction becomes train, the remainder test.
    Both periods consume the *same* permutation, so the two splits are
    disjoint regardless of which is instantiated first.

    Normalization: per-channel z-score statistics are computed on the train
    split only and reused to normalize whichever period this instance
    represents. ``unnormalize`` inverts the same transform.
    """

    def __init__(
        self,
        name,
        data_root,
        window,
        split_ratio=0.9,
        seed=123,
        period="train",
        **kwargs,
    ):
        super().__init__()
        if period not in ("train", "test"):
            raise ValueError(f"period must be 'train' or 'test', got {period!r}")
        if not 0.0 < split_ratio < 1.0:
            raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")

        self.name = name
        self.window = int(window)
        self.period = period

        csv_path = os.path.join(data_root, f"{name}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Conditional dataset CSV not found at {csv_path}. "
                "PaCoDi's conditional generation task uses the TSFragment-600K dataset "
                "(https://huggingface.co/datasets/WinfredGe/TSFragment-600K). "
                "Download the relevant CSV files and place them under ./data/TSFragment-600K/."
            )

        series, condition = self._load_csv(csv_path)
        if series.shape[1] != self.window:
            raise ValueError(
                f"Window mismatch for {csv_path}: CSV provides length "
                f"{series.shape[1]}, but config requested window={self.window}."
            )

        train_idx, test_idx = self._make_split(len(series), split_ratio, seed)

        self.mean = series[train_idx].mean(axis=(0, 1), keepdims=False).astype(np.float32)
        self.std = (series[train_idx].std(axis=(0, 1), keepdims=False) + _EPS).astype(np.float32)

        selected = train_idx if period == "train" else test_idx
        self.samples = ((series[selected] - self.mean) / self.std).astype(np.float32)
        self.condition = condition[selected].astype(np.float32)

        self.var_num = self.samples.shape[-1]
        self.sample_num = self.samples.shape[0]

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        x = torch.from_numpy(self.samples[index]).float()
        cond = torch.from_numpy(self.condition[index]).float()
        return x, cond

    def unnormalize(self, data):
        """Invert the z-score applied at construction time.

        Accepts NumPy arrays or PyTorch tensors with a trailing channel axis
        matching ``self.var_num``. Returns a NumPy array in physical units.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return data.astype(np.float32) * self.std + self.mean

    @staticmethod
    def _make_split(num_rows, ratio, seed):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(num_rows)
        cutoff = int(np.floor(num_rows * ratio))
        if cutoff == 0 or cutoff == num_rows:
            raise ValueError(
                f"split_ratio={ratio} on {num_rows} rows produces an empty split"
            )
        return indices[:cutoff], indices[cutoff:]

    @staticmethod
    def _load_csv(csv_path):
        frame = pd.read_csv(csv_path)
        for column in ("OT", "TextEmbedding"):
            if column not in frame.columns:
                raise ValueError(f"{csv_path} missing required column '{column}'.")

        series = np.stack(
            [np.asarray(ast.literal_eval(row), dtype=np.float32) for row in frame["OT"]],
            axis=0,
        )
        # (N, window) -> (N, window, channels=1)
        series = series[..., np.newaxis]

        condition = np.stack(
            [ConditionalDataset._parse_embedding(row) for row in frame["TextEmbedding"]],
            axis=0,
        ).astype(np.float32)

        return series, condition

    @staticmethod
    def _parse_embedding(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return np.asarray(value, dtype=np.float32)
        text = str(value).strip()
        try:
            return np.asarray(ast.literal_eval(text), dtype=np.float32)
        except (ValueError, SyntaxError):
            cleaned = text.replace("[", " ").replace("]", " ").replace(",", " ")
            return np.fromstring(cleaned, sep=" ", dtype=np.float32)
