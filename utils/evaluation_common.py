from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance


def paired_subset(real_data, fake_data, max_samples=None, seed=123):
    real_data = np.asarray(real_data, dtype=np.float32)
    fake_data = np.asarray(fake_data, dtype=np.float32)
    sample_count = min(len(real_data), len(fake_data))
    if max_samples is not None:
        sample_count = min(sample_count, int(max_samples))

    rng = np.random.default_rng(seed)
    real_idx = rng.permutation(len(real_data))[:sample_count]
    fake_idx = rng.permutation(len(fake_data))[:sample_count]
    return real_data[real_idx], fake_data[fake_idx]


def compute_wasserstein_distance(real_data, fake_data):
    real_data = np.asarray(real_data, dtype=np.float32)
    fake_data = np.asarray(fake_data, dtype=np.float32)
    if real_data.ndim == 2:
        real_data = real_data[..., np.newaxis]
    if fake_data.ndim == 2:
        fake_data = fake_data[..., np.newaxis]
    if real_data.ndim != 3 or fake_data.ndim != 3:
        raise ValueError(
            f"Expected time-series data with shape (N, L, C), got {real_data.shape} and {fake_data.shape}."
        )

    channels = min(real_data.shape[-1], fake_data.shape[-1])
    real_flat = real_data[..., :channels].reshape(-1, channels)
    fake_flat = fake_data[..., :channels].reshape(-1, channels)
    distances = [
        scipy_wasserstein_distance(real_flat[:, channel], fake_flat[:, channel])
        for channel in range(channels)
    ]
    return float(np.mean(distances))


def append_metrics_csv(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        output = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        output = new_row
    output.to_csv(csv_path, index=False, float_format="%.5f")
    return str(csv_path)


def update_total_metrics_csv(csv_path, row, key_columns=("experiment", "milestone")):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])

    if csv_path.exists():
        output = pd.read_csv(csv_path)
        for column in new_row.columns:
            if column not in output.columns:
                output[column] = np.nan
        for column in output.columns:
            if column not in new_row.columns:
                new_row[column] = np.nan

        output = output[list(new_row.columns)]
        mask = pd.Series(True, index=output.index)
        for column in key_columns:
            if column in output.columns and column in new_row.columns:
                mask &= output[column].astype(str) == str(new_row.iloc[0][column])
        output = output.loc[~mask]
        output = pd.concat([output, new_row], ignore_index=True, sort=False)
    else:
        output = new_row

    sort_columns = [
        column
        for column in ("seq_length", "backbone", "real_imag_interaction", "experiment")
        if column in output.columns
    ]
    if sort_columns:
        output = output.sort_values(sort_columns).reset_index(drop=True)
    output.to_csv(csv_path, index=False, float_format="%.5f")
    return str(csv_path)
