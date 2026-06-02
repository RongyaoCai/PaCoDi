import numpy as np

from utils.evaluation_common import compute_wasserstein_distance
from utils.visualization import save_distribution_visualizations


def compute_conditional_metrics(real_data, fake_data):
    real_data = np.asarray(real_data, dtype=np.float32)
    fake_data = np.asarray(fake_data, dtype=np.float32)
    sample_count = min(len(real_data), len(fake_data))
    real_data = real_data[:sample_count]
    fake_data = fake_data[:sample_count]

    mse = float(np.mean((fake_data - real_data) ** 2))
    denom = float(np.sum(np.abs(real_data)))
    wape = float(np.sum(np.abs(fake_data - real_data)) / denom) if denom != 0 else 0.0
    return {
        "mse": mse,
        "wape": wape,
        "wasserstein_distance": compute_wasserstein_distance(real_data, fake_data),
        "metric_samples": int(sample_count),
    }


def save_conditional_visualizations(real_data, fake_data, output_dir, compare=3000, seed=123):
    return save_distribution_visualizations(real_data, fake_data, output_dir, compare=compare, seed=seed)
