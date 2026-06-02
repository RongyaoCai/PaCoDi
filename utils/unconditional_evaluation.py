import numpy as np
import torch

from utils.context_fid import context_fid
from utils.cross_correlation import CrossCorrelationLoss
from utils.discriminative_metric import discriminative_score_metrics
from utils.evaluation_common import compute_wasserstein_distance, paired_subset
from utils.predictive_metric import predictive_score_metrics
from utils.visualization import save_distribution_visualizations


def compute_unconditional_metrics(
    real_data,
    fake_data,
    max_samples=3000,
    discriminative_iterations=2000,
    predictive_iterations=5000,
    seed=123,
    device=None,
):
    real_data, fake_data = paired_subset(real_data, fake_data, max_samples=max_samples, seed=seed)

    x_real = torch.from_numpy(real_data).float()
    x_fake = torch.from_numpy(fake_data).float()
    idx_r = np.random.default_rng(seed + 1).permutation(len(x_real))[: min(1000, len(x_real))]
    idx_f = np.random.default_rng(seed + 2).permutation(len(x_fake))[: min(1000, len(x_fake))]

    corr = CrossCorrelationLoss(x_real[idx_r], name="CrossCorrelationLoss")
    cross_corr = corr.compute(x_fake[idx_f]).item()

    discriminative_score, fake_accuracy, real_accuracy = discriminative_score_metrics(
        real_data,
        fake_data,
        iterations=discriminative_iterations,
        device=device,
    )
    predictive_score = predictive_score_metrics(
        real_data,
        fake_data,
        iterations=predictive_iterations,
        device=device,
    )

    return {
        "context_fid": float(context_fid(real_data, fake_data)),
        "cross_correlation": float(cross_corr),
        "wasserstein_distance": compute_wasserstein_distance(real_data, fake_data),
        "discriminative_score": float(discriminative_score),
        "discriminative_fake_accuracy": float(fake_accuracy),
        "discriminative_real_accuracy": float(real_accuracy),
        "predictive_score": float(predictive_score),
        "metric_samples": int(len(real_data)),
    }


def save_unconditional_visualizations(real_data, fake_data, output_dir, compare=3000, seed=123):
    return save_distribution_visualizations(real_data, fake_data, output_dir, compare=compare, seed=seed)
