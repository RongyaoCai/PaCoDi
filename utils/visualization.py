import inspect
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


PNG_DPI = 600
SINGLE_FIGSIZE = (7.2, 6.0)
SAMPLE_PANEL_FIGSIZE = (5.8, 4.0)


def _as_time_series(data):
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[..., np.newaxis]
    if data.ndim != 3:
        raise ValueError(f"Expected time-series data with shape (N, L, C), got {data.shape}")
    return data.astype(np.float32)


def _paired_subset(real_data, fake_data, compare, seed):
    real_data = _as_time_series(real_data)
    fake_data = _as_time_series(fake_data)
    sample_count = min(compare, real_data.shape[0], fake_data.shape[0])
    if sample_count <= 1:
        raise ValueError("Need at least two samples for distribution visualization.")

    rng = np.random.default_rng(seed)
    real_idx = rng.permutation(real_data.shape[0])[:sample_count]
    fake_idx = rng.permutation(fake_data.shape[0])[:sample_count]
    return real_data[real_idx], fake_data[fake_idx]


def _sequence_features(real_data, fake_data, compare=3000, seed=123):
    real_data, fake_data = _paired_subset(real_data, fake_data, compare, seed)
    return real_data.mean(axis=2), fake_data.mean(axis=2)


def _save_figure(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def _scatter(real_points, fake_points, title, xlabel, ylabel, output_path):
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.scatter(real_points[:, 0], real_points[:, 1], c="#d62728", alpha=0.28, s=14, label="Original")
    ax.scatter(fake_points[:, 0], fake_points[:, 1], c="#1f77b4", alpha=0.28, s=14, label="Synthetic")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(alpha=0.18, linewidth=0.6)
    return _save_figure(fig, output_path)


def _kde_curve(values, x_grid):
    if np.std(values) == 0:
        values = values + np.random.default_rng(0).normal(0, 1e-6, size=values.shape)
    return gaussian_kde(values)(x_grid)


def save_pca_plot(real_data, fake_data, output_dir, compare=3000, seed=123):
    real_features, fake_features = _sequence_features(real_data, fake_data, compare, seed)
    pca = PCA(n_components=2)
    pca.fit(real_features)
    real_points = pca.transform(real_features)
    fake_points = pca.transform(fake_features)
    return _scatter(
        real_points,
        fake_points,
        title="PCA plot",
        xlabel="x-pca",
        ylabel="y-pca",
        output_path=Path(output_dir) / "figure" / "vis_pca.png",
    )


def save_tsne_plot(real_data, fake_data, output_dir, compare=3000, seed=123):
    real_features, fake_features = _sequence_features(real_data, fake_data, compare, seed)
    features = np.concatenate([real_features, fake_features], axis=0)
    perplexity = min(40, max(2, (features.shape[0] - 1) // 3))

    tsne_kwargs = {
        "n_components": 2,
        "verbose": 0,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": seed,
    }
    iteration_arg = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    tsne_kwargs[iteration_arg] = 300
    tsne = TSNE(**tsne_kwargs)

    points = tsne.fit_transform(features)
    sample_count = real_features.shape[0]
    return _scatter(
        points[:sample_count],
        points[sample_count:],
        title="t-SNE plot",
        xlabel="x-tsne",
        ylabel="y-tsne",
        output_path=Path(output_dir) / "figure" / "vis_tsne.png",
    )


def _sample_values(values, max_points=200000, seed=123):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(values.shape[0], size=max_points, replace=False)]
    return values


def _density_figure_from_values(real_values, fake_values, title, xlabel, seed=123):
    real_values = _sample_values(real_values, seed=seed)
    fake_values = _sample_values(fake_values, seed=seed + 1)
    value_min = min(real_values.min(), fake_values.min())
    value_max = max(real_values.max(), fake_values.max())
    if value_min == value_max:
        value_min -= 1e-3
        value_max += 1e-3
    x_grid = np.linspace(value_min, value_max, 512)

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.plot(x_grid, _kde_curve(real_values, x_grid), linewidth=2.6, label="Original", color="#d62728")
    ax.plot(x_grid, _kde_curve(fake_values, x_grid), linewidth=2.6, linestyle="--", label="Synthetic", color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18, linewidth=0.6)
    return fig


def save_value_density_plot(real_data, fake_data, output_dir, compare=3000, seed=123):
    real_data, fake_data = _paired_subset(real_data, fake_data, compare, seed)
    fig = _density_figure_from_values(
        real_data.reshape(-1),
        fake_data.reshape(-1),
        title="Value density",
        xlabel="Data value",
        seed=seed,
    )
    return _save_figure(fig, Path(output_dir) / "figure" / "vis_value_density.png")


def save_diff_density_plot(real_data, fake_data, output_dir, compare=3000, seed=123):
    real_data, fake_data = _paired_subset(real_data, fake_data, compare, seed)
    fig = _density_figure_from_values(
        np.diff(real_data, axis=1).reshape(-1),
        np.diff(fake_data, axis=1).reshape(-1),
        title="First-difference density",
        xlabel="First difference",
        seed=seed,
    )
    return _save_figure(fig, Path(output_dir) / "figure" / "vis_diff_density.png")


def save_spectrum_density_plot(real_data, fake_data, output_dir, compare=3000, seed=123):
    real_data, fake_data = _paired_subset(real_data, fake_data, compare, seed)
    real_amp = np.abs(np.fft.rfft(real_data, axis=1))[:, 1:, :]
    fake_amp = np.abs(np.fft.rfft(fake_data, axis=1))[:, 1:, :]
    fig = _density_figure_from_values(
        np.log1p(real_amp).reshape(-1),
        np.log1p(fake_amp).reshape(-1),
        title="Spectrum amplitude density",
        xlabel="log(1 + amplitude)",
        seed=seed,
    )
    return _save_figure(fig, Path(output_dir) / "figure" / "vis_spectrum_density.png")


def save_density_plot(real_data, fake_data, output_dir, compare=3000, seed=123):
    real_data, fake_data = _paired_subset(real_data, fake_data, compare, seed)
    fig = _density_figure_from_values(
        real_data.reshape(-1),
        fake_data.reshape(-1),
        title="Data density estimate",
        xlabel="Data value",
        seed=seed,
    )
    return _save_figure(fig, Path(output_dir) / "figure" / "vis_density.png")


def save_sample_plot(real_data, fake_data, output_dir, seed=123):
    real_data, fake_data = _paired_subset(real_data, fake_data, compare=4, seed=seed)
    sample_count = real_data.shape[0]
    rows = 2 if sample_count > 2 else 1
    cols = 2 if sample_count > 1 else 1
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(SAMPLE_PANEL_FIGSIZE[0] * cols, SAMPLE_PANEL_FIGSIZE[1] * rows),
    )
    axes = np.atleast_1d(axes).flatten()

    for idx in range(sample_count):
        ax = axes[idx]
        ax.plot(real_data[idx, :, 0], label="Original", color="#d62728", alpha=0.82, linewidth=1.8)
        ax.plot(fake_data[idx, :, 0], label="Synthetic", color="#1f77b4", alpha=0.82, linewidth=1.8, linestyle="--")
        ax.set_title(f"Sample {idx + 1}")
        ax.grid(alpha=0.18, linewidth=0.6)
        if idx == 0:
            ax.legend(frameon=False)

    for ax in axes[sample_count:]:
        ax.axis("off")

    fig.tight_layout()
    return _save_figure(fig, Path(output_dir) / "figure" / "vis_sample.png")


def save_sample_comparison(real_data, fake_data, output_dir, seed=123):
    return save_sample_plot(real_data, fake_data, output_dir, seed=seed)


def save_temporal_statistics(real_data, fake_data, output_dir, compare=3000, seed=123):
    real_features, fake_features = _sequence_features(real_data, fake_data, compare, seed)
    real_mean = real_features.mean(axis=0)
    fake_mean = fake_features.mean(axis=0)
    real_std = real_features.std(axis=0)
    fake_std = fake_features.std(axis=0)
    time = np.arange(real_mean.shape[0])

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.plot(time, real_mean, label="Original mean", color="#d62728", linewidth=2.4)
    ax.fill_between(time, real_mean - real_std, real_mean + real_std, color="#d62728", alpha=0.18, linewidth=0)
    ax.plot(time, fake_mean, label="Synthetic mean", color="#1f77b4", linewidth=2.4, linestyle="--")
    ax.fill_between(time, fake_mean - fake_std, fake_mean + fake_std, color="#1f77b4", alpha=0.18, linewidth=0)
    ax.set_title("Temporal statistics")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Channel-averaged value")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18, linewidth=0.6)
    return _save_figure(fig, Path(output_dir) / "figure" / "vis_temporal_stats.png")


def _mean_autocorrelation(data, max_lag):
    data = _as_time_series(data)
    centered = data - data.mean(axis=1, keepdims=True)
    scaled = centered / (centered.std(axis=1, keepdims=True) + 1e-6)
    acf = []
    for lag in range(max_lag + 1):
        left = scaled[:, : scaled.shape[1] - lag, :]
        right = scaled[:, lag:, :]
        acf.append((left * right).mean())
    return np.asarray(acf, dtype=np.float32)


def save_autocorrelation_plot(real_data, fake_data, output_dir, compare=3000, seed=123):
    real_data, fake_data = _paired_subset(real_data, fake_data, compare, seed)
    max_lag = min(128, real_data.shape[1] - 1)
    lags = np.arange(max_lag + 1)
    real_acf = _mean_autocorrelation(real_data, max_lag)
    fake_acf = _mean_autocorrelation(fake_data, max_lag)

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    ax.plot(lags, real_acf, label="Original", color="#d62728", linewidth=2.4)
    ax.plot(lags, fake_acf, label="Synthetic", color="#1f77b4", linewidth=2.4, linestyle="--")
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Mean autocorrelation")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18, linewidth=0.6)
    return _save_figure(fig, Path(output_dir) / "figure" / "vis_autocorrelation.png")


def save_distribution_visualizations(real_data, fake_data, output_dir, compare=3000, seed=123):
    saved = {
        "pca": save_pca_plot(real_data, fake_data, output_dir, compare=compare, seed=seed),
        "tsne": save_tsne_plot(real_data, fake_data, output_dir, compare=compare, seed=seed),
        "density": save_density_plot(real_data, fake_data, output_dir, compare=compare, seed=seed),
        "value_density": save_value_density_plot(real_data, fake_data, output_dir, compare=compare, seed=seed),
        "diff_density": save_diff_density_plot(real_data, fake_data, output_dir, compare=compare, seed=seed),
        "spectrum_density": save_spectrum_density_plot(real_data, fake_data, output_dir, compare=compare, seed=seed),
        "autocorrelation": save_autocorrelation_plot(real_data, fake_data, output_dir, compare=compare, seed=seed),
        "sample": save_sample_plot(real_data, fake_data, output_dir, seed=seed),
    }
    return saved
