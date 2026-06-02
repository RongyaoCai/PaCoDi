import argparse
from pathlib import Path

import numpy as np


def generate_sines(num_series, series_length, channels, seed):
    rng = np.random.default_rng(seed)
    steps = np.arange(series_length, dtype=np.float32)
    data = np.empty((num_series, series_length, channels), dtype=np.float32)

    for index in range(num_series):
        freq = rng.uniform(0.0, 0.1, size=(channels, 1)).astype(np.float32)
        phase = rng.uniform(0.0, 0.1, size=(channels, 1)).astype(np.float32)
        sample = np.sin(freq * steps[None, :] + phase)
        data[index] = ((sample.T + 1.0) * 0.5).astype(np.float32)

    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fixed sine benchmark datasets.")
    parser.add_argument("--output-dir", default="data/datasets")
    parser.add_argument("--series-length", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=5)
    parser.add_argument("--num-series", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = generate_sines(args.num_series, args.series_length, args.channels, args.seed)

    data_path = output_dir / "sines.npy"
    np.save(data_path, data)
    print(f"saved {data_path} shape={data.shape} seed={args.seed}")


if __name__ == "__main__":
    main()
