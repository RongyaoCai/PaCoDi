import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.conditional_evaluation import compute_conditional_metrics, save_conditional_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate conditional generation experiment outputs.")
    parser.add_argument("--root", default="experiments/conditional", help="Root directory for conditional experiments.")
    parser.add_argument("--model-name", default="pacodi_ddpm")
    parser.add_argument("--dataset-name", default="etth1")
    parser.add_argument("--experiment", default=None, help="Single experiment name. Defaults to all dataset experiments.")
    parser.add_argument("--output-csv", default="metrics_conditional.csv")
    parser.add_argument("--compare", type=int, default=3000)
    return parser.parse_args()


def iter_experiments(root, model_name, dataset_name, experiment):
    dataset_root = Path(root) / model_name / dataset_name
    if experiment:
        yield dataset_root / experiment
        return

    if not dataset_root.exists():
        return
    for path in sorted(dataset_root.iterdir()):
        if path.is_dir():
            yield path


def run_sort_key(path):
    name = path.name
    if name.startswith("run_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            pass
    return name


def evaluate_experiment(exp_dir, compare):
    samples_dir = exp_dir / "samples"
    real_path = samples_dir / "reals.npy"
    if not real_path.exists():
        print(f"Skip {exp_dir.name}: missing {real_path}")
        return None

    ground_truth = np.load(real_path)
    runs = sorted(
        [path for path in samples_dir.iterdir() if path.is_dir() and path.name.startswith("run_")],
        key=run_sort_key,
    )

    rows = []
    for run_dir in runs:
        sample_path = run_dir / "samples.npy"
        if not sample_path.exists():
            continue
        samples = np.load(sample_path)
        sample_count = min(len(samples), len(ground_truth))
        samples = samples[:sample_count]
        truth = ground_truth[:sample_count]

        if run_dir.name == "run_0":
            paths = save_conditional_visualizations(truth, samples, run_dir, compare=compare)
            print(f"Saved conditional visualizations for {exp_dir.name}: {paths}")

        metrics = compute_conditional_metrics(truth, samples)
        metrics["run_id"] = run_dir.name
        rows.append(metrics)

    if not rows:
        print(f"Skip {exp_dir.name}: no run samples found")
        return None

    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float, np.integer, np.floating))]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in numeric_keys}
    summary.update(
        {
            "mode": "conditional",
            "dataset_name": exp_dir.parent.name,
            "experiment": exp_dir.name,
            "runs": len(rows),
        }
    )
    return summary


def main():
    args = parse_args()
    rows = []
    for exp_dir in iter_experiments(args.root, args.model_name, args.dataset_name, args.experiment):
        row = evaluate_experiment(exp_dir, args.compare)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No conditional experiments evaluated.")
        return

    output = pd.DataFrame(rows)
    write_header = not os.path.exists(args.output_csv)
    output.to_csv(args.output_csv, mode="a", index=False, header=write_header, float_format="%.5f")
    print(f"Saved conditional metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
