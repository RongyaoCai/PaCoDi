import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import torch

from utils.context_fid import context_fid
from utils.cross_correlation import CrossCorrelationLoss
from utils.discriminative_metric import discriminative_score_metrics
from utils.predictive_metric import predictive_score_metrics
from utils.unconditional_evaluation import save_unconditional_visualizations

warnings.filterwarnings("ignore")


def evaluate_metrics(real_data, fake_data, iterations=3):
    scores = {
        "Context_FID": [],
        "Cross_Corr": [],
        "Discriminative_Score": [],
        "Predictive_Score": [],
    }

    x_real_torch = torch.from_numpy(real_data).float()
    x_fake_torch = torch.from_numpy(fake_data).float()

    for _ in range(iterations):
        scores["Context_FID"].append(context_fid(real_data, fake_data))

        idx_r = np.random.randint(0, len(x_real_torch), min(1000, len(x_real_torch)))
        idx_f = np.random.randint(0, len(x_fake_torch), min(1000, len(x_fake_torch)))
        corr = CrossCorrelationLoss(x_real_torch[idx_r], name="CrossCorrelationLoss")
        scores["Cross_Corr"].append(corr.compute(x_fake_torch[idx_f]).item())

        d_score, _, _ = discriminative_score_metrics(real_data, fake_data)
        scores["Discriminative_Score"].append(d_score)
        scores["Predictive_Score"].append(predictive_score_metrics(real_data, fake_data))

    final_results = {}
    for key, values in scores.items():
        values = [value for value in values if not np.isnan(value)]
        if not values:
            final_results[key] = "NaN"
            continue

        mean = np.mean(values)
        if len(values) > 1:
            sem = scipy.stats.sem(values)
            ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2.0, len(values) - 1)
        else:
            ci = 0.0
        final_results[key] = f"{mean:.3f} +/- {ci:.3f}"

    return final_results


if __name__ == "__main__":
    experiments = [("etth", 64), ("etth", 128), ("etth", 256), ("etth", 512)]

    csv_file = "metrics_uncond.csv"
    model_name = "pacodi_sde"
    backbone_tag = "ditv1"
    interaction_tag = "it1"

    rename_map = {
        "Cross_Corr": "Correlational Score",
        "Context_FID": "Context-FID Score",
        "Discriminative_Score": "Discriminative Score",
        "Predictive_Score": "Predictive Score",
    }

    target_cols = [
        "Dataset",
        "Length",
        "Context-FID Score",
        "Correlational Score",
        "Discriminative Score",
        "Predictive Score",
    ]

    for dataset, seq_len in experiments:
        exp_name = f"{dataset}_seq{seq_len}_{backbone_tag}_{interaction_tag}"
        base_dir = f"./experiments/uncond/{model_name}/{dataset}/{exp_name}"
        real_path = f"{base_dir}/samples/truth/norm_truth_train.npy"
        fallback_real_path = f"{base_dir}/samples/truth/ground_truth_train.npy"
        fake_path = f"{base_dir}/samples/samples.npy"
        if not os.path.exists(real_path):
            real_path = fallback_real_path

        if not (os.path.exists(real_path) and os.path.exists(fake_path)):
            print(f"Skip {exp_name}: missing {real_path} or {fake_path}")
            continue

        real_data = np.load(real_path)
        fake_data = np.load(fake_path)

        min_len = min(len(real_data), len(fake_data))
        real_data = real_data[:min_len]
        fake_data = fake_data[:min_len]

        paths = save_unconditional_visualizations(
            real_data,
            fake_data,
            os.path.join(base_dir, "samples"),
            compare=3000,
        )
        print(f"Saved visualizations for {exp_name}: {paths}")

        results = evaluate_metrics(real_data, fake_data, iterations=1)

        row = {"Dataset": dataset, "Length": seq_len}
        row.update(results)

        df_row = pd.DataFrame([row])
        df_row = df_row.rename(columns=rename_map)
        df_row = df_row[[col for col in target_cols if col in df_row.columns]]

        write_header = not os.path.exists(csv_file)
        df_row.to_csv(csv_file, mode="a", index=False, header=write_header)
