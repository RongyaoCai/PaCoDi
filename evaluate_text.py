import os
import numpy as np
import pandas as pd

def calculate_metrics(pred, true):
    min_len = min(len(pred), len(true))
    pred = pred[:min_len]
    true = true[:min_len]
    mse = np.mean((pred - true) ** 2)
    sum_abs_diff = np.sum(np.abs(pred - true))
    sum_abs_true = np.sum(np.abs(true))
    wape = sum_abs_diff / sum_abs_true if sum_abs_true != 0 else 0.0
    return mse, wape

def align_shapes(preds, truth):
    if preds.ndim == 3 and preds.shape[-1] == 1 and truth.ndim == 2:
        preds = preds.squeeze(-1)
    elif truth.ndim == 3 and truth.shape[-1] == 1 and preds.ndim == 2:
        truth = truth.squeeze(-1)
    return preds, truth

if __name__ == "__main__":
    experiments = [('etth1', 24)]
    csv_file = 'metrics_text.csv'
    model_name = "pacodi_ddpm"

    for dataset, length in experiments:
        exp_name = f"{dataset}_{length}"
        base_dir = f"./experiments/text/{model_name}/{exp_name}"
        real_path = f"{base_dir}/samples/reals.npy"
        samples_dir = f"{base_dir}/samples"

        ground_truth = np.load(real_path)
        runs = [d for d in os.listdir(samples_dir)
                if d.startswith('run_') and os.path.isdir(os.path.join(samples_dir, d))]

        run_mses = []
        run_wapes = []

        for run_name in runs:
            sample_file_path = os.path.join(samples_dir, run_name, 'samples.npy')
            if not os.path.exists(sample_file_path):
                continue
            preds = np.load(sample_file_path)
            preds, truth = align_shapes(preds, ground_truth)

            mse, wape = calculate_metrics(preds, truth)
            run_mses.append(mse)
            run_wapes.append(wape)

        if run_mses:
            avg_mse = np.mean(run_mses)
            avg_wape = np.mean(run_wapes)
            row = {
                'Datasets': dataset,
                'Length': length,
                'Pacodi WAPE↓': f"{avg_wape:.4f}",
                'MSE↓': f"{avg_mse:.4f}"
            }
        else:
            row = {
                'Datasets': dataset,
                'Length': length,
                'Pacodi WAPE↓': "-",
                'MSE↓': "-"
            }

        df_row = pd.DataFrame([row])
        write_header = not os.path.exists(csv_file)
        df_row.to_csv(csv_file, mode='a', index=False, header=write_header)

