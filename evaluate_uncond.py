import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import torch
from Utils.context_fid import Context_FID
from Utils.cross_correlation import CrossCorrelLoss
from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


def save_visualization(ori_data, generated_data, save_path, analysis='pca', compare=3000):
    """ PCA/t-SNE/Kernel"""
    anal_sample_no = min([compare, ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    if ori_data.ndim == 2: ori_data = ori_data[..., np.newaxis]
    if generated_data.ndim == 2: generated_data = generated_data[..., np.newaxis]

    prep_data = np.mean(ori_data, axis=2)
    prep_data_hat = np.mean(generated_data, axis=2)

    plt.figure(figsize=(8, 6))

    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c="red", alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], c="blue", alpha=0.2, label="Synthetic")
        plt.title('PCA plot')

    elif analysis == 'tsne':
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)
        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1], c="red", alpha=0.2,
                    label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1], c="blue", alpha=0.2,
                    label="Synthetic")
        plt.title('t-SNE plot')

    elif analysis == 'kernel':
        sns.kdeplot(prep_data.flatten(), linewidth=3, label='Original', color="red")
        sns.kdeplot(prep_data_hat.flatten(), linewidth=3, linestyle='--', label='Synthetic', color="blue")
        plt.title('Kernel Density Estimation')

    plt.legend()
    figure_dir = os.path.join(save_path, "figure")
    os.makedirs(figure_dir, exist_ok=True)
    save_file = os.path.join(figure_dir, f"vis_{analysis}.png")
    plt.savefig(save_file)
    plt.close()


def plot_4_samples(ori_data, fake_data, save_path):
    f, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()
    for i in range(4):
        idx_real = np.random.randint(len(ori_data))
        idx_fake = np.random.randint(len(fake_data))
        dim_idx = 0
        ax[i].plot(ori_data[idx_real, :, dim_idx], label='Original', color='red', alpha=0.7)
        ax[i].plot(fake_data[idx_fake, :, dim_idx], label='Synthetic', color='blue', alpha=0.7, linestyle='--')
        ax[i].set_title(f"Sample {i + 1}")
        ax[i].legend()
    plt.tight_layout()
    figure_dir = os.path.join(save_path, "figure")
    os.makedirs(figure_dir, exist_ok=True)
    save_file = os.path.join(figure_dir, "sample_comparison.png")
    plt.savefig(save_file)
    plt.close()

def evaluate_metrics(real_data, fake_data, iterations=3):
    scores = {
        'Context_FID': [],
        'Cross_Corr': [],
        'Discriminative_Score': [],
        'Predictive_Score': []
    }

    x_real_torch = torch.from_numpy(real_data).float()
    x_fake_torch = torch.from_numpy(fake_data).float()

    for i in range(iterations):
        # Context-FID
        cfid = Context_FID(real_data, fake_data)
        scores['Context_FID'].append(cfid)

        # Correlational Score (Cross-Corr)
        idx_r = np.random.randint(0, len(x_real_torch), min(1000, len(x_real_torch)))
        idx_f = np.random.randint(0, len(x_fake_torch), min(1000, len(x_fake_torch)))
        corr = CrossCorrelLoss(x_real_torch[idx_r], name='CrossCorrelLoss')
        loss = corr.compute(x_fake_torch[idx_f])
        scores['Cross_Corr'].append(loss.item())

        # Discriminative Score
        d_score, _, _ = discriminative_score_metrics(real_data, fake_data)
        scores['Discriminative_Score'].append(d_score)

        # Predictive Score
        p_score = predictive_score_metrics(real_data, fake_data)
        scores['Predictive_Score'].append(p_score)

    final_results = {}
    for key, val_list in scores.items():
        val_list = [v for v in val_list if not np.isnan(v)]
        if len(val_list) > 0:
            mean = np.mean(val_list)
            sem = scipy.stats.sem(val_list)
            if len(val_list) > 1:
                ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., len(val_list) - 1)
            else:
                ci = 0.0

            mean_str = f"{mean:.3f}"
            ci_str = f"{ci:.3f}".lstrip('0')
            if ci_str.startswith('.'):
                pass
            elif ci == 0:
                ci_str = ".000"

            final_results[key] = f"{mean_str} ± {ci_str}"
        else:
            final_results[key] = "NaN"

    return final_results

if __name__ == "__main__":

    experiments = [ ('etth', 24)
        # ('etth', 24), ('etth', 64), ('etth', 128), ('etth', 256),
        #('sines', 24), ('sines', 64), ('sines', 128), ('sines', 256)
        # ('air', 24), ('air', 64), ('air', 128), ('air', 256),
        # ('stocks', 24), ('stocks', 64), ('stocks', 128), ('stocks', 256)
    ]


    csv_file = "metrics_uncond.csv"

    model_name = "pacodi_sde"

    rename_map = {
        'Cross_Corr': 'Correlational Score',
        'Context_FID': 'Context-FID Score',
        'Discriminative_Score': 'Discriminative Score',
        'Predictive_Score': 'Predictive Score'
    }

    target_cols = [
        'Dataset', 'Length',
        'Context-FID Score',
        'Correlational Score',
        'Discriminative Score',
        'Predictive Score'
    ]

    for dataset, seq_len in experiments:
        exp_name = f"{dataset}_{seq_len}"
        if dataset == 'sines':
            base_dir = f"./experiments/uncond/{model_name}/{exp_name}"
            real_path = f"{base_dir}/samples/truth/ground_truth_train.npy"
            fake_path = f"{base_dir}/samples/samples.npy"
        else:
            base_dir = f"./experiments/uncond/{model_name}/{exp_name}"
            real_path = f"{base_dir}/samples/truth/norm_truth_train.npy"
            fake_path = f"{base_dir}/samples/samples.npy"

        if os.path.exists(real_path) and os.path.exists(fake_path):
            ori_data = np.load(real_path)
            fake_data = np.load(fake_path)

            print(f"DEBUG Check:")
            print(f"ori_data type: {type(ori_data)}")
            if hasattr(ori_data, 'shape'):
                print(f"ori_data shape: {ori_data.shape}")
            else:
                print(f"ori_data is {ori_data}")

            print(f"fake_data type: {type(fake_data)}")
            if hasattr(fake_data, 'shape'):
                print(f"fake_data shape: {fake_data.shape}")
            else:
                print(f"fake_data is {fake_data}")

            min_len = min(len(ori_data), len(fake_data))
            ori_data = ori_data[:min_len]
            fake_data = fake_data[:min_len]

            save_visualization(ori_data, fake_data, base_dir, 'pca', compare=3000)
            save_visualization(ori_data, fake_data, base_dir, 'tsne', compare=3000)
            save_visualization(ori_data, fake_data, base_dir, 'kernel', compare=3000)
            plot_4_samples(ori_data, fake_data, base_dir)

            results = evaluate_metrics(ori_data, fake_data, iterations=1)

            row = {'Dataset': dataset, 'Length': seq_len}
            row.update(results)

            df_row = pd.DataFrame([row])
            df_row = df_row.rename(columns=rename_map)

            current_cols = [c for c in target_cols if c in df_row.columns]
            df_row = df_row[current_cols]

            write_header = not os.path.exists(csv_file)
            df_row.to_csv(csv_file, mode='a', index=False, header=write_header)
