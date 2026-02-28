## Necessary Packages
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def display_scores(results):
   mean = np.mean(results)
   sigma = scipy.stats.sem(results)
   sigma = sigma * scipy.stats.t.ppf((1 + 0.95) / 2., 5-1)
  #  sigma = 1.96*(np.std(results)/np.sqrt(len(results)))
   print('Final Score: ', f'{mean} \xB1 {sigma}')


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def visualization(ori_data, generated_data, analysis, compare=3000, save_path=None):
    """
    Using PCA or tSNE for generated and original data visualization.

    Args:
        - ori_data:  (N, L, M)
        - generated_data:  (N, L, M)
        - analysis: 'pca', 'tsne' 或 'kernel'
        - compare: Number of samples used for visualization
        - save_path: path to the saved image（如 './results/pca.png'）
    """
    anal_sample_no = min([compare, ori_data.shape[0], generated_data.shape[0]])
    idx_ori = np.random.permutation(ori_data.shape[0])[:anal_sample_no]
    idx_gen = np.random.permutation(generated_data.shape[0])[:anal_sample_no]

    ori_data = ori_data[idx_ori]
    generated_data = generated_data[idx_gen]

    no, seq_len, dim = ori_data.shape


    prep_data = np.mean(ori_data, axis=2)
    prep_data_hat = np.mean(generated_data, axis=2)

    colors = ["red" for _ in range(anal_sample_no)] + ["blue" for _ in range(anal_sample_no)]

    plt.figure(figsize=(8, 6))

    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c='red', alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c='blue', alpha=0.2, label="Synthetic")
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y-pca')
        plt.legend()

    elif analysis == 'tsne':
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c='red', alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c='blue', alpha=0.2, label="Synthetic")
        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y-tsne')
        plt.legend()

    elif analysis == 'kernel':
        sns.kdeplot(prep_data.flatten(), label='Original', color="red", linewidth=2)
        sns.kdeplot(prep_data_hat.flatten(), label='Synthetic', color="blue", linewidth=2, linestyle='--')
        plt.title('Kernel Density Estimation')
        plt.xlabel('Data Value')
        plt.ylabel('Density')
        plt.legend()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #print(f"图像已保存至: {save_path}")

    plt.show()
    plt.close()

if __name__ == '__main__':
   pass