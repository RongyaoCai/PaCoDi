import scipy
import numpy as np
from Models.ts2vec.ts2vec import TS2Vec


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def Context_FID(ori_data, generated_data):
    model = TS2Vec(input_dims=ori_data.shape[-1], device=0, batch_size=8, lr=0.001, output_dims=320,
                   max_train_length=3000)
    model.fit(ori_data, verbose=False)
    ori_represenation = model.encode(ori_data, encoding_window='full_series')
    gen_represenation = model.encode(generated_data, encoding_window='full_series')
    idx = np.random.permutation(ori_data.shape[0])
    ori_represenation = ori_represenation[idx]
    gen_represenation = gen_represenation[idx]
    results = calculate_fid(ori_represenation, gen_represenation)
    return results


# def calculate_fid(act1, act2):
#     # calculate mean and covariance statistics
#     mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
#     mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
#     # calculate sum squared difference between means
#     ssdiff = np.sum((mu1 - mu2)**2.0)
#     # calculate sqrt of product between cov
#     covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
#     # check and correct imaginary numbers from sqrt
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     # calculate score
#     fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
#     return fid
#
# def Context_FID(ori_data, generated_data):
#     model = TS2Vec(input_dims=ori_data.shape[-1], device=0, batch_size=8, lr=0.001, output_dims=320,
#                    max_train_length=3000)
#     model.fit(ori_data, verbose=False)
#     ori_represenation = model.encode(ori_data, encoding_window='full_series')
#     gen_represenation = model.encode(generated_data, encoding_window='full_series')
#     idx = np.random.permutation(ori_data.shape[0])
#     ori_represenation = ori_represenation[idx]
#     gen_represenation = gen_represenation[idx]
#     results = calculate_fid(ori_represenation, gen_represenation)
#     return results
#
# if __name__ == '__main__':
#     ori_data = np.load('./OUTPUT/PaCoDi_etth/samples/etth_norm_truth_24_train.npy')  # 原始正弦波数据
#     gen_data = np.load('./OUTPUT/PaCoDi_etth/ddpm_fake_PaCoDi_etth.npy')
#
#     # gen_data = np.load('./OUTPUT/test-sine/samples/sine_ground_truth_24_train.npy')
#     # ori_data = np.load('./OUTPUT/test-sine/ddpm_fake_test-sine.npy')  # 原始正弦波数据
#
#     print(f"Original data shape: {ori_data.shape}")  # 预期: (N, 24, 5)
#     print(f"Generated data shape: {gen_data.shape}")  # 预期: (N, 24, 5)
#
#     min_len = min(len(ori_data), len(gen_data))
#     ori_data = ori_data[:min_len]
#     gen_data = gen_data[:min_len]
#
#     print("开始训练 TS2Vec 模型并提取特征...")
#     fid_score = Context_FID(ori_data, gen_data)
#
#     print(f"======================================")
#     print(f"Context-FID Score: {fid_score:.4f}")
#     print(f"======================================")
