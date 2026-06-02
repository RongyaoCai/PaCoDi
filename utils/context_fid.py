import scipy
import numpy as np
from models.ts2vec.ts2vec import TS2Vec


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

def context_fid(ori_data, generated_data):
    model = TS2Vec(input_dims=ori_data.shape[-1], device=0, batch_size=8, lr=0.001, output_dims=320,
                   max_train_length=3000)
    model.fit(ori_data, verbose=False)
    ori_representation = model.encode(ori_data, encoding_window='full_series')
    gen_representation = model.encode(generated_data, encoding_window='full_series')
    idx = np.random.permutation(ori_data.shape[0])
    ori_representation = ori_representation[idx]
    gen_representation = gen_representation[idx]
    results = calculate_fid(ori_representation, gen_representation)
    return results

