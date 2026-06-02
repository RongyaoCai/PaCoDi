import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from utils.runtime import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data_root,
        window,
        channels,
        stride=1,
        split_ratio=0.9,
        period="train",
        normalize=True,
        neg_one_to_one=True,
        save2npy=True,
        output_dir="./output",
    ):
        super().__init__()
        assert period in ["train", "test"], "period must be train or test."

        self.data_root = data_root
        self.window = int(window)
        self.var_num = int(channels)
        self.stride = int(stride)
        self.period = period
        self.normalize_data = normalize
        self.auto_norm = neg_one_to_one
        self.save2npy = save2npy
        self.dir = os.path.join(output_dir, "truth")
        os.makedirs(self.dir, exist_ok=True)

        rawdata = np.load(data_root).astype(np.float32)
        self.rawdata = self._to_windows(rawdata)
        if self.rawdata.shape[1] != self.window or self.rawdata.shape[2] != self.var_num:
            raise ValueError(
                f"Expected data shape (N, {self.window}, {self.var_num}), got {self.rawdata.shape}."
            )

        self.scaler = MinMaxScaler()
        if self.normalize_data:
            flattened = self.rawdata.reshape(-1, self.var_num)
            self.scaler.fit(flattened)
            zero_one = self.scaler.transform(flattened).reshape(self.rawdata.shape)
        else:
            zero_one = self.rawdata.copy()

        model_data = normalize_to_neg_one_to_one(zero_one) if self.auto_norm else zero_one
        train_data, test_data = self._split(model_data, split_ratio)
        self.samples = train_data if period == "train" else test_data
        self.sample_num = self.samples.shape[0]

        if self.save2npy:
            train_raw, test_raw = self._split(self.rawdata, split_ratio)
            if period == "train":
                self._save_truth("train", train_data, train_raw)
            else:
                self._save_truth("test", test_data, test_raw)

    def _to_windows(self, rawdata):
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}.")

        if rawdata.ndim == 3:
            if rawdata.shape[2] != self.var_num:
                raise ValueError(f"Expected {self.var_num} channels, got {rawdata.shape[2]}.")
            if rawdata.shape[1] < self.window:
                raise ValueError(f"Series length {rawdata.shape[1]} is shorter than window {self.window}.")

            samples = []
            for series in rawdata:
                for start in range(0, series.shape[0] - self.window + 1, self.stride):
                    samples.append(series[start:start + self.window])
            return np.asarray(samples, dtype=np.float32)

        if rawdata.ndim != 2:
            raise ValueError(f"Expected .npy data with shape (T, C) or (N, T, C), got {rawdata.shape}.")
        if rawdata.shape[1] != self.var_num:
            raise ValueError(f"Expected {self.var_num} channels, got {rawdata.shape[1]}.")

        sample_num = rawdata.shape[0] - self.window + 1
        if sample_num <= 0:
            raise ValueError(f"Data length {rawdata.shape[0]} is shorter than window {self.window}.")

        return np.asarray(
            [rawdata[start:start + self.window] for start in range(0, sample_num, self.stride)],
            dtype=np.float32,
        )

    @staticmethod
    def _split(data, split_ratio):
        split = int(round(len(data) * float(split_ratio)))
        return data[:split], data[split:]

    def _save_truth(self, period, model_data, raw_data):
        np.save(os.path.join(self.dir, f"ground_truth_{period}.npy"), raw_data)
        if self.auto_norm:
            np.save(os.path.join(self.dir, f"norm_truth_{period}.npy"), unnormalize_to_zero_to_one(model_data))
        else:
            np.save(os.path.join(self.dir, f"norm_truth_{period}.npy"), model_data)

    def unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        if not self.normalize_data:
            return data
        shape = data.shape
        return self.scaler.inverse_transform(data.reshape(-1, self.var_num)).reshape(shape)

    def __getitem__(self, ind):
        return torch.from_numpy(self.samples[ind]).float()

    def __len__(self):
        return self.sample_num
