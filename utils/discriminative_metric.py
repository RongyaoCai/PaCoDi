"""Discriminative score for synthetic time-series evaluation.

The metric follows the TimeGAN protocol: train a post-hoc recurrent classifier
to distinguish original and generated sequences, then report abs(acc - 0.5).
"""

import numpy as np
import sys
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class GRUDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        output, _ = self.gru(x)
        last_index = (lengths - 1).clamp(min=0).view(-1, 1, 1)
        last_index = last_index.expand(-1, 1, output.shape[-1])
        last_state = output.gather(dim=1, index=last_index).squeeze(1)
        return self.proj(last_state)


def _extract_time(data):
    time = []
    max_seq_len = 0
    for seq in data:
        length = len(seq[:, 0])
        max_seq_len = max(max_seq_len, length)
        time.append(length)
    return time, max_seq_len


def _train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def _batch_to_tensors(data, time, indices, device):
    batch = [(np.asarray(data[i], dtype=np.float32), int(time[i])) for i in indices]
    batch_size = len(batch)
    max_len = max(length for _, length in batch)
    dim = batch[0][0].shape[-1]

    x = torch.zeros(batch_size, max_len, dim, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    for row, (seq, length) in enumerate(batch):
        x[row, :length] = torch.as_tensor(seq[:length], device=device)
        lengths[row] = length
    return x, lengths


def _sample_batch(data, time, batch_size, device):
    effective_batch = min(batch_size, len(data))
    indices = np.random.permutation(len(data))[:effective_batch]
    return _batch_to_tensors(data, time, indices, device)


def _accuracy(labels, preds):
    labels = np.asarray(labels, dtype=bool)
    preds = np.asarray(preds, dtype=bool)
    return float((labels == preds).mean())


def discriminative_score_metrics(
    ori_data,
    generated_data,
    iterations=2000,
    batch_size=128,
    device=None,
):
    """Use a post-hoc RNN classifier to separate original and synthetic data.

    Args:
        ori_data: original data with shape/list entries like (L, C).
        generated_data: generated synthetic data with shape/list entries like (L, C).
        iterations: optimizer steps for the post-hoc discriminator.
        batch_size: mini-batch size for discriminator training.
        device: optional torch device.

    Returns:
        Tuple of (discriminative_score, fake_acc, real_acc).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ori_data = [np.asarray(seq, dtype=np.float32) for seq in ori_data]
    generated_data = [np.asarray(seq, dtype=np.float32) for seq in generated_data]
    if not ori_data or not generated_data:
        raise ValueError("discriminative_score_metrics received an empty dataset.")
    dim = ori_data[0].shape[-1]

    ori_time, _ = _extract_time(ori_data)
    generated_time, _ = _extract_time(generated_data)

    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = _train_test_divide(
        ori_data,
        generated_data,
        ori_time,
        generated_time,
    )

    hidden_dim = max(1, dim // 2)
    model = GRUDiscriminator(input_dim=dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in tqdm(range(iterations), desc="training", total=iterations, disable=not sys.stderr.isatty()):
        real_x, real_t = _sample_batch(train_x, train_t, batch_size, device)
        fake_x, fake_t = _sample_batch(train_x_hat, train_t_hat, batch_size, device)

        real_logit = model(real_x, real_t)
        fake_logit = model(fake_x, fake_t)
        real_label = torch.ones_like(real_logit)
        fake_label = torch.zeros_like(fake_logit)
        loss = criterion(real_logit, real_label) + criterion(fake_logit, fake_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        real_x, real_t = _batch_to_tensors(test_x, test_t, np.arange(len(test_x)), device)
        fake_x, fake_t = _batch_to_tensors(test_x_hat, test_t_hat, np.arange(len(test_x_hat)), device)
        real_pred = torch.sigmoid(model(real_x, real_t)).detach().cpu().numpy().reshape(-1)
        fake_pred = torch.sigmoid(model(fake_x, fake_t)).detach().cpu().numpy().reshape(-1)

    y_pred = np.concatenate((real_pred, fake_pred), axis=0)
    y_label = np.concatenate((np.ones(len(real_pred)), np.zeros(len(fake_pred))), axis=0)

    acc = _accuracy(y_label, y_pred > 0.5)
    fake_acc = _accuracy(np.zeros(len(fake_pred)), fake_pred > 0.5)
    real_acc = _accuracy(np.ones(len(real_pred)), real_pred > 0.5)
    discriminative_score = np.abs(0.5 - acc)
    return discriminative_score, fake_acc, real_acc
