"""Predictive score for synthetic time-series evaluation.

The metric follows the TimeGAN protocol: train a post-hoc recurrent predictor on
generated data, then report one-step-ahead MAE on the original data.
"""

import numpy as np
import sys
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        return torch.sigmoid(self.proj(output))


def _prepare_sequences(data):
    data = [np.asarray(seq, dtype=np.float32) for seq in data]
    if not data:
        raise ValueError("predictive_score_metrics received an empty dataset.")

    dim = data[0].shape[-1]
    input_dim = dim - 1 if dim > 1 else 1

    prepared = []
    for seq in data:
        if len(seq) < 2:
            continue
        if dim > 1:
            x = seq[:-1, : dim - 1]
            y = seq[1:, dim - 1 : dim]
        else:
            x = seq[:-1, :1]
            y = seq[1:, :1]
        prepared.append((x, y, len(x)))

    if not prepared:
        raise ValueError("predictive_score_metrics needs sequences with length at least 2.")
    return prepared, input_dim


def _batch_to_tensors(prepared, indices, device):
    batch = [prepared[i] for i in indices]
    batch_size = len(batch)
    max_len = max(length for _, _, length in batch)
    input_dim = batch[0][0].shape[-1]

    x = torch.zeros(batch_size, max_len, input_dim, device=device)
    y = torch.zeros(batch_size, max_len, 1, device=device)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    for row, (seq_x, seq_y, length) in enumerate(batch):
        x[row, :length] = torch.as_tensor(seq_x, device=device)
        y[row, :length] = torch.as_tensor(seq_y, device=device)
        mask[row, :length] = True

    return x, y, mask


def predictive_score_metrics(
    ori_data,
    generated_data,
    iterations=5000,
    batch_size=128,
    device=None,
):
    """Report post-hoc one-step prediction MAE.

    Args:
        ori_data: original data with shape/list entries like (L, C).
        generated_data: generated synthetic data with shape/list entries like (L, C).
        iterations: optimizer steps for the post-hoc predictor.
        batch_size: mini-batch size for predictor training.
        device: optional torch device.

    Returns:
        MAE of predictions on original data.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ori_prepared, input_dim = _prepare_sequences(ori_data)
    generated_prepared, generated_input_dim = _prepare_sequences(generated_data)
    if generated_input_dim != input_dim:
        raise ValueError(
            f"Feature mismatch between original and generated data: {input_dim} vs {generated_input_dim}"
        )

    hidden_dim = max(1, input_dim // 2)
    model = GRUPredictor(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    train_size = len(generated_prepared)
    effective_batch = min(batch_size, train_size)
    for _ in tqdm(range(iterations), desc="training", total=iterations, disable=not sys.stderr.isatty()):
        indices = np.random.permutation(train_size)[:effective_batch]
        x, y, mask = _batch_to_tensors(generated_prepared, indices, device)

        pred = model(x)
        loss = torch.abs(pred - y)[mask].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(ori_prepared), batch_size):
            indices = np.arange(start, min(start + batch_size, len(ori_prepared)))
            x, y, mask = _batch_to_tensors(ori_prepared, indices, device)
            pred = model(x)
            errors.append(torch.abs(pred - y)[mask].detach().cpu())

    return float(torch.cat(errors).mean().item())
