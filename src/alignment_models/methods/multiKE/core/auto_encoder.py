"""AutoEncoder: PyTorch Module for literal compression (1500-dim → 75-dim)."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 1500, hidden_dims=None, dim: int = 75,
                 activation: str = 'tanh', normalize: bool = True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, dim]

        act_map = {'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        act_cls = act_map.get(activation.lower(), nn.Tanh)

        self.normalize_output = normalize

        enc_dims = [input_dim] + hidden_dims
        enc_layers = []
        for i in range(len(hidden_dims)):
            enc_layers.extend([nn.Linear(enc_dims[i], enc_dims[i + 1]), act_cls()])
        self.encoder_net = nn.Sequential(*enc_layers)

        dec_dims = list(reversed(hidden_dims)) + [input_dim]
        dec_layers = []
        for i in range(len(hidden_dims)):
            dec_layers.extend([nn.Linear(dec_dims[i], dec_dims[i + 1]), act_cls()])
        self.decoder_net = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder_net(x)
        if self.normalize_output:
            out = F.normalize(out, dim=1)
        return out

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def encode_batches(self, data: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        results = []
        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i:i + batch_size], dtype=torch.float32).to(device)
            results.append(self.encode(batch).cpu().numpy())
        return np.vstack(results)


def train_autoencoder(word_vec_array: np.ndarray, dim: int = 75,
                      hidden_dims=None, activation: str = 'tanh',
                      normalize: bool = True, epochs: int = 100,
                      batch_size: int = 512, learning_rate: float = 0.001,
                      optimizer_name: str = 'Adagrad') -> AutoEncoder:
    if normalize:
        word_vec_array = preprocessing.normalize(word_vec_array).astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoEncoder(
        input_dim=word_vec_array.shape[1], hidden_dims=hidden_dims,
        dim=dim, activation=activation, normalize=normalize)
    model.to(device)

    opt_cls = {
        'Adam': torch.optim.Adam,
        'Adagrad': torch.optim.Adagrad,
        'Adadelta': torch.optim.Adadelta,
        'SGD': torch.optim.SGD,
    }.get(optimizer_name, torch.optim.Adagrad)
    optimizer = opt_cls(model.parameters(), lr=learning_rate)

    data_tensor = torch.tensor(word_vec_array, dtype=torch.float32).to(device)
    n = len(data_tensor)

    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            batch = data_tensor[perm[i:i + batch_size]]
            optimizer.zero_grad()
            recon = model(batch)
            loss = torch.mean((recon - batch) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f'epoch {epoch} of literal encoder, loss: {avg:.4f}, time: {time.time() - t0:.4f}s')

    return model
