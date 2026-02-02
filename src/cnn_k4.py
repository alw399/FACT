"""
Multi-input BPNet-style model: sequence + additional bigWig -> target bigWig.

This builds off `cnn.py`:
- Uses the same BPNetModel-style encoder/heads idea (implemented here with 5 input channels)
- Uses BPNetLoss (BCEWithLogitsLoss) and TrainConfig from `cnn.py`

Expected dataset: `SequenceDualBigWigDataset` from `datas.py`, which yields:
    (sequence, additional_y, target_y)
where target_y is binary (0s and 1s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import enlighten
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from cnn import BPNetLoss, TrainConfig
from datas import SequenceDualBigWigDataset


class BPNetK4Model(nn.Module):
    """
    BPNet-like model with an extra 1-channel signal input.

    Inputs:
      - sequence: (batch, 4, L_seq)
      - additional_y: (batch, L_sig) or (batch, 1, L_sig)

    The additional signal is interpolated to L_seq if needed, then concatenated
    to the sequence channels => (batch, 5, L_seq).

    Outputs:
      - profile_logits: (batch, 1, L_out) - logits for binary classification
    """

    def __init__(
        self,
        seq_len: int,
        output_len: int,
        n_channels_seq: int = 4,
        hidden_channels: int = 64,
        n_encoder_layers: int = 9,
        kernel_size: int = 25,
        profile_kernel_size: int = 75,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.output_len = output_len
        self.n_channels_seq = n_channels_seq

        n_in = n_channels_seq + 1  # +1 for additional bigWig channel

        # Encoder (same padding, dilated conv)
        encoder_layers = []
        in_ch = n_in
        for i in range(n_encoder_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation // 2
            encoder_layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            encoder_layers.append(nn.BatchNorm1d(hidden_channels))
            encoder_layers.append(nn.ReLU())
            in_ch = hidden_channels
        self.encoder = nn.Sequential(*encoder_layers)

        # Profile head
        profile_padding = (profile_kernel_size - 1) // 2
        self.profile_head = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=profile_kernel_size,
            padding=profile_padding,
        )

    def forward(self, sequence: torch.Tensor, additional_y: torch.Tensor) -> torch.Tensor:
        # Ensure additional has shape (B, 1, L_sig)
        if additional_y.dim() == 2:
            additional_y = additional_y.unsqueeze(1)

        # Match length to sequence length if binned
        L_seq = sequence.shape[-1]
        if additional_y.shape[-1] != L_seq:
            additional_y = nn.functional.interpolate(
                additional_y, size=L_seq, mode="linear", align_corners=False
            )

        x = torch.cat([sequence, additional_y], dim=1)  # (B, 5, L_seq)

        h = self.encoder(x)  # (B, hidden, L_seq)

        profile_logits = self.profile_head(h)  # (B, 1, L_seq)

        # Match profile length to target bins (like BPNetModel in cnn.py)
        if profile_logits.shape[-1] != self.output_len:
            profile_logits = nn.functional.interpolate(
                profile_logits, size=self.output_len, mode="linear", align_corners=False
            )

        return profile_logits


def train_bpnet_k4(
    dataset: SequenceDualBigWigDataset,
    batch_size: int = 32,
    config: Optional[TrainConfig] = None,
    num_workers: int = 0,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_seed: int = 42,
) -> BPNetK4Model:
    """
    Train BPNetK4Model using BPNetLoss (BCEWithLogitsLoss) from `cnn.py`.

    Dataset items must be: (sequence, additional_y, target_y)
    where target_y is binary (0s and 1s).
    """
    if config is None:
        config = TrainConfig()

    frac_sum = train_frac + val_frac + test_frac
    train_frac /= frac_sum
    val_frac /= frac_sum
    test_frac /= frac_sum

    n = len(dataset)
    train_size = int(train_frac * n)
    val_size = int(val_frac * n)
    test_size = n - train_size - val_size
    if train_size == 0 or val_size == 0 or test_size == 0:
        raise ValueError("Dataset too small for requested train/val/test split.")

    g = torch.Generator().manual_seed(split_seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Infer lengths
    seq0, add0, y0 = next(iter(train_loader))
    seq_len = seq0.shape[-1]
    out_len = y0.shape[-1]

    model = BPNetK4Model(seq_len=seq_len, output_len=out_len).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = BPNetLoss()

    # attach loaders for visualization
    model.train_loader = train_loader
    model.val_loader = val_loader
    model.test_loader = test_loader

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    no_improve = 0

    manager = enlighten.get_manager()

    def _epoch_loss(loader, train: bool) -> float:
        total = 0.0
        n_batches = 0
        if train:
            model.train()
        else:
            model.eval()
        for seq, add, y in loader:
            seq = seq.to(config.device)
            add = add.to(config.device)
            y = y.to(config.device)

            if train:
                optimizer.zero_grad()

            profile_logits = model(seq, add)
            loss = criterion(profile_logits, y)

            if train:
                loss.backward()
                optimizer.step()

            total += float(loss.item())
            n_batches += 1
        return total / max(n_batches, 1)

    for epoch in range(config.epochs):
        pbar = manager.counter(total=1, desc=f"Epoch {epoch+1}/{config.epochs}", unit="epoch", leave=False)
        train_loss = _epoch_loss(train_loader, train=True)
        val_loss = _epoch_loss(val_loader, train=False)
        pbar.update()
        pbar.close()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{config.epochs} - train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.patience:
                print(f"Early stopping after {epoch+1} epochs (patience={config.patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = _epoch_loss(test_loader, train=False)
    print("\nFinal losses:")
    print(f"  Train: {train_losses[-1]:.4f}")
    print(f"  Val:   {val_losses[-1]:.4f}")
    print(f"  Test:  {test_loss:.4f}")

    model.history = {"train": train_losses, "val": val_losses, "final_test": test_loss}
    return model

__all__ = ["BPNetK4Model", "train_bpnet_k4"]

