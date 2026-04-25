"""
Multi-input BPNet-style model: sequence + additional bigWig -> target bigWig.

This builds off `cnn.py`:
- Uses the same BPNetModel-style encoder/heads idea (implemented here with 5 input channels)
- Uses BPNetLoss and TrainConfig from `cnn.py`

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from cnn import BPNetModel, BPNetLoss, TrainConfig
from datas import SequenceDualBigWigDataset


class BPNetK4Model(BPNetModel):
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
        n_channels: int = 4,
        hidden_channels: int = 64,
        n_encoder_layers: int = 9,
        kernel_size: int = 25,
        profile_kernel_size: int = 75,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            n_channels=n_channels_seq + 1,
            hidden_channels=hidden_channels,
            n_encoder_layers=n_encoder_layers,
            kernel_size=kernel_size,
            profile_kernel_size=profile_kernel_size,
            output_len=output_len,
        )
        self.n_channels_seq = n_channels_seq

    def forward(self, sequence: torch.Tensor, sequence_k4: torch.Tensor) -> torch.Tensor:
        # Ensure additional has shape (B, 1, L_sig)
        if sequence_k4.dim() == 2:
            sequence_k4 = sequence_k4.unsqueeze(1)

        # Match length to sequence length if binned
        L_seq = sequence.shape[-1]
        x = torch.cat([sequence, sequence_k4], dim=1)  # (B, 5, L_seq)

        h = self.encoder(x)  # (B, hidden, L_seq)

        profile_logits = self.profile_head(h)  # (B, 1, L_seq)
        profile_logits = self.mlp(profile_logits)
        
        total_counts = torch.sum(profile_logits, dim=-1)
        return profile_logits, total_counts
    
    
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
    Train BPNetK4Model using BPNetLoss from `cnn.py`.

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
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    criterion = BPNetLoss(loss_type=config.loss_type)
    model.loss_type = config.loss_type

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

            profile_logits, pred_counts = model(seq, add)

            loss = criterion(profile_logits, pred_counts, y)

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

    # Attach full history and loaders to the model for later inspection/saving
    model.history = {
        "train": train_losses,
        "val": val_losses,
        "final": {
            "train": train_losses[-1],
            "val": val_losses[-1],
            "test": test_loss,
        },
    }
    return model



def save_model(model: nn.Module, model_kwargs: dict, path: str) -> None:
    """Save the model architecture, parameters, and current history."""
    import os
    p = os.path.abspath(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)

    class_name = model.__class__.__name__

    checkpoint = {
        "model_class": class_name,
        "model_kwargs": model_kwargs,
        "state_dict": model.state_dict(),
        "history": getattr(model, "history", None),
        "loss_type": getattr(model, "loss_type", "kldiv"),
    }

    torch.save(checkpoint, p)
    print(f"Model saved to {p}")


def load_model(path: str, device: Optional[str] = None) -> Union[BPNetK4Model]:
    """Load model from checkpoint."""
    import os
    p = os.path.abspath(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    checkpoint = torch.load(p, map_location=device)
    model_class_name = checkpoint["model_class"]
    model_kwargs = checkpoint["model_kwargs"]

    if model_class_name == "BPNetK4Model":
        model = BPNetK4Model(**model_kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        if checkpoint.get("history") is not None:
            model.history = checkpoint["history"]
        if checkpoint.get("loss_type") is not None:
            model.loss_type = checkpoint["loss_type"]
        if device is not None:
            model.to(device)
        return model
    else:
        raise ValueError(f"Unsupported model class: {model_class_name}")


__all__ = ["BPNetK4Model", "train_bpnet_k4", "save_model", "load_model"]

