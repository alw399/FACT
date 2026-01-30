"""
CNN-based predictor for CUT&RUN signal from mm10 sequence.

This module defines a simple ML model based on BPNet that
operates on one-hot encoded DNA (4 x L) and predicts a 1D signal
profile (e.g. binned CUT&RUN bigWig values) for the same window
and total counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import os

import enlighten
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from datas import SequenceBigWigDataset


class BPNetModel(nn.Module):
    """
    BPNet-style architecture for CUT&RUN prediction.
    
    Architecture:
    - Encoder: Dilated convolutional stack (exponentially increasing dilation)
    - Profile Head: Predicts binary binding profile (batch, 1, L)
    
    Input:  (batch, 4, L) - one-hot encoded DNA
    Output: profile_logits of shape (batch, 1, L) - logits for binary classification
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int = 4,
        hidden_channels: int = 64,
        n_encoder_layers: int = 9,
        kernel_size: int = 25,
        profile_kernel_size: int = 75,
        output_len: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        seq_len
            Input sequence length.
        n_channels
            Input channels (4 for one-hot DNA: A, C, G, T).
        hidden_channels
            Number of filters in encoder layers (default 64, matching BPNet).
        n_encoder_layers
            Number of dilated conv layers in encoder (default 9, matching BPNet).
        kernel_size
            Kernel size for encoder layers (default 25, matching BPNet).
        profile_kernel_size
            Kernel size for profile head (default 75, matching BPNet).
        output_len
            Output profile length. If None, uses seq_len.
        """
        super().__init__()

        self.seq_len = seq_len
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.n_encoder_layers = n_encoder_layers
        self.kernel_size = kernel_size
        self.profile_kernel_size = profile_kernel_size

        if output_len is None:
            output_len = seq_len

        # --- Encoder: Dilated Convolutional Stack ---
        # Exponentially increasing dilation: 1, 2, 4, 8, 16, 32, 64, 128, 256
        encoder_layers = []
        in_ch = n_channels
        
        for i in range(n_encoder_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32, 64, 128, 256, ...
            padding = (kernel_size - 1) * dilation // 2  # "same" padding for dilated conv
            
            conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )
            encoder_layers.append(conv)
            encoder_layers.append(nn.BatchNorm1d(hidden_channels))
            encoder_layers.append(nn.ReLU())
            in_ch = hidden_channels
        
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Profile Head: Predicts WHERE binding occurs ---
        profile_padding = (profile_kernel_size - 1) // 2
        self.profile_head = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=1, 
            kernel_size=profile_kernel_size,
            padding=profile_padding,
        )

        self.output_len = output_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Tensor of shape (batch, 4, L).

        Returns
        -------
        profile_logits
            Tensor of shape (batch, 1, L_out) - logits for binary classification.
        """
        # Encoder
        h = self.encoder(x)  # (batch, hidden_channels, L)

        # Profile head: predicts binding profile
        profile_logits = self.profile_head(h)  # (batch, 1, L) - logits

        # Optionally crop/interpolate profile if output_len != seq_len.
        # This is needed when the sequence is long (e.g. 10kb) but the
        # target profile is binned (e.g. 100 bins). We match the last
        # dimension of profile_logits to the target length.
        if profile_logits.shape[-1] != self.output_len:
            profile_logits = nn.functional.interpolate(
                profile_logits,
                size=self.output_len,
                mode="linear",
                align_corners=False,
            )

        return profile_logits


class BPNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.criterion(y_pred.squeeze(), y_true.squeeze())


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 10
    patience: int = 5
    device: str = (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def train_cnn_regressor(
    dataset: SequenceBigWigDataset,
    batch_size: int = 32,
    config: Optional[TrainConfig] = None,
    num_workers: int = 0,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_seed: int = 42,
) -> BPNetModel:
    """
    Training loop for BPNetModel (BPNet-style architecture) on SequenceBigWigDataset.

    The model uses:
    - Dilated convolutional encoder (exponentially increasing dilation)
    - Separate profile head (predicts binding profile) and count head (predicts total counts)
    - BPNetLoss combining multinomial NLL for profile and MSE for counts

    This assumes that each dataset item returns (x, y) where
    - x has shape (4, L) - one-hot encoded DNA sequence
    - y has shape (L,) or (T,) - binary CUT&RUN signal (0s and 1s)
    """
    if config is None:
        config = TrainConfig()

    # --- Split into train/val/test ---
    if not (0.0 < train_frac <= 1.0 and 0.0 <= val_frac <= 1.0 and 0.0 <= test_frac <= 1.0):
        raise ValueError("train_frac, val_frac, and test_frac must be between 0 and 1.")
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

    g = torch.Generator()
    g.manual_seed(split_seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  # shuffles order every epoch
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Peek at one batch to infer sequence and target lengths
    x0, y0 = next(iter(train_loader))
    seq_len = x0.shape[-1]
    output_len = y0.shape[-1]

    model = BPNetModel(seq_len=seq_len, output_len=output_len)
    model.to(config.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    criterion = BPNetLoss()

    # Expose loaders and datasets on the model so that downstream
    # notebooks can reuse the exact same splits and sampling logic.
    model.train_dataset = train_ds
    model.val_dataset = val_ds
    model.test_dataset = test_ds
    model.train_loader = train_loader
    model.val_loader = val_loader
    model.test_loader = test_loader

    # Track losses for all epochs
    train_losses = []
    val_losses = []

    # Early stopping state
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    manager = enlighten.get_manager()
    model.train()
    for epoch in range(config.epochs):
        # ---- Training ----
        epoch_train_loss = 0.0
        train_batches = 0
        pbar = manager.counter(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{config.epochs} | train",
            unit="batch",
            leave=False,
        )
        for x, y in train_loader:
            x = x.to(config.device)
            y = y.to(config.device)

            optimizer.zero_grad()
            profile_logits = model(x)  # (batch, 1, L)

            loss = criterion(profile_logits, y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_batches += 1

            pbar.update()
        pbar.close()

        avg_train_loss = epoch_train_loss / max(train_batches, 1)

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(config.device)
                y = y.to(config.device)
                profile_logits = model(x)
                loss = criterion(profile_logits, y)
                val_loss_sum += loss.item()
                val_batches += 1
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        model.train()

        # Record history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        summary_bar = manager.counter(
            total=1,
            desc=(
                f"Epoch {epoch + 1}/{config.epochs} "
                f"| train: {avg_train_loss:.4f}, val: {avg_val_loss:.4f}"
            ),
            unit="epoch",
            leave=False,
        )
        summary_bar.update()
        summary_bar.close()

        print(
            f"Epoch {epoch + 1}/{config.epochs} "
            f"- train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}"
        )

        if epochs_no_improve >= config.patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"(no val improvement for {config.patience} epochs)."
            )
            break

    # Restore best model weights if we saw an improvement
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ---- Final evaluation on train/val/test ----
    def evaluate(loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(config.device)
                y = y.to(config.device)
                profile_logits = model(x)
                loss = criterion(profile_logits, y)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    final_train_loss = evaluate(train_loader)
    final_val_loss = evaluate(val_loader)
    final_test_loss = evaluate(test_loader)

    print("\nFinal losses:")
    print(f"  Train: {final_train_loss:.4f}")
    print(f"  Val:   {final_val_loss:.4f}")
    print(f"  Test:  {final_test_loss:.4f}")

    # Attach full history to the model for later inspection/saving
    model.history = {
        "train": train_losses,
        "val": val_losses,
        "final": {
            "train": final_train_loss,
            "val": final_val_loss,
            "test": final_test_loss,
        },
    }

    return model


def visualize_split_predictions(
    model: nn.Module,
    device: str = "cpu",
    n_examples_per_split: int = 3,
    loader: Optional[DataLoader] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize model predictions vs true signal for a few examples from
    the train/val/test loaders used during training.

    Requires that the model was created by train_cnn_regressor so that
    it has train_loader, val_loader, and test_loader attributes.
    """
    import matplotlib.pyplot as plt

    model.eval()
    model.to(device)

    if loader is None:
        loaders = {
            "train": getattr(model, "train_loader", None),
            "val": getattr(model, "val_loader", None),
            "test": getattr(model, "test_loader", None),
        }
    else:
        loaders = {
            "all": loader,
        }

    for split_name, loader in loaders.items():
        if loader is None:
            continue

        n_show = min(n_examples_per_split, len(loader.dataset))
        if n_show == 0:
            continue

        fig, axes = plt.subplots(n_show, 1, figsize=(10, 3 * n_show), squeeze=False)

        it = iter(loader)
        for i in range(n_show):
            if type(model) == BPNetModel:
                x, y_true = next(it)
            else:
                x, x_add, y_true = next(it)
                x_add = x_add.to(device)
                x_add = x_add[0:1]
            
            x = x.to(device)
            y_true = y_true.to(device)

            # just take the first example
            x_single = x[0:1, :, :]      # Keep batch dimension: (1, 4, L)
            y_true_single = y_true[0, :] # (L,)

            with torch.no_grad():
                if type(model) == BPNetModel:
                    profile_logits = model(x_single)  # (1, 1, L)
                else:
                    profile_logits = model(x_single, x_add)  # (1, 1, L)

            # For binary classification, use sigmoid to get probabilities
            y_pred = torch.sigmoid(profile_logits.squeeze())  # (L,)

            ax = axes[i, 0]
            ax.plot(y_true_single.cpu().numpy(), label="true", alpha=0.7)
            ax.plot(y_pred.cpu().numpy(), label="pred", alpha=0.7)
            ax.set_title(f"{split_name} example {i+1}")
            ax.set_xlabel("Bins")
            ax.set_ylabel("Signal")
            ax.legend()

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(f'{save_path}_{split_name}.png')
            plt.close()
        else:
            plt.show()


def save_model(model: BPNetModel, path: Union[str, "os.PathLike[str]"]) -> None:
    """
    Save a BPNetModel in a way that captures both its architecture
    configuration and its learned parameters.

    The saved checkpoint is a dictionary with:
    - "model_class": class name (currently "BPNetModel")
    - "model_kwargs": keyword arguments needed to reconstruct the model
    - "state_dict": the model's state_dict
    - "history": optional training history attached to the model
    """
    from pathlib import Path

    p = Path(path)

    model_kwargs = {
        "seq_len": model.seq_len,
        "n_channels": model.n_channels,
        "hidden_channels": model.hidden_channels,
        "n_encoder_layers": model.n_encoder_layers,
        "kernel_size": model.kernel_size,
        "profile_kernel_size": model.profile_kernel_size,
        "output_len": model.output_len,
    }

    checkpoint = {
        "model_class": model.__class__.__name__,
        "model_kwargs": model_kwargs,
        "state_dict": model.state_dict(),
        "history": getattr(model, "history", None),
    }

    torch.save(checkpoint, p)


def load_model(path: Union[str, "os.PathLike[str]"], device: Optional[str] = None) -> BPNetModel:
    """
    Load a BPNetModel saved with save_model().

    This also supports older checkpoints that only contain a raw
    state_dict (in that case you must construct the model yourself
    before calling load_state_dict).
    """
    from pathlib import Path

    p = Path(path)
    if device is None:
        map_location = None
    else:
        map_location = device

    checkpoint = torch.load(p, map_location=map_location)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "model_kwargs" in checkpoint:
        model = BPNetModel(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        if checkpoint.get("history") is not None:
            model.history = checkpoint["history"]
        if device is not None:
            model.to(device)
        return model

    # Fallback: assume this is a bare state_dict; caller must load it manually.
    raise ValueError(
        "Checkpoint does not contain model configuration. "
        "It looks like a raw state_dict; construct BPNetModel manually "
        "and call load_state_dict on it."
    )


__all__ = [
    "BPNetModel",
    "BPNetLoss",
    "TrainConfig",
    "train_cnn_regressor",
    "visualize_split_predictions",
    "save_model",
    "load_model",
]

