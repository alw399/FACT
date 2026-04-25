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
from pathlib import Path

import enlighten
import torch
from torch import nn
import torch.nn.functional as F
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

        kernel_size=51
        padding = (kernel_size - 1) // 2
        self.mlp = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(output_len),
            nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False)
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
        
        # Profile Head
        profile_logits = self.profile_head(h)  # (batch, 1, L)
        
        # Spatial Smoothing / Pooling
        profile_logits = self.mlp(profile_logits) # (batch, 1, L_out)
        total_counts = torch.sum(profile_logits, dim=-1)

        return profile_logits, total_counts
    

    def save_model(self, path: Union[str, "os.PathLike[str]"]) -> None:
        """
        Save a BPNetModel or BPNetK4Model in a way that captures both its architecture
        configuration and its learned parameters.

        The saved checkpoint is a dictionary with:
        - "model_class": class name ("BPNetModel" or "BPNetK4Model")
        - "model_kwargs": keyword arguments needed to reconstruct the model
        - "state_dict": the model's state_dict
        - "history": optional training history attached to the model
        """
        model = self

        p = Path(path)
        class_name = model.__class__.__name__

        model_kwargs = {
            "seq_len": model.seq_len,
            "n_channels_seq": model.n_channels_seq,
            "hidden_channels": model.hidden_channels,
            "n_encoder_layers": model.n_encoder_layers,
            "kernel_size": model.kernel_size,
            "profile_kernel_size": model.profile_kernel_size,
            "output_len": model.output_len,
        }
        checkpoint = {
            "model_class": class_name,
            "model_kwargs": model_kwargs,
            "state_dict": model.state_dict(),
            "history": getattr(model, "history", None),
            "loss_type": getattr(model, "loss_type", "kldiv"),
        }
        torch.save(checkpoint, p)
    
    @classmethod
    def load_model(cls, path: Union[str, "os.PathLike[str]"], device: Optional[str] = None) -> "BPNetModel":
        p = Path(path)
        # map_location handles device placement during the initial load
        checkpoint = torch.load(p, map_location=device)

        # Check if model_kwargs exists (for your 'self' replacement)
        if "model_kwargs" in checkpoint:
            # This calls the constructor of whichever subclass called load_model
            model = cls(**checkpoint["model_kwargs"])
        else:
            # Fallback logic for raw state_dict checkpoints
            raise ValueError(
                f"Checkpoint at {path} does not contain 'model_kwargs'. "
                "For older checkpoints, instantiate the model manually."
            )

        model.load_state_dict(checkpoint["state_dict"])
        
        # Optional attributes
        model.history = checkpoint.get("history")
        model.loss_type = checkpoint.get("loss_type")

        if device is not None:
            model.to(device)
            
        return model


class BPNetLoss(nn.Module):
    def __init__(self, loss_type: str = 'kldiv', counts_weight: float = 1e-7):
        super().__init__()
        self.loss_type = loss_type
        self.counts_weight = counts_weight
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.counts_loss = nn.L1Loss(reduction='sum')
        self.scale_counts = 200        # it's impossible for a

    def forward(self, y_pred: torch.Tensor, counts_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: (batch, 1, output_len) or (batch, output_len) - raw logits
        y_true: (batch, output_len) - signal (may contain -1 for missing)
        """
        if y_pred.dim() == 3:
            y_pred = y_pred.squeeze(1) # (batch, output_len)
        
        # 1. Create global mask for valid bins
        mask = (y_true != -1) # (batch, output_len)
        valid_samples = mask.any(dim=1) # (batch,)
        
        if not valid_samples.any():
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # 2. Counts Loss
        # target_counts is sum of valid bins only per sample
        target_counts = (y_true * mask.float()).sum(dim=-1) # (batch,)
        pred_counts = (counts_pred).flatten() # (batch,)

        if self.loss_type == 'bpnet':
            # BPNet Counts Loss: MSE of log-transformed counts
            # Paper: (log(1 + total_true_counts) - log(1 + total_pred_counts))^2
            log_true_counts = torch.log1p(target_counts[valid_samples])
            log_pred_counts = torch.log1p(pred_counts[valid_samples])
            counts_loss = F.mse_loss(log_pred_counts, log_true_counts, reduction='sum')
        else:
            # Default: L1 loss (MSE if configured in __init__) only on valid samples
            counts_loss = self.counts_loss(pred_counts[valid_samples], target_counts[valid_samples])
        
        total_loss = self.counts_weight * counts_loss

        # 3. Profile Loss
        if self.loss_type == 'kldiv':
            # target_prob: normalize valid bins to sum to 1 per sample
            target_sum = target_counts.unsqueeze(-1) + 1e-8
            target_prob = (y_true * mask.float()) / target_sum
            
            # pred_log_prob: masked log_softmax
            # Set invalid bins to -inf so they don't contribute to the softmax denominator
            y_pred_masked = torch.where(mask, y_pred, torch.full_like(y_pred, -1e9))
            pred_log_prob = F.log_softmax(y_pred_masked, dim=-1)
            
            # KL divergence calculation: target * (log(target) - pred_log_prob)
            kl_div = target_prob * (torch.log(target_prob + 1e-8) - pred_log_prob)
            # Sum up valid contributions
            total_loss += (kl_div * mask.float()).sum()
            
        elif self.loss_type == 'mse':
            # Only count loss from valid bins
            mse = F.mse_loss(y_pred * mask.float(), y_true * mask.float(), reduction='sum')
            total_loss += mse
            
        elif self.loss_type == 'bce':
            # Primary for binary signal; ignore bins with y_true == -1
            target_bin = torch.clamp(y_true, 0, 1)
            bce = F.binary_cross_entropy_with_logits(y_pred, target_bin, reduction='none')
            total_loss += (bce * mask.float()).sum()
        elif self.loss_type == 'bpnet':
            # Profile Loss: Multinomial NLL
            # Set invalid bins to -inf for log_softmax
            y_pred_masked = torch.where(mask, y_pred, torch.full_like(y_pred, -1e9))
            pred_log_prob = F.log_softmax(y_pred_masked, dim=-1)
            
            # MNLL: -sum(true_counts * log_prob)
            # Note: y_true are the raw counts
            profile_loss = -(y_true * mask.float() * pred_log_prob).sum()
            total_loss += profile_loss
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Average by the number of samples that had at least one valid bin
        return total_loss / valid_samples.float().sum()


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 10
    patience: int = 5
    loss_type: str = 'kldiv'
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    criterion = BPNetLoss(loss_type=config.loss_type)
    model.loss_type = config.loss_type

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
            profile_logits, pred_counts = model(x)  # (batch, 1, L)

            loss = criterion(profile_logits, pred_counts, y)
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
                profile_logits, pred_counts = model(x)
                loss = criterion(profile_logits, pred_counts, y)
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

        if epoch % 1 == 0:
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
                profile_logits, pred_counts = model(x)
                loss = criterion(profile_logits, pred_counts, y)
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
    n_cols: int = 1,
    loader: Optional[DataLoader] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize model predictions vs true signal for a few examples from
    the train/val/test loaders used during training.
    """
    import matplotlib.pyplot as plt
    import math

    model.eval()
    model.to(device)

    if loader is None:
        loaders = {
            "train": getattr(model, "train_loader", None),
            "val": getattr(model, "val_loader", None),
            "test": getattr(model, "test_loader", None),
        }
    else:
        loaders = {"mixed": loader}

    for split_name, loader in loaders.items():
        if loader is None:
            continue

        n_show = min(n_examples_per_split, len(loader.dataset))
        if n_show == 0:
            continue

        n_cols_actual = min(n_cols, n_show)
        n_rows = math.ceil(n_show / n_cols_actual)
        fig, axes = plt.subplots(n_rows, n_cols_actual, figsize=(5 * n_cols_actual, 3 * n_rows), squeeze=False)

        count = 0
        done = False
        with torch.no_grad():
            for batch_data in loader:
                if done:
                    break
                
                # Unpack: everything before the last item is input, the last is ground truth
                *x_batches, y_true_batch = batch_data
                x_batches = [x.to(device) for x in x_batches]
                y_true_batch = y_true_batch.to(device)

                batch_curr = y_true_batch.shape[0]
                for b in range(batch_curr):
                    if count >= n_show:
                        done = True
                        break
                    
                    x_singles = [x[b:b+1] for x in x_batches]
                    y_true_single = y_true_batch[b]
                    
                    profile_logits, pred_counts = model(*x_singles)

                    # Adjust prediction and normalization based on loss_type
                    loss_type = getattr(model, "loss_type", "kldiv")
                    if loss_type == "bce":
                        y_pred = torch.sigmoid(profile_logits.squeeze())
                        y_true_norm = torch.clamp(y_true_single, 0, 1)
                    elif loss_type == "mse":
                        # Value-based visualization for regression
                        y_pred = profile_logits.squeeze()
                        y_true_norm = y_true_single
                    elif loss_type in ["kldiv", "bpnet"]:
                        # Distribution-based visualization (multinomial/kldiv)
                        y_pred = F.softmax(profile_logits.squeeze(), dim=-1)
                        y_true_norm = y_true_single / (y_true_single.sum() + 1e-8)

                    row, col = divmod(count, n_cols_actual)
                    ax = axes[row, col]
                    ax.plot(y_true_norm.cpu().numpy(), label="true", alpha=0.7)
                    ax.plot(y_pred.cpu().numpy(), label="pred", alpha=0.7)
                    
                    true_total = y_true_single.sum().item()
                    pred_total = pred_counts.item()
                    ax.set_title(f"{split_name} {count+1}\ntrue: {true_total:.2f}, pred: {pred_total:.2f}")
                    ax.set_xlabel("Bins")
                    ax.set_ylabel("Signal")
                    # Only show legend on the first plot to save space
                    if count == 0:
                        ax.legend()
                    count += 1
        
        # Hide unused subplots
        for i in range(count, n_rows * n_cols_actual):
            row, col = divmod(i, n_cols_actual)
            axes[row, col].axis('off')

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(f'{save_path}_{split_name}.png')
            plt.close()
        else:
            plt.show()



__all__ = [
    "BPNetModel",
    "BPNetLoss",
    "TrainConfig",
    "train_cnn_regressor",
    "visualize_split_predictions"
]

