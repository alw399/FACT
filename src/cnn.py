"""
CNN-based predictor for CUT&RUN signal from mm10 sequence.

This module defines a simple 1D convolutional neural network that
operates on one-hot encoded DNA (4 x L) and predicts a 1D signal
profile (e.g. binned CUT&RUN bigWig values) for the same window.

It also includes a minimal training loop that you can customize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import enlighten
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data import SequenceBigWigDataset


class BPNetModel(nn.Module):
    """
    BPNet-style architecture for CUT&RUN prediction.
    
    Architecture:
    - Encoder: Dilated convolutional stack (exponentially increasing dilation)
    - Profile Head: Predicts binding profile (batch, 2, L) for forward/reverse strands
    - Count Head: Predicts total counts (batch, 1)
    
    Input:  (batch, 4, L) - one-hot encoded DNA
    Output: (profile_logits, counts) where:
        - profile_logits: (batch, 2, L) - logits for forward/reverse strand profiles
        - counts: (batch, 1) - predicted total counts
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
        self.normalize_profile = nn.Softmax(dim=-1)

        # --- Count Head: Predicts HOW MUCH total signal ---
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (batch, hidden, L) -> (batch, hidden, 1)
            nn.Flatten(),              # (batch, hidden)
            nn.Linear(hidden_channels, 1),  # (batch, 1)
        )

        self.seq_len = seq_len
        self.output_len = output_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x
            Tensor of shape (batch, 4, L).

        Returns
        -------
        profile_logits
            Tensor of shape (batch, 2, L_out) - logits for forward/reverse profiles.
        counts
            Tensor of shape (batch, 1) - predicted total counts.
        """
        # Encoder
        h = self.encoder(x)  # (batch, hidden_channels, L)

        # Profile head: predicts binding profile
        profile_logits = self.profile_head(h)  # (batch, 1, L)
        profile_logits = self.normalize_profile(profile_logits)

        # Count head: predicts total counts
        counts = self.count_head(h)  # (batch, 1)

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

        return profile_logits, counts


class BPNetLoss(nn.Module):
    """
    BPNet-style loss combining:
    1. Multinomial Negative Log-Likelihood for profile (distribution over positions)
    2. MSE for total counts
    
    Total_Loss = Loss_profile + lambda * Loss_counts
    """
    
    def __init__(self, count_loss_weight: float = 1.0, eps: float = 1e-8):
        """
        Parameters
        ----------
        count_loss_weight
            Lambda weight for the count MSE loss (default 1.0, often 1-10).
        eps
            Small epsilon for numerical stability in log.
        """
        super().__init__()
        self.count_loss_weight = count_loss_weight
        self.eps = eps
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                counts_pred: torch.Tensor, counts_true: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred
            Predicted signal profile (logits), shape (batch, L) or (batch, 1, L).
        y_true
            Observed signal profile (counts), shape (batch, L) or (batch, 1, L).
        counts_pred
            Predicted total counts from separate head, shape (batch, 1).
        counts_true
            Observed total counts, shape (batch, 1).
        
        Returns
        -------
        Combined loss (scalar tensor).
        """
        # Ensure 2D: (batch, L)
        if y_pred.dim() == 3:
            y_pred = y_pred.squeeze(1)
        if y_true.dim() == 3:
            y_true = y_true.squeeze(1)
        
        # --- Profile loss: Multinomial NLL ---
        log_pred_prob = torch.log_softmax(y_pred, dim=-1)  # (batch, L)
        
        # Multinomial NLL: -sum(observed_counts * log(predicted_probabilities))
        profile_loss = -(y_true * log_pred_prob).sum(dim=-1)  # (batch,)
        profile_loss = profile_loss.mean()  # Average over batch
        
        # --- Count loss: MSE on *log*-counts (to match BPNet) ---
        #
        # In BPNet, the counts head predicts log-counts and the loss is
        # typically MSE in log space (or Poisson on log-counts). Here we
        # mirror the MSE-on-log-counts behavior.
        #
        y_true_total = counts_true.squeeze(-1)
        y_log_true = torch.log(torch.clamp(y_true_total, min=self.eps))
        y_log_pred = counts_pred.squeeze(-1)
        count_loss = torch.mean((y_log_true - y_log_pred) ** 2)
        # Combined loss
        total_loss = profile_loss + self.count_loss_weight * count_loss
        
        return total_loss


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 10
    patience: int = 5
    count_loss_weight: float = 0.1  # Lambda weight for count MSE loss in BPNetLoss (normalized, so smaller default)
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

    This assumes that each dataset item returns (x, y, y_count) where
    - x has shape (4, L) - one-hot encoded DNA sequence
    - y has shape (L,) or (T,) - normalized CUT&RUN profile (sums to 1 when >0)
    - y_count is a scalar total count for that window
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
    x0, y0, y0_count = next(iter(train_loader))
    seq_len = x0.shape[-1]
    output_len = y0.shape[-1]

    model = BPNetModel(seq_len=seq_len, output_len=output_len)
    model.to(config.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = BPNetLoss(count_loss_weight=config.count_loss_weight)

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
        for x, y, y_count in train_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            y_count = y_count.to(config.device)

            optimizer.zero_grad()
            profile_logits, counts_pred = model(x)  # (batch, 2, L), (batch, 1)

            loss = criterion(
                profile_logits,
                y,
                counts_pred=counts_pred,
                counts_true=y_count,
            )
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
            for x, y, y_count in val_loader:
                x = x.to(config.device)
                y = y.to(config.device)
                y_count = y_count.to(config.device)
                profile_logits, counts_pred = model(x)
                loss = criterion(
                    profile_logits,
                    y,
                    counts_pred=counts_pred,
                    counts_true=y_count,
                )
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
            for x, y, y_count in loader:
                x = x.to(config.device)
                y = y.to(config.device)
                y_count = y_count.to(config.device)
                profile_logits, counts_pred = model(x)
                profile_logits_summed = profile_logits.sum(dim=1)  # (batch, L)
                counts_true = y_count.unsqueeze(-1)                # (batch, 1)
                y_counts = y * counts_true                         # (batch, L)
                loss = criterion(
                    profile_logits_summed,
                    y_counts,
                    counts_pred=counts_pred,
                    counts_true=counts_true,
                )
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

    loaders = {
        "train": getattr(model, "train_loader", None),
        "val": getattr(model, "val_loader", None),
        "test": getattr(model, "test_loader", None),
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
            x, y_true, y_count = next(it)
            x = x.to(device)
            y_true = y_true.to(device)
            y_count = y_count.to(device)

            # just take the first example
            x_single = x[0:1, :, :]      # Keep batch dimension: (1, 4, L)
            y_true_single = y_true[0, :] # (L,)
            y_count_single = y_count[0]  # scalar

            with torch.no_grad():
                profile_logits, counts_pred = model(x_single)  # (1, 2, L), (1, 1)
                # Sum forward/reverse profiles for CUT&RUN
                y_pred = profile_logits.sum(dim=1).squeeze(0)  # (L,)
            # Plot normalized ground truth profile (y_true_single) and prediction
            y_true_plot = y_true_single

            ax = axes[i, 0]
            ax.plot(y_true_plot.cpu().numpy(), label="true (normalized)", alpha=0.7)
            ax.plot(y_pred.cpu().numpy(), label="pred", alpha=0.7)
            ax.set_title(f"{split_name} example {i+1} (total counts ~ {float(y_count_single):.1f}, predicted ~ {float(counts_pred.item()):.1f})")
            ax.set_xlabel("Bins")
            ax.set_ylabel("Signal")
            ax.legend()

        plt.tight_layout()
        plt.show()


__all__ = [
    "BPNetModel",
    "BPNetLoss",
    "TrainConfig",
    "train_cnn_regressor",
    "visualize_split_predictions",
]

