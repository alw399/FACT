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

import torch
from torch import nn
import torch.nn.functional as F


def compute_receptive_field(model: nn.Module, seq_len: int) -> tuple:
    """
    Compute the effective receptive field of a convolutional model.

    Args:
        model: PyTorch convolutional model
        seq_len: Input sequence length

    Returns:
        Tuple of (receptive_field, padding)
    """
    rf = 1
    pad = 0
    
    # Use .modules() to recursively find all layers
    for layer in model.modules():
        if isinstance(layer, nn.Conv1d):
            k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
            d = layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation
            rf = rf + (k - 1) * d
            pad = pad + (k - 1) * d // 2
        elif isinstance(layer, nn.MaxPool1d):
            k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
            rf = rf + (k - 1)
            pad = pad + (k - 1) // 2

    if seq_len < rf:
        print(f"WARNING: Sequence length ({seq_len}) < receptive field ({rf}). Edge effects may dominate.")
    else:
        print(f"Receptive field: {rf} bp (Sequence length: {seq_len} bp)")

    return rf, pad


class ResidualConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, 
                              dilation=dilation, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Additive residual connection is best for DeepLIFT/MoDISco
        return x + self.relu(self.conv(x))


class BPNetModel(nn.Module):
    def __init__(
        self, 
        seq_len: int, 
        n_channels: int = 4, 
        hidden_channels: int = 128, 
        n_encoder_layers: int = 8, 
        kernel_size: int = 3,
        profile_kernel_size: int = 25,
        sidelines: int = 0,
        min_profile: int = 0
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.n_encoder_layers = n_encoder_layers
        self.kernel_size = kernel_size
        self.profile_kernel_size = profile_kernel_size
        self.sidelines = sidelines
        self.min_profile = min_profile
        
        # 1. Initial Conv Layer
        padding = (kernel_size - 1) // 2
        stem = nn.Sequential(
            nn.Conv1d(n_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU()
        )

        # 2. Dilated Residual Body
        body = nn.Sequential(*[
            ResidualConv1d(hidden_channels, kernel_size, dilation=2**i)
            for i in range(1, n_encoder_layers)
        ])
        
        self.encoder = nn.Sequential(stem, body)

        # 3. Profile Head: High-resolution output (Batch, 1, SeqLen)
        self.profile_head = nn.Conv1d(
            hidden_channels, 1, kernel_size=profile_kernel_size, padding=(profile_kernel_size - 1) // 2
        )

        # 4. Count Head: Scalar output (Batch, 1)
        self.counts_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for convolutional and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.encoder(x)
        
        profile_logits = self.profile_head(h)
        count_logits = self.counts_head(h)

        return profile_logits, count_logits
        

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
            "n_channels": model.n_channels,
            "hidden_channels": model.hidden_channels,
            "n_encoder_layers": model.n_encoder_layers,
            "kernel_size": model.kernel_size,
            "profile_kernel_size": model.profile_kernel_size,
            "sidelines": getattr(model, "sidelines", 0),
            "min_profile": getattr(model, "min_profile", 0),
        }
        checkpoint = {
            "model_class": class_name,
            "model_kwargs": model_kwargs,
            "state_dict": model.state_dict(),
            "min_profile": model.min_profile,
            "history": getattr(model, "history", None),
        }
        torch.save(checkpoint, p)
    
    @classmethod
    def load_model(cls, path: Union[str, "os.PathLike[str]"], device: Optional[str] = None) -> "BPNetModel":
        p = Path(path)
        # map_location handles device placement during the initial load
        checkpoint = torch.load(p, map_location=device, weights_only=False)

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
        model.min_profile = checkpoint.get("min_profile")
        model.history = checkpoint.get("history")

        if device is not None:
            model.to(device)
            
        return model


class BPNetK4Model(BPNetModel):
    """
    BPNet-like model with an extra 1-channel signal input.

    Inputs:
      - sequence: (batch, 4, L_seq)
      - sequence_k4: (batch, 1, L_seq)

    The additional signal is concatenated to the sequence channels => (batch, 5, L_seq).

    Outputs:
      - profile_logits: (batch, 1, L_seq)
      - total_counts: (batch, 1)
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int = 5,
        hidden_channels: int = 128,
        n_encoder_layers: int = 8,
        kernel_size: int = 3,
        profile_kernel_size: int = 25,
        sidelines: int = 0,
        min_profile: int = 0
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            n_channels=n_channels,
            hidden_channels=hidden_channels,
            n_encoder_layers=n_encoder_layers,
            kernel_size=kernel_size,
            profile_kernel_size=profile_kernel_size,
            sidelines=sidelines,
            min_profile=min_profile
        )

    def forward(self, sequence: torch.Tensor, sequence_k4: torch.Tensor) -> torch.Tensor:
        # Ensure sequence_k4 has shape (B, 1, L_sig)
        if sequence_k4.dim() == 2:
            sequence_k4 = sequence_k4.unsqueeze(1)

        # 1. Encoder (DNA + K4)
        x = torch.cat([sequence, sequence_k4], dim=1)  # (B, 5, L_seq)
        h = self.encoder(x) 

        # 2. Heads
        profile_logits = self.profile_head(h)  # (B, 1, L)
        total_counts = self.counts_head(h).squeeze(-1) # (B, 1)

        return profile_logits, total_counts



class BPNetLoss(nn.Module):
    def __init__(self, loss_type: str = 'bpnet', min_profile=0, sidelines=0, target_smoothing=False, smoothing_sigma=3.0):
        super().__init__()
        self.loss_type = loss_type
        self.counts_loss = nn.MSELoss(reduction='sum')
        self.min_profile = min_profile
        self.sidelines = sidelines
        self.target_smoothing = target_smoothing
        self.smoothing_sigma = smoothing_sigma

        if self.target_smoothing:
            kernel = self.get_gaussian_kernel(self.smoothing_sigma)
            self.register_buffer('smoothing_kernel', kernel)
            self.smoothing_padding = kernel.shape[-1] // 2

    @staticmethod
    def get_gaussian_kernel(sigma: float) -> torch.Tensor:
        """Creates a 1D Gaussian kernel as a 3D tensor (1, 1, K)."""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0: kernel_size += 1
        x = torch.arange(kernel_size).float()
        center = kernel_size // 2
        kernel = torch.exp(-0.5 * ((x - center) / sigma)**2)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, -1)

    @staticmethod
    def smooth_signal(signal: torch.Tensor, sigma: float = None, kernel: torch.Tensor = None) -> torch.Tensor:
        """Applies Gaussian smoothing to a 1D signal (B, L) or (B, 1, L)."""
        if kernel is None:
            if sigma is None:
                raise ValueError("Either sigma or kernel must be provided for smoothing.")
            kernel = BPNetLoss.get_gaussian_kernel(sigma).to(signal.device)
        
        # Determine if we need to add a channel dimension
        # (Batch, Length) -> (Batch, 1, Length)
        is_2d = (signal.dim() == 2)
        if is_2d:
            signal = signal.unsqueeze(1)
            
        pad = kernel.shape[-1] // 2
        smoothed = F.conv1d(signal, kernel, padding=pad)
        
        return smoothed.squeeze(1) if is_2d else smoothed

    def compute_counts_loss(self, counts_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculates the weighted MSE counts loss."""
        # 1. Slice sidelines to match the active region
        if self.sidelines > 0:
            y_true = y_true[:, self.sidelines:-self.sidelines]
        
        target_counts = y_true.sum(dim=-1)
        log_true_counts = torch.log1p(target_counts)
        pred_log_counts = counts_pred.flatten()

        # 2. Weighted MSE
        counts_loss = F.mse_loss(pred_log_counts, log_true_counts, reduction='none')
        counts_weight = (target_counts / 2) * 0.1
        return (counts_weight * counts_loss).sum()

    def compute_profile_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculates the Multinomial NLL profile loss."""
        # 1. Slicing and Dimensions
        if y_pred.dim() == 3:
            y_pred = y_pred.squeeze(1)
        if self.sidelines > 0:
            y_true = y_true[:, self.sidelines:-self.sidelines]
            y_pred = y_pred[:, self.sidelines:-self.sidelines]

        target_counts = y_true.sum(dim=-1)
        
        # 2. Filter low-signal samples
        good_profiles = target_counts > self.min_profile
        if not good_profiles.any():
            return torch.tensor(0.0, device=y_true.device)

        y_p_good = y_pred[good_profiles]
        y_t_good = y_true[good_profiles]

        # 3. Smoothing
        if self.target_smoothing:
            y_t_good = self.smooth_signal(y_t_good, kernel=self.smoothing_kernel)

        # 4. Multinomial NLL
        if self.loss_type == 'bpnet':
            log_probs = F.log_softmax(y_p_good, dim=-1)
            return -(y_t_good * log_probs).sum()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def forward(self, y_pred: torch.Tensor, counts_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates total loss normalized by batch size.
        """
        batch_size = y_true.shape[0]
        c_loss = self.compute_counts_loss(counts_pred, y_true)
        p_loss = self.compute_profile_loss(y_pred, y_true)
        
        return (c_loss + p_loss) / batch_size



class BPNetClusterModel(BPNetModel):
    def __init__(
        self, 
        seq_len: int, 
        n_channels: int = 4, 
        hidden_channels: int = 128, 
        n_encoder_layers: int = 8, 
        kernel_size: int = 3,
        profile_kernel_size: int = 25,
        sidelines: int = 0,
        min_profile: int = 0,
    ):

        super(BPNetClusterModel, self).__init__(
            seq_len=seq_len,
            n_channels=n_channels,
            hidden_channels=hidden_channels,
            n_encoder_layers=n_encoder_layers,
            kernel_size=kernel_size,
            profile_kernel_size=profile_kernel_size,
            sidelines=sidelines,
            min_profile=min_profile,
        )

    def forward(self, x):
        h = self.encoder(x)
        
        profile_logits = self.profile_head(h)
        label = self.counts_head(h)

        return profile_logits, label
    
class ClassifierLoss(BPNetLoss):
    def __init__(self, min_profile=0, sidelines=0, target_smoothing=False, smoothing_sigma=3.0):
        super(ClassifierLoss, self).__init__(
            loss_type='bpnet',
            min_profile=min_profile,
            sidelines=sidelines,
            target_smoothing=target_smoothing,
            smoothing_sigma=smoothing_sigma
        )

    def compute_class_loss(self, label_pred: torch.Tensor, label_true: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(label_pred, label_true, reduction='none').sum()

    def forward(self, y_pred: torch.Tensor, label_pred: torch.Tensor, y_true: torch.Tensor, label_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates total loss normalized by batch size.
        """
        batch_size = y_true.shape[0]
        c_loss = self.compute_class_loss(label_pred, label_true)
        # p_loss = self.compute_profile_loss(y_pred, y_true)
        
        return c_loss / batch_size


__all__ = [
    "BPNetModel",
    "BPNetK4Model",
    "BPNetClusterModel",
    "BPNetLoss",
    "ClassifierLoss",
    "compute_receptive_field"
]

