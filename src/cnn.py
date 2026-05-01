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


class ResidualConv1d(nn.Module):
    """Residual block with dilated 1D convolution."""
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv(x)
        res = self.bn(res)
        res = self.relu(res)
        return x + res


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
        self.output_len = output_len
        

        # --- Encoder: Dilated Convolutional Stack ---
        # Exponentially increasing dilation: 1, 2, 4, 8, 16, 32, 64, 128, 256
        encoder_layers = []
        in_ch = n_channels
        
        for i in range(n_encoder_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32, 64, 128, 256, ...
            padding = (kernel_size - 1) * dilation // 2  # "same" padding for dilated conv
            
            if in_ch != hidden_channels:
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
            else:
                encoder_layers.append(ResidualConv1d(hidden_channels, kernel_size, dilation))
                
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

        # --- Count Head: Predicts total read counts ---

        self.counts_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Tensor of shape (batch, 4, L).

        Return
        -------
        profile_logits
            Tensor of shape (batch, 1, L_out). Predicted binding profile
        total_counts
            Tensor of shape (batch,). Predicted log(1+counts) in the window
        """
        # Encoder
        h = self.encoder(x)  # (batch, hidden_channels, L)
        
        # Profile
        profile_logits = self.profile_head(h)  # (batch, 1, L)
        profile_logits = self.mlp(profile_logits) # (batch, 1, L_out)

        # Counts 
        total_counts = self.counts_head(h).squeeze(-1)

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
            "n_channels": model.n_channels,
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
        n_channels: int = 5,
        hidden_channels: int = 64,
        n_encoder_layers: int = 9,
        kernel_size: int = 25,
        profile_kernel_size: int = 75,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            n_channels=n_channels,
            hidden_channels=hidden_channels,
            n_encoder_layers=n_encoder_layers,
            kernel_size=kernel_size,
            profile_kernel_size=profile_kernel_size,
            output_len=output_len,
        )

    def forward(self, sequence: torch.Tensor, sequence_k4: torch.Tensor) -> torch.Tensor:
        # Ensure additional has shape (B, 1, L_sig)
        if sequence_k4.dim() == 2:
            sequence_k4 = sequence_k4.unsqueeze(1)

        x = torch.cat([sequence, sequence_k4], dim=1)  # (B, 5, L_seq)

        h = self.encoder(x)  # (B, hidden, L_seq)

        profile_logits = self.profile_head(h)  # (batch, 1, L)
        profile_logits = self.mlp(profile_logits) # (batch, 1, L_out)
        total_counts = self.counts_head(h).squeeze(-1)

        return profile_logits, total_counts
    

class BPNetLoss(nn.Module):
    def __init__(self, loss_type: str = 'bpnet', counts_weight: float = 1e2, min_profile=0):
        super().__init__()
        self.loss_type = loss_type
        self.counts_weight = counts_weight
        self.counts_loss = nn.MSELoss(reduction='sum')
        self.min_profile = min_profile

    def forward(self, y_pred: torch.Tensor, counts_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: (batch, 1, output_len) or (batch, output_len) - raw logits
        counts_pred: (batch,) - predicted raw counts in the window
        y_true: (batch, output_len) - raw signal profile (may contain -1 for missing)
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
        pred_log_counts = (counts_pred).flatten() # (batch,)

        # BPNet trains the model to predict log(1 + counts) directly.
        # target_counts are raw counts, so we log-transform them.
        log_true_counts = torch.log1p(target_counts[valid_samples])
        counts_loss = F.mse_loss(pred_log_counts[valid_samples], log_true_counts, reduction='sum')

        total_loss = self.counts_weight * counts_loss

        # 3. Profile Loss
        # filter the samples with total counts < profile_min because it's probably just noise
        good_profiles = target_counts > self.min_profile
        y_pred = y_pred[good_profiles]
        y_true = y_true[good_profiles]
        mask = mask[good_profiles]
        
        if self.loss_type == 'kldiv':
            # target_prob: normalize valid bins to sum to 1 per sample
            target_sum = target_counts[good_profiles].unsqueeze(-1) + 1e-8
            target_prob = (y_true * mask.float()) / target_sum
            
            # pred_log_prob: masked log_softmax
            # Set invalid bins to -inf so they don't contribute to the softmax denominator
            y_pred_masked = torch.where(mask, y_pred, torch.full_like(y_pred, -1e9))
            pred_log_prob = F.log_softmax(y_pred_masked, dim=-1)
            
            # KL divergence calculation: target * (log(target) - pred_log_prob)
            kl_div = target_prob * (torch.log(target_prob + 1e-8) - pred_log_prob)
            # Sum up valid contributions
            total_loss += (kl_div * mask.float()).sum()
            
        elif self.loss_type == 'bpnet':
            # Profile Loss: Multinomial NLL
            # Set invalid bins to -inf for log_softmax
            y_pred_masked = torch.where(mask, y_pred, torch.full_like(y_pred, -1e9))
            pred_log_prob = F.log_softmax(y_pred_masked, dim=-1)
            
            # MNLL: -sum(true_counts * log_prob)
            profile_loss = -(y_true * mask.float() * pred_log_prob).sum()
            total_loss += profile_loss

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Average by the number of samples that had at least one valid bin
        return total_loss / valid_samples.float().sum()



__all__ = [
    "BPNetModel",
    "BPNetK4Model",
    "BPNetLoss",
]

