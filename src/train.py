from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

from cnn import BPNetModel, BPNetK4Model, BPNetLoss
from datas import SequenceBigWigDataset, SequenceDualBigWigDataset

from tqdm.auto import tqdm
import enlighten
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon



@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 10
    patience: int = 5
    loss_type: str = 'bpnet',
    min_profile: int = 6000,
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
    testing_mode: bool = False,
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
    *x0, y0 = next(iter(train_loader))
    seq_len = x0[0].shape[-1]
    output_len = y0.shape[-1]

    # Initialize model
    if isinstance(dataset, SequenceBigWigDataset):
        model = BPNetModel(seq_len=seq_len, n_channels=4, output_len=output_len)
    elif isinstance(dataset, SequenceDualBigWigDataset):
        model = BPNetK4Model(seq_len=seq_len, n_channels=5, output_len=output_len)
    else:
        raise ValueError(f"Dataset type {type(dataset)} not supported")
    model.to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    criterion = BPNetLoss(
        loss_type=config.loss_type, 
        min_profile=config.min_profile
    )

    # Expose loaders and datasets on the model so that downstream
    # notebooks can reuse the exact same splits and sampling logic.
    model.train_dataset = train_ds
    model.val_dataset = val_ds
    model.test_dataset = test_ds
    model.train_loader = train_loader
    model.val_loader = val_loader
    model.test_loader = test_loader
    model.min_profile = config.min_profile # save so we can keep track later

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
        for batch_data in train_loader:
            *x_batches, y = batch_data
            x_batches = [x.to(config.device) for x in x_batches]
            y = y.to(config.device)

            optimizer.zero_grad()
            profile_logits, pred_counts = model(*x_batches)  # (batch, 1, L)

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
            for batch_data in val_loader:
                *x_batches, y = batch_data
                x_batches = [x.to(config.device) for x in x_batches]
                y = y.to(config.device)
                profile_logits, pred_counts = model(*x_batches)
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
            import copy
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        summary_bar = manager.counter(
            total=1,
            desc=(
                f"Epoch {epoch + 1}/{config.epochs} "
                f"| train: {avg_train_loss:.0f}, val: {avg_val_loss:.0f}"
            ),
            unit="epoch",
            leave=False,
        )
        summary_bar.update()
        summary_bar.close()

        if epoch % 1 == 0:
            print(
                f"Epoch {epoch + 1}/{config.epochs} "
                f"- train loss: {avg_train_loss:.0f}, val loss: {avg_val_loss:.0f}"
            )

        if epochs_no_improve >= config.patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"(no val improvement for {config.patience} epochs)."
            )
            break

    # Restore best model weights if we saw an improvement
    if best_state_dict is not None and not testing_mode:
        model.load_state_dict(best_state_dict)

    # ---- Final evaluation on train/val/test ----
    def evaluate(loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch_data in loader:
                *x_batches, y = batch_data
                x_batches = [x.to(config.device) for x in x_batches]
                y = y.to(config.device)
                profile_logits, pred_counts = model(*x_batches)
                loss = criterion(profile_logits, pred_counts, y)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    final_train_loss = evaluate(train_loader)
    final_val_loss = evaluate(val_loader)
    final_test_loss = evaluate(test_loader)

    print("\nFinal losses:")
    print(f"  Train: {final_train_loss:.0f}")
    print(f"  Val:   {final_val_loss:.0f}")
    print(f"  Test:  {final_test_loss:.0f}")

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
        loader = model.test_loader
    else:
        loaders = {"mixed": loader}

    # take a peak to get the right dimensions
    *x, y = next(iter(loader))
    S, G = x[0].shape[1:]
    N = y.shape[0]

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

                    # Distribution-based visualization (multinomial/kldiv)
                    y_pred = F.softmax(profile_logits.squeeze(), dim=-1)
                    y_true_norm = y_true_single / (y_true_single.sum() + 1e-8)

                    row, col = divmod(count, n_cols_actual)
                    ax = axes[row, col]
                    ax.plot(y_true_norm.cpu().numpy(), label="true", alpha=0.7)
                    ax.plot(y_pred.cpu().numpy(), label="pred", alpha=0.7)
                    
                    true_total = y_true_single.sum().item()
                    # The model outputs log(1 + counts), so we invert it for the visualization
                    pred_total = torch.expm1(pred_counts).item()
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

@torch.no_grad()
def evaluate_model_metrics(model, loader, device="cpu", min_profile=0):
    model.eval()
    model.to(device)
    all_true_log_counts, all_pred_log_counts, all_jsds = [], [], []

    with torch.no_grad():
        for batch_data in loader:
            *x_batches, y_true_batch = batch_data
            x_batches = [x.to(device) for x in x_batches]
            y_true_batch = y_true_batch.to(device)

            profile_logits, pred_counts = model(*x_batches)

            mask = (y_true_batch != -1)
            target_counts = (y_true_batch * mask.float()).sum(dim=-1)

            all_pred_log_counts.extend(pred_counts.squeeze(-1).cpu().numpy().tolist())
            all_true_log_counts.extend(torch.log1p(target_counts).cpu().numpy().tolist())

            y_pred_masked = torch.where(mask, profile_logits.squeeze(1), torch.full_like(profile_logits.squeeze(1), -1e9))
            y_pred_prob = F.softmax(y_pred_masked, dim=-1).cpu().numpy()
            y_true_prob = ((y_true_batch * mask.float()) / (target_counts.unsqueeze(-1) + 1e-8)).cpu().numpy()

            for i in range(y_pred_prob.shape[0]):
                if target_counts[i].item() > min_profile:
                    jsd = jensenshannon(y_true_prob[i], y_pred_prob[i])
                    if not np.isnan(jsd):
                        all_jsds.append(jsd)

    return {
        "profile_jsd": all_jsds,
        "log_counts_true": all_true_log_counts,
        "log_counts_pred": all_pred_log_counts,
    }

__all__ = [
    "train_cnn_regressor",
    "visualize_split_predictions",
    "evaluate_model_metrics",
    "TrainConfig"
]