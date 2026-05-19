from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.optim as optim

from cnn import BPNetModel, BPNetK4Model, BPNetClusterModel, BPNetLoss, ClassifierLoss, compute_receptive_field
from datas import SequenceBigWigDataset, SequenceDualBigWigDataset, SequenceBigWigPeaksDataset, SequenceBigWigClusterDataset

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
    loss_type: str = 'bpnet'
    min_profile: int = 0
    sidelines: int = 0
    target_smoothing: bool = False
    smoothing_sigma: float = 3.0
    rc_augment: bool = False
    device: str = (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

def create_loaders(dataset, train_frac, val_frac, test_frac, batch_size, num_workers, split_seed, rc_augment):
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
    
    train_ds.rc_augment = rc_augment
    val_ds.rc_augment = False
    test_ds.rc_augment = False

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

    return train_loader, val_loader, test_loader

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

    train_loader, val_loader, test_loader = create_loaders(
        dataset, 
        train_frac, 
        val_frac, 
        test_frac, 
        batch_size, 
        num_workers, 
        split_seed, 
        config.rc_augment)

    # Peek at one batch to infer sequence and target lengths
    *x0, y0 = next(iter(train_loader))
    seq_len = x0[0].shape[-1]

    # Initialize model
    if isinstance(dataset, SequenceBigWigClusterDataset):
        model = BPNetClusterModel(seq_len=seq_len, n_channels=4, sidelines=config.sidelines, min_profile=config.min_profile)
    elif isinstance(dataset, SequenceDualBigWigDataset):
        model = BPNetK4Model(seq_len=seq_len, n_channels=5, sidelines=config.sidelines, min_profile=config.min_profile)
    elif isinstance(dataset, SequenceBigWigDataset):
        model = BPNetModel(seq_len=seq_len, n_channels=4, sidelines=config.sidelines, min_profile=config.min_profile)
    elif isinstance(dataset, SequenceBigWigPeaksDataset):
        model = BPNetModel(seq_len=seq_len, n_channels=4, sidelines=config.sidelines, min_profile=config.min_profile)
    else:
        raise ValueError(f"Dataset type {type(dataset)} not supported")
    model.to(config.device)


    compute_receptive_field(model, seq_len)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if isinstance(dataset, SequenceBigWigClusterDataset):
        criterion = ClassifierLoss(min_profile=config.min_profile, sidelines=config.sidelines, target_smoothing=config.target_smoothing, smoothing_sigma=config.smoothing_sigma)
    else:
        criterion = BPNetLoss(
            loss_type=config.loss_type, 
            min_profile=config.min_profile,
            sidelines=config.sidelines,
            target_smoothing=config.target_smoothing,
            smoothing_sigma=config.smoothing_sigma,
        )

    criterion.to(config.device)

    # Expose loaders and datasets on the model so that downstream
    # notebooks can reuse the exact same splits and sampling logic.
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
        for batch_data in train_loader:
            if isinstance(dataset, SequenceBigWigClusterDataset):
                one_hot, label, signal = batch_data
                x_batches = [one_hot]
                y_true = signal
                label_true = label.float().unsqueeze(1).to(config.device)
            elif isinstance(dataset, SequenceDualBigWigDataset):
                sequence, k4_cutrun, target_y = batch_data
                x_batches = [sequence, k4_cutrun]
                y_true = target_y
            else:
                one_hot, signal = batch_data
                x_batches = [one_hot]
                y_true = signal
            
            # --- RC Augmentation ---
            if config.rc_augment:
                flip_mask = torch.rand(x_batches[0].shape[0], device=x_batches[0].device) < 0.5
                if flip_mask.any():
                    # DNA: swap A/T (0/3) and C/G (1/2), then flip the length axis
                    flipped = x_batches[0][flip_mask][:, [3, 2, 1, 0], :].flip(dims=[-1])
                    x_batches[0] = x_batches[0].clone()
                    x_batches[0][flip_mask] = flipped

                    # Auxiliary tracks: flip length axis only (whatever rank they are)
                    for i in range(1, len(x_batches)):
                        flipped_aux = x_batches[i][flip_mask].flip(dims=[-1])
                        x_batches[i] = x_batches[i].clone()
                        x_batches[i][flip_mask] = flipped_aux

                    # Target profile: flip length axis
                    flipped_y = y_true[flip_mask].flip(dims=[-1])
                    y_true = y_true.clone()
                    y_true[flip_mask] = flipped_y

            x_batches = [x.to(config.device) for x in x_batches]
            y_true = y_true.to(config.device)

            optimizer.zero_grad()
            profile_logits, pred_outputs = model(*x_batches)

            if isinstance(dataset, SequenceBigWigClusterDataset):
                loss = criterion(profile_logits, pred_outputs, y_true, label_true)
            else:
                loss = criterion(profile_logits, pred_outputs, y_true)
            
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
        val_counts_loss_sum = 0.0
        val_profile_loss_sum = 0.0
        val_batches = 0
        n_val_samples = 0
        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(dataset, SequenceBigWigClusterDataset):
                    one_hot, label, signal = batch_data
                    x_batches = [one_hot]
                    y_true = signal
                    label_true = label.float().unsqueeze(1).to(config.device)
                elif isinstance(dataset, SequenceDualBigWigDataset):
                    sequence, k4_cutrun, target_y = batch_data
                    x_batches = [sequence, k4_cutrun]
                    y_true = target_y
                else:
                    one_hot, signal = batch_data
                    x_batches = [one_hot]
                    y_true = signal
                
                x_batches = [x.to(config.device) for x in x_batches]
                y_true = y_true.to(config.device)
                
                profile_logits, pred_outputs = model(*x_batches)

                # Calculate components separately
                if isinstance(dataset, SequenceBigWigClusterDataset):
                    c_loss = criterion.compute_class_loss(pred_outputs, label_true)
                    p_loss = torch.tensor(0.0, device=c_loss.device)
                else: 
                    c_loss = criterion.compute_counts_loss(pred_outputs, y_true)
                    p_loss = criterion.compute_profile_loss(profile_logits, y_true)
                
                val_counts_loss_sum += c_loss.item()
                val_profile_loss_sum += p_loss.item()
                val_loss_sum += (c_loss + p_loss).item()
                val_batches += 1
                n_val_samples += y_true.shape[0]

        avg_val_loss = val_loss_sum / max(n_val_samples, 1)
        avg_val_counts_loss = val_counts_loss_sum / max(n_val_samples, 1)
        avg_val_profile_loss = val_profile_loss_sum / max(n_val_samples, 1)
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
            if isinstance(dataset, SequenceBigWigClusterDataset):
                print(
                    f"Epoch {epoch + 1}/{config.epochs} "
                    f"- train: {avg_train_loss:.4f} | val: {avg_val_loss:.4f} "
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{config.epochs} "
                    f"- train: {avg_train_loss:.4f} | val: {avg_val_loss:.4f} "
                    f"(counts: {avg_val_counts_loss:.4f}, profile: {avg_val_profile_loss:.4f})"
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
                if isinstance(dataset, SequenceBigWigClusterDataset):
                    one_hot, label, signal = batch_data
                    x_batches = [one_hot]
                    y_true = signal
                    label_true = label.float().unsqueeze(1).to(config.device)
                elif isinstance(dataset, SequenceDualBigWigDataset):
                    sequence, k4_cutrun, target_y = batch_data
                    x_batches = [sequence, k4_cutrun]
                    y_true = target_y
                else:
                    one_hot, signal = batch_data
                    x_batches = [one_hot]
                    y_true = signal
                
                x_batches = [x.to(config.device) for x in x_batches]
                y_true = y_true.to(config.device)
                
                profile_logits, pred_outputs = model(*x_batches)
                if isinstance(dataset, SequenceBigWigClusterDataset):
                    loss = criterion(profile_logits, pred_outputs, y_true, label_true)
                else:
                    loss = criterion(profile_logits, pred_outputs, y_true)
                
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
    smooth_pred: bool = False,
    smooth_sigma: float = 3.0,
    smooth_true: bool = False,
    smooth_sigma_true: float = 3.0,
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
                
                if isinstance(model, BPNetClusterModel):
                    one_hot, label, signal = batch_data
                    x_model_inputs = [one_hot]
                    y_true_batch = signal
                    label_batch = label
                elif isinstance(model, BPNetK4Model):
                    sequence, k4_cutrun, target_y = batch_data
                    x_model_inputs = [sequence, k4_cutrun]
                    y_true_batch = target_y
                else:
                    one_hot, signal = batch_data
                    x_model_inputs = [one_hot]
                    y_true_batch = signal

                x_model_inputs = [x.to(device) for x in x_model_inputs]
                y_true_batch = y_true_batch.to(device)

                batch_curr = y_true_batch.shape[0]
                for b in range(batch_curr):
                    if count >= n_show:
                        done = True
                        break
                    
                    x_singles = [x[b:b+1] for x in x_model_inputs]
                    y_true_single = y_true_batch[b]
                    
                    profile_logits, pred_outputs = model(*x_singles)

                    # Distribution-based visualization (multinomial/kldiv)
                    y_pred = F.softmax(profile_logits.squeeze(), dim=-1)
                    y_true_norm = y_true_single / (y_true_single.sum() + 1e-8)

                    if smooth_pred:
                        y_pred = BPNetLoss.smooth_signal(y_pred.view(1, -1), sigma=smooth_sigma).squeeze()

                    if smooth_true:
                        y_true_norm = BPNetLoss.smooth_signal(y_true_norm.view(1, -1), sigma=smooth_sigma_true).squeeze()

                    row, col = divmod(count, n_cols_actual)
                    ax = axes[row, col]
                    ax.plot(y_true_norm.cpu().numpy(), label="true", alpha=0.7)
                    ax.plot(y_pred.cpu().numpy(), label="pred", alpha=0.7)
                    
                    sidelines = getattr(model, "sidelines", 0)
                    if sidelines > 0:
                        ax.axvline(x=sidelines, color='k', linestyle='--', alpha=0.5)
                        ax.axvline(x=len(y_true_norm) - sidelines, color='k', linestyle='--', alpha=0.5)
                    
                    true_total = y_true_single.sum().item()
                    if isinstance(model, BPNetClusterModel):
                        true_label = label_batch[b].item()
                        pred_prob = pred_outputs.item()
                        ax.set_title(f"{split_name} {count+1}\ntrue total: {true_total:.1f}, pred: {pred_prob:.2f} (class: {true_label})")
                    else:
                        pred_total = torch.expm1(pred_outputs).item()
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
def evaluate_model_metrics(model, loader, device="cpu", min_profile=0, 
        smooth_pred=False, smooth_sigma=3.0, smooth_true=False, smooth_sigma_true=3.0):
    
    model.eval()
    model.to(device)
    all_true_log_counts, all_pred_log_counts, all_jsds = [], [], []

    all_true_labels, all_pred_probs = [], []
    is_classifier = isinstance(model, BPNetClusterModel)

    with torch.no_grad():
        for batch_data in loader:
            if is_classifier:
                one_hot, label, signal = batch_data
                x_model_inputs = [one_hot]
                y_true_batch = signal
                label_true = label
            elif isinstance(model, BPNetK4Model):
                sequence, k4_cutrun, target_y = batch_data
                x_model_inputs = [sequence, k4_cutrun]
                y_true_batch = target_y
            else:
                one_hot, signal = batch_data
                x_model_inputs = [one_hot]
                y_true_batch = signal

            x_model_inputs = [x.to(device) for x in x_model_inputs]
            y_true_batch = y_true_batch.to(device)

            profile_logits, pred_outputs = model(*x_model_inputs)

            target_counts = y_true_batch.sum(dim=-1)

            if is_classifier:
                pred_outputs = F.sigmoid(pred_outputs)
                all_pred_probs.extend(pred_outputs.squeeze(-1).cpu().numpy().tolist())
                all_true_labels.extend(label_true.cpu().numpy().tolist())
            else:
                all_pred_log_counts.extend(pred_outputs.squeeze(-1).cpu().numpy().tolist())
                all_true_log_counts.extend(torch.log1p(target_counts).cpu().numpy().tolist())

            y_pred_prob = F.softmax(profile_logits.squeeze(1), dim=-1).cpu()
            y_true_prob = (y_true_batch / (target_counts.unsqueeze(-1) + 1e-8)).cpu()

            if smooth_pred:
                y_pred_prob = BPNetLoss.smooth_signal(y_pred_prob.to(device), sigma=smooth_sigma).cpu()

            if smooth_true:
                y_true_prob = BPNetLoss.smooth_signal(y_true_prob.to(device), sigma=smooth_sigma_true).cpu()
            
            y_pred_prob = y_pred_prob.numpy()
            y_true_prob = y_true_prob.numpy()

            for i in range(y_pred_prob.shape[0]):
                if target_counts[i].item() > min_profile:
                    jsd = jensenshannon(y_true_prob[i], y_pred_prob[i])
                    if not np.isnan(jsd):
                        all_jsds.append(jsd)

    result = {
        "profile_jsd": all_jsds,
    }
    if is_classifier:
        result["class_true"] = all_true_labels
        result["class_pred"] = all_pred_probs
    else:
        result["log_counts_true"] = all_true_log_counts
        result["log_counts_pred"] = all_pred_log_counts
    return result

__all__ = [
    "create_loaders",
    "train_cnn_regressor",
    "visualize_split_predictions",
    "evaluate_model_metrics",
    "TrainConfig"
]