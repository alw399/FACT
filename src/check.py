import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, mannwhitneyu
from Bio.Seq import Seq
from Bio import motifs
from tqdm import tqdm
import os

from datas import one_hot_encode_sequence
from train import evaluate_model_metrics

def find_best_motif(seq: str, pssm, pssm_rev, window_size: int) -> tuple[float, int, str]:
    """Finds best PWM score over both strands. Returns (score, position, strand)."""
    if "N" in seq or len(seq) < window_size:
        return -np.inf, -1, "."
    bs = Seq(seq)
    f = np.asarray(pssm.calculate(bs))
    r = np.asarray(pssm_rev.calculate(bs))
    fi, ri = int(np.argmax(f)), int(np.argmax(r))
    return ((float(f[fi]), fi, "+") if f[fi] >= r[ri]
            else (float(r[ri]), ri, "-"))

def knock_out(one_hot: np.ndarray, pos: int, width: int, mode: str, rng) -> np.ndarray:
    """Modifies a one-hot array to 'knock out' a specific region."""
    oh = one_hot.copy()
    if mode == "N":
        oh[:, pos:pos + width] = 0.0
    elif mode == "random":
        oh[:, pos:pos + width] = 0.0
        for j in range(pos, pos + width):
            oh[rng.integers(0, 4), j] = 1.0
    elif mode == "shuffle":
        idx = np.arange(pos, pos + width)
        perm = rng.permutation(idx)
        oh[:, idx] = one_hot[:, perm]
    else:
        raise ValueError(f"Unknown knockout mode: {mode}")
    return oh

def implant_consensus(one_hot: np.ndarray, pos: int, consensus: str) -> np.ndarray:
    """Overwrites a one-hot region with a consensus string."""
    oh = one_hot.copy()
    oh[:, pos:pos + len(consensus)] = 0.0
    base_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for j, b in enumerate(consensus):
        if b in base_idx:
            oh[base_idx[b], pos + j] = 1.0
    return oh

@torch.no_grad()
def _predict_batch(model, one_hots, device):
    """Internal helper for batched predictions."""
    x = torch.from_numpy(np.stack(one_hots)).float().to(device)
    profile_logits, log_counts = model(x)
    # Return probabilities and log counts
    probs = torch.softmax(profile_logits, dim=-1).cpu().numpy()
    counts = log_counts.cpu().numpy()
    return probs, counts

def run_ism_diagnostic(
    model, 
    dataset, 
    motif_seq,
    device='cpu', 
    n_high=50, 
    n_low=50, 
    min_score=None,
    knockout_mode='shuffle',
    random_seed=0,
    verbose=True
):
    """
    General In-Silico Mutagenesis (ISM) Diagnostic.
    
    motif_seq: A string of letters (e.g., "CCTTTGTT") to test.
    """
    model.to(device).eval()
    rng = np.random.default_rng(random_seed)
    
    # Create PSSM from the string for searching
    motif_obj = motifs.create([motif_seq.upper()])
    # Use small pseudocounts to allow for some variation if desired
    pwm = motif_obj.counts.normalize(pseudocounts=0.5)
    pssm = pwm.log_odds()
    pssm_rev = pssm.reverse_complement()
    
    consensus = motif_seq.upper()
    W = len(consensus)
    
    # Default threshold: roughly 70% of max possible score
    if min_score is None:
        min_score = pssm.max * 0.7
    
    # 1. Rank intervals by signal
    if verbose: print("Indexing intervals by total signal...")
    totals = []
    for i in range(len(dataset)):
        iv = dataset.intervals[i]
        try:
            s = dataset._bw.values(iv.chrom, iv.start, iv.end, numpy=True)
            s = np.nan_to_num(s, nan=0.0)
            totals.append(float(np.maximum(s, 0).sum()))
        except Exception:
            totals.append(0.0)
    totals = np.array(totals)
    order_high = np.argsort(-totals)
    order_low  = np.argsort(totals)

    # 2. Knockout Test
    if verbose: print(f"Running Knockout Test (n={n_high})...")
    ko_records = []
    n_used = 0
    for idx in order_high:
        if n_used >= n_high: break
        iv = dataset.intervals[int(idx)]
        seq_str = str(dataset._fasta[iv.chrom][iv.start:iv.end]).upper()
        if "N" in seq_str or len(seq_str) < W: continue
        
        # Find motif
        score, pos, strand = find_best_motif(seq_str, pssm, pssm_rev, W)
        if score < min_score: continue
        
        one_hot = one_hot_encode_sequence(seq_str)
        
        # Baseline vs KO
        x_base = torch.from_numpy(one_hot).float().unsqueeze(0).to(device)
        
        # Dual-track support: fetch K4 signal if model requires it
        if hasattr(model, 'k4_conv') or "K4" in model.__class__.__name__:
             item = dataset[int(idx)]
             if len(item) >= 3:
                 x_aux = item[1].clone().detach().float().unsqueeze(0).to(device)
                 if x_aux.dim() == 2:
                     x_aux = x_aux.unsqueeze(1)
             else:
                 x_aux = None
             _, logc_base = model(x_base, x_aux)
        else:
             x_aux = None
             _, logc_base = model(x_base)

        counts_base = float(np.expm1(logc_base.item()))
        
        oh_ko = knock_out(one_hot, pos, W, knockout_mode, rng)
        x_ko = torch.from_numpy(oh_ko).float().unsqueeze(0).to(device)
        
        if x_aux is not None:
            _, logc_ko = model(x_ko, x_aux)
        else:
            _, logc_ko = model(x_ko)
        
        counts_ko = float(np.expm1(logc_ko.item()))
        
        # Controls
        ctrl_deltas = []
        forbidden = set(range(max(0, pos - W), min(len(seq_str), pos + 2 * W)))
        valid_starts = [p for p in range(0, len(seq_str) - W) if p not in forbidden]
        if valid_starts:
            for c_start in rng.choice(valid_starts, size=min(5, len(valid_starts)), replace=False):
                oh_c = knock_out(one_hot, int(c_start), W, knockout_mode, rng)
                x_c = torch.from_numpy(oh_c).float().unsqueeze(0).to(device)
                if x_aux is not None:
                    _, logc_c = model(x_c, x_aux)
                else:
                    _, logc_c = model(x_c)
                ctrl_deltas.append(float(np.expm1(logc_c.item())) - counts_base)
        
        ko_records.append({
            "delta_motif": counts_ko - counts_base,
            "delta_ctrl_mean": np.mean(ctrl_deltas) if ctrl_deltas else 0.0,
            "base_counts": counts_base
        })
        n_used += 1

    # 3. Implant Test
    if verbose: print(f"Running Implant Test (n={n_low})...")
    imp_records = []
    n_used = 0
    for idx in order_low:
        if n_used >= n_low: break
        iv = dataset.intervals[int(idx)]
        seq_str = str(dataset._fasta[iv.chrom][iv.start:iv.end]).upper()
        if "N" in seq_str or len(seq_str) < W: continue
        
        # Ensure no existing motif
        score, _, _ = find_best_motif(seq_str, pssm, pssm_rev, W)
        if score >= 8.0: continue
        
        one_hot = one_hot_encode_sequence(seq_str)
        center = len(seq_str) // 2
        
        # Baseline
        x_base = torch.from_numpy(one_hot).float().unsqueeze(0).to(device)
        
        if hasattr(model, 'k4_conv') or "K4" in model.__class__.__name__:
             item = dataset[int(idx)]
             if len(item) >= 3:
                 x_aux = item[1].clone().detach().float().unsqueeze(0).to(device)
                 if x_aux.dim() == 2:
                     x_aux = x_aux.unsqueeze(1)
             else:
                 x_aux = None
             _, logc_base = model(x_base, x_aux)
        else:
             x_aux = None
             _, logc_base = model(x_base)
             
        counts_base = float(np.expm1(logc_base.item()))
        
        # Implant
        oh_imp = implant_consensus(one_hot, center, consensus)
        x_imp = torch.from_numpy(oh_imp).float().unsqueeze(0).to(device)
        if x_aux is not None:
            _, logc_imp = model(x_imp, x_aux)
        else:
            _, logc_imp = model(x_imp)
        counts_imp = float(np.expm1(logc_imp.item()))
        
        # Random Control
        rand_seq = "".join(rng.choice(list("ACGT"), size=W))
        oh_rand = implant_consensus(one_hot, center, rand_seq)
        x_rand = torch.from_numpy(oh_rand).float().unsqueeze(0).to(device)
        if x_aux is not None:
            _, logc_rand = model(x_rand, x_aux)
        else:
            _, logc_rand = model(x_rand)
        counts_rand = float(np.expm1(logc_rand.item()))
        
        imp_records.append({
            "delta_motif": counts_imp - counts_base,
            "delta_rand": counts_rand - counts_base
        })
        n_used += 1

    ko_df = pd.DataFrame(ko_records)
    imp_df = pd.DataFrame(imp_records)
    
    if verbose:
        _summarize_results(ko_df, imp_df)
        
    return ko_df, imp_df

def _summarize_results(ko_df, imp_df):
    print("\n" + "="*30)
    print("ISM DIAGNOSTIC SUMMARY")
    print("="*30)
    if not ko_df.empty:
        motif_drop = ko_df["delta_motif"].mean()
        ctrl_drop = ko_df["delta_ctrl_mean"].mean()
        print(f"Knockout (n={len(ko_df)}):")
        print(f"  Mean Motif Delta: {motif_drop:+.2f}")
        print(f"  Mean Ctrl Delta:   {ctrl_drop:+.2f}")
        try:
            _, p = wilcoxon(ko_df["delta_motif"], ko_df["delta_ctrl_mean"], alternative="less")
            print(f"  Wilcoxon p-val:    {p:.2e}")
        except: pass

    if not imp_df.empty:
        motif_gain = imp_df["delta_motif"].mean()
        rand_gain = imp_df["delta_rand"].mean()
        print(f"\nImplant (n={len(imp_df)}):")
        print(f"  Mean Motif Delta: {motif_gain:+.2f}")
        print(f"  Mean Rand Delta:  {rand_gain:+.2f}")
        try:
            _, p = wilcoxon(imp_df["delta_motif"], imp_df["delta_rand"], alternative="greater")
            print(f"  Wilcoxon p-val:    {p:.2e}")
        except: pass
    print("="*30 + "\n")

def plot_ism_diagnostic(ko_df, imp_df, motif_name="Motif"):
    """Visualizes the ISM results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # KO Plot
    if not ko_df.empty:
        ax = axes[0]
        ax.scatter(ko_df["delta_ctrl_mean"], ko_df["delta_motif"], alpha=0.5)
        lim = max(abs(ko_df["delta_motif"]).max(), abs(ko_df["delta_ctrl_mean"]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.7)
        ax.set_xlabel("Control Delta")
        ax.set_ylabel(f"{motif_name} KO Delta")
        ax.set_title("Knockout Test")
        
    # Implant Plot
    if not imp_df.empty:
        ax = axes[1]
        ax.scatter(imp_df["delta_rand"], imp_df["delta_motif"], alpha=0.5, color='green')
        lim = max(abs(imp_df["delta_motif"]).max(), abs(imp_df["delta_rand"]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.7)
        ax.set_xlabel("Random Implant Delta")
        ax.set_ylabel(f"{motif_name} Implant Delta")
        ax.set_title("Implant Test")
        
    plt.tight_layout()
    plt.show()
