import torch
import numpy as np
from modiscolite.tfmodisco import TFMoDISco
from modiscolite import io
from captum.attr import DeepLiftShap
from scipy.spatial.distance import jensenshannon
import torch.nn.functional as F

from cnn import BPNetModel, BPNetK4Model


# ─── Dinucleotide shuffle (Altschul-Erikson) ─────────────────────────────────
#
# Preserves both mononucleotide AND dinucleotide composition exactly.
# Used as the composition-matched baseline for DeepLIFT attribution: GC
# content, CpG-island character, and dinucleotide structure all cancel
# between input and baseline, so attributions reflect motif content rather
# than composition bias.

def _last_edges_form_tree(last_edge, last, successors):
    for v in range(4):
        if v == last or not successors[v]:
            continue
        seen, cur = {v}, last_edge[v]
        while cur != last:
            if cur is None or cur in seen:
                return False
            seen.add(cur)
            cur = last_edge[cur]
    return True


def _dinuc_shuffle_indices(seq_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle a (L,) array of base indices in {0,1,2,3} preserving dinucleotides."""
    n = len(seq_idx)
    if n < 2:
        return seq_idx.copy()
    first, last = int(seq_idx[0]), int(seq_idx[-1])

    successors = [[] for _ in range(4)]
    for i in range(n - 1):
        successors[int(seq_idx[i])].append(int(seq_idx[i + 1]))

    # Sample a "last-edge tree" rooted at `last`. With 4 vertices this almost
    # never iterates more than once.
    for _ in range(500):
        last_edge = [None] * 4
        for v in range(4):
            if v != last and successors[v]:
                last_edge[v] = int(rng.choice(successors[v]))
        if _last_edges_form_tree(last_edge, last, successors):
            break
    else:
        return seq_idx.copy()

    queues = [[] for _ in range(4)]
    for v in range(4):
        if not successors[v]:
            continue
        if v == last or last_edge[v] is None:
            queues[v] = list(successors[v])
            rng.shuffle(queues[v])
        else:
            rest = list(successors[v])
            rest.remove(last_edge[v])
            rng.shuffle(rest)
            rest.append(last_edge[v])
            queues[v] = rest

    out = [first]
    cur = first
    for _ in range(n - 1):
        cur = queues[cur].pop(0)
        out.append(cur)
    return np.array(out, dtype=seq_idx.dtype)


def dinuc_shuffle(seq_tensor: torch.Tensor, num_shuffles: int = 10,
                  seed: int | None = None) -> torch.Tensor:
    """Generate `num_shuffles` dinuc-shuffled copies of a (4, L) one-hot tensor."""
    seq_np = seq_tensor.cpu().numpy()
    seq_idx = np.argmax(seq_np, axis=0)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_shuffles):
        shuf = _dinuc_shuffle_indices(seq_idx, rng)
        oh = np.zeros_like(seq_np)
        oh[shuf, np.arange(len(shuf))] = 1.0
        out.append(oh)
    return torch.tensor(np.stack(out), dtype=torch.float32)


# ─── Attribution engine ─────────────────────────────────────────────────────
#
# For BPNet, attributing `prof.sum()` is broken: the multinomial profile loss
# is invariant to a constant shift in the logits (softmax over positions),
# so the model has no incentive to assign meaningful absolute values to the
# logit sum, and attribution picks up arbitrary noise.
#
# The correct BPNet attribution target is the *profile-shape contribution
# scalar*: sum_pos w_pos · logit_pos, where w_pos = softmax(logits) computed
# on the input and held FROZEN during attribution. The frozen weights make
# the scalar shape-sensitive (it weights logit changes by where the model
# thinks signal lives) and stable across DeepLIFT's input AND baseline
# forward passes — the contribution decomposes as
#   sum_pos w_pos · (logit_input - logit_baseline)
# which is exactly what we want for motif discovery.

class _ProfileShapeWrapper(torch.nn.Module):
    """Wraps a BPNet model to return a (B,) profile-shape scalar for Captum."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.is_dual = isinstance(model, BPNetK4Model)
        self._weights: torch.Tensor | None = None

    def _profile(self, seq, k4):
        return self.model(seq, k4)[0] if self.is_dual else self.model(seq)[0]

    def freeze_weights(self, seq: torch.Tensor, k4: torch.Tensor | None = None) -> None:
        """Compute and freeze softmax weights on the input. Call once before attribute()."""
        with torch.no_grad():
            prof = self._profile(seq, k4)
            self._weights = torch.softmax(prof, dim=-1).detach()

    def forward(self, seq, k4=None):
        prof = self._profile(seq, k4)
        if self._weights is None:
            raise RuntimeError("Call freeze_weights() before attribute().")
        return (self._weights * prof).sum(dim=(1, 2))


class _CountsWrapper(torch.nn.Module):
    """Wraps a BPNet model to return the (B,) total counts scalar for Captum."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.is_dual = isinstance(model, BPNetK4Model)

    def forward(self, seq, k4=None):
        return self.model(seq, k4)[1] if self.is_dual else self.model(seq)[1]


class MotifAttributor:
    def __init__(self, model: torch.nn.Module, device: str = "cpu", head: str = "profile"):
        self.model = model.to(device).eval()
        self.device = device
        self.head = head
        
        if head == "profile":
            self.wrapper = _ProfileShapeWrapper(self.model).to(device)
        elif head == "counts":
            self.wrapper = _CountsWrapper(self.model).to(device)
        else:
            raise ValueError(f"Unknown head: {head}")
            
        self.dl = DeepLiftShap(self.wrapper)

    def compute(self, x: torch.Tensor, k4: torch.Tensor | None = None,
                num_baselines: int = 20) -> np.ndarray:
        x_t = x.unsqueeze(0).to(self.device)                                  # (1, 4, L)
        dna_baselines = dinuc_shuffle(x[:4], num_shuffles=num_baselines).to(self.device)  # (N, 4, L)

        if self.wrapper.is_dual:
            k4_t = k4.to(self.device)
            while k4_t.dim() < 3:
                k4_t = k4_t.unsqueeze(0)
            k4_baselines = k4_t.repeat(num_baselines, 1, 1)
            
            if self.head == "profile":
                self.wrapper.freeze_weights(x_t, k4_t)
                
            attr = self.dl.attribute(
                (x_t, k4_t),
                baselines=(dna_baselines, k4_baselines),
            )
            return attr[0][0].detach().cpu().numpy()                          # (4, L)
        else:
            if self.head == "profile":
                self.wrapper.freeze_weights(x_t)
                
            attr = self.dl.attribute(x_t, baselines=dna_baselines)
            return attr[0].detach().cpu().numpy()                             # (4, L)


# ─── Discovery driver ───────────────────────────────────────────────────────

def discover_motifs(
    model: torch.nn.Module,
    loader,
    device: str = "cpu",
    head: str = "profile",
    n_seqs: int = 5000,
    min_counts: float = 1000,
    save_path: str | None = None,
    sliding_window_size: int = 11,
    flank_size: int = 4,
    target_seqlet_fdr: float = 0.25,
    n_leiden_runs: int = 100,
    min_metacluster_size: int = 30,
    max_seqlets_per_metacluster: int = 100_000,
    final_min_cluster_size: int = 20,
    num_baselines: int = 20,
    max_base_fraction: float | None = 0.55,
    jsd_thresh: float = 0.2,
):
    """Run dinuc-baseline DeepLIFT attribution + TF-MoDISco motif discovery.

    Parameters of note for SOX2-style discovery
    -------------------------------------------
    sliding_window_size : 11
        Default in modisco-lite is 21, optimised for 12-15bp motifs. SOX2 is
        ~7bp; an 11bp window centres better on the SOX2 footprint and lets
        SOX seqlets cluster on motif content rather than being dominated by
        flanking sequence.
    flank_size : 4
        Smaller flanks pair with the smaller window. Keeps the seqlet/window
        ratio similar to the default (~0.45-0.5).
    target_seqlet_fdr : 0.25
        Slightly more permissive than the modisco-lite default of 0.20. SOX2
        seqlets have lower individual attribution magnitude than GC-rich
        KLF/ZNF seqlets (AT-rich → smaller (input-baseline) contributions),
        so a stricter FDR drops them disproportionately.
    min_metacluster_size : 30
        Default is 100. If your SOX2 cluster has 50-80 seqlets after Leiden,
        it gets discarded entirely with the default. This is the most likely
        single reason a known-good motif fails to appear in the output.
    n_leiden_runs : 100
        More Leiden runs → more stable cluster boundaries. Helps separate
        the SOX2 cluster from partially-overlapping POU/T-box clusters.
    max_base_fraction : 0.55 (set to None to disable)
        Skip intervals where any single base (A/C/G/T) makes up more than
        this fraction of the sequence. Pure poly-G and poly-A regions
        otherwise eat seqlet mass into uninformative low-complexity clusters
        (pattern_3, pattern_5, pattern_6 in earlier runs were literally
        20bp of G). The dinuc-shuffled baseline does not adequately
        suppress attribution at these regions because their dinuc-shuffle
        is essentially the same sequence. 0.55 catches ~poly-G/A while
        keeping ordinary GC-rich enhancers (which top out around 65-70%
        single-base when they're CpG-rich) — tune up if you find too many
        real intervals being dropped.
    """
    attributor = MotifAttributor(model, device, head=head)
    all_ohs, all_attrs = [], []

    print(f"Generating attributions on {device}...")
    count = 0
    n_skipped_counts = 0
    n_skipped_complexity = 0
    n_skipped_jsd = 0
    for batch in loader:
        if count >= n_seqs:
            break
        x_batch, y_batch = batch[0], batch[-1]
        k4_batch = batch[1] if len(batch) == 3 else None
        
        # Compute predicted profiles for JSD filtering
        with torch.no_grad():
            x_batch_dev = x_batch.to(device)
            k4_batch_dev = k4_batch.to(device) if k4_batch is not None else None
            if k4_batch_dev is not None:
                profile_logits, _ = model(x_batch_dev, k4_batch_dev)
            else:
                profile_logits, _ = model(x_batch_dev)
            
            y_pred_prob = F.softmax(profile_logits.squeeze(1), dim=-1).cpu().numpy()
            target_counts = y_batch.sum(dim=-1)
            y_true_prob = (y_batch / (target_counts.unsqueeze(-1) + 1e-8)).numpy()
        for i in range(x_batch.shape[0]):
            if count >= n_seqs:
                break
            if y_batch[i].sum() < min_counts:
                n_skipped_counts += 1
                continue

            # Low-complexity filter: skip intervals where any one base
            # dominates the sequence. Operates on the DNA channels only
            # (first 4 channels), regardless of whether the model is dual-track.
            if max_base_fraction is not None:
                dna = x_batch[i][:4].cpu().numpy()
                base_fractions = dna.sum(axis=-1) / dna.shape[-1]  # (4,)
                if base_fractions.max() > max_base_fraction:
                    n_skipped_complexity += 1
                    continue

            # JSD filter
            jsd = jensenshannon(y_true_prob[i], y_pred_prob[i])
            if np.isnan(jsd) or jsd >= jsd_thresh:
                n_skipped_jsd += 1
                continue

            with torch.enable_grad():
                attr = attributor.compute(
                    x_batch[i],
                    k4_batch[i] if k4_batch is not None else None,
                    num_baselines=num_baselines,
                )
            all_ohs.append(x_batch[i][:4].numpy())
            all_attrs.append(attr[:4])
            count += 1

    if not all_ohs:
        raise ValueError("No samples passed thresholds.")

    print(f"  kept {count} sequences for attribution")
    print(f"  skipped {n_skipped_counts} (below min_counts={min_counts})")
    if max_base_fraction is not None:
        print(f"  skipped {n_skipped_complexity} "
              f"(low complexity, max_base_fraction>{max_base_fraction})")
    print(f"  skipped {n_skipped_jsd} (jsd distance >= {jsd_thresh})")

    ohs_mod = np.transpose(np.array(all_ohs), (0, 2, 1)).astype(np.float32)
    attrs_mod = np.transpose(np.array(all_attrs), (0, 2, 1)).astype(np.float32)

    print(f"Executing MoDISco-lite on {len(ohs_mod)} sequences "
          f"(window={sliding_window_size}, flank={flank_size}, "
          f"FDR={target_seqlet_fdr}, leiden={n_leiden_runs})...")
    pos_pats, neg_pats = TFMoDISco(
        one_hot=ohs_mod,
        hypothetical_contribs=attrs_mod,
        sliding_window_size=sliding_window_size,
        flank_size=flank_size,
        target_seqlet_fdr=target_seqlet_fdr,
        n_leiden_runs=n_leiden_runs,
        min_metacluster_size=min_metacluster_size,
        max_seqlets_per_metacluster=max_seqlets_per_metacluster,
        final_min_cluster_size=final_min_cluster_size,
        verbose=True,
    )

    if save_path:
        io.save_hdf5(save_path, pos_pats, neg_pats, window_size=sliding_window_size)
    return pos_pats, neg_pats


def read_report(html_path: str, meme_path: str):
    """
    Reads the motifs.html report and maps the JASPAR IDs in match0, match1, match2
    to their human-readable motif names using the JASPAR MEME file.
    """
    import pandas as pd
    
    # 1. Parse the MEME file to build a JASPAR ID to Motif Name mapping
    jaspar_to_name = {}
    with open(meme_path, 'r') as f:
        for line in f:
            if line.startswith("MOTIF"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    jaspar_to_name[parts[1]] = parts[2]

    # 2. Read the HTML report into a DataFrame
    df = pd.read_html(html_path)[0]
    df = df.dropna(axis=1, how='all')

    # 3. Add columns mapping the JASPAR IDs to motif names
    for i in range(3):
        col = f'match{i}'
        name_col = f'match{i}_name'
        if col in df.columns:
            df[name_col] = df[col].map(jaspar_to_name)

    # 4. Reorder the columns to place the names right next to the match IDs
    cols = list(df.columns)
    name_cols = [f'match{i}_name' for i in range(3) if f'match{i}_name' in cols]
    cols = [c for c in cols if c not in name_cols]
    
    for i in range(3):
        match_col = f'match{i}'
        name_col = f'match{i}_name'
        if match_col in cols and name_col in name_cols:
            idx = cols.index(match_col)
            cols.insert(idx + 1, name_col)

    df = df[cols]
    df.set_index('pattern', inplace=True)
    return df