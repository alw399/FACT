"""
BPNet-style Motif Discovery for CUT&RUN models.

Implements:
1. Input x Gradient attribution (DeepLIFT approximation)
2. TF-MoDISco motif discovery via modisco-lite
3. First-layer filter PWM extraction
4. JASPAR database matching
5. Premium visualizations
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import logomaker
from typing import Optional, List, Tuple, Union
import os, json, urllib.request, math
from Bio import motifs
import Bio.motifs.jaspar
from cnn import BPNetK4Model, BPNetModel
from modiscolite.tfmodisco import TFMoDISco
from modiscolite import io
from captum.attr import DeepLift

# ─── Section 1: Attribution / Contribution Scores ────────────────────────────

def compute_input_gradient(model, x, device="cpu", head="profile"):
    """Compute input x gradient attribution. Automatically detects model type.
    Returns (4, L) for BPNetModel or (5, L) for BPNetK4Model."""
    model.eval()
    model.to(device)
    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    x_t.requires_grad_(True)

    if isinstance(model, BPNetK4Model):
        profile, counts = model(x_t[:, :4, :], x_t[:, 4:, :])
    else:
        profile, counts = model(x_t)

    output = counts if head == "counts" else profile.sum()
    output.backward()

    grad = x_t.grad[0].detach().cpu().numpy()  # (C, L) — all channels
    dna_attr = grad[:4] * x[:4]
    if isinstance(model, BPNetK4Model):
        k4_attr = grad[4:] * x[4:]
        return np.concatenate([dna_attr, k4_attr], axis=0)  # (5, L)
    return dna_attr  # (4, L)


class DeepLiftWrapper(torch.nn.Module):
    """Wrapper to make BPNet models compatible with Captum DeepLIFT."""
    def __init__(self, model, head="profile"):
        super().__init__()
        self.model = model
        self.head = head
        self.is_dual = isinstance(model, BPNetK4Model)

    def forward(self, seq, add=None):
        if self.is_dual:
            prof, counts = self.model(seq, add)
        else:
            prof, counts = self.model(seq)
        
        if self.head == "counts":
            return counts # Return (B,) log-counts
        return prof.sum(dim=(1, 2)) # Return (B,) profile sums


def compute_deeplift(model, x, device="cpu", head="profile"):
    """Compute DeepLIFT attribution (Global interpretation relative to reference).
    Returns (4, L) for BPNetModel or (5, L) for BPNetK4Model."""
    if DeepLift is None:
        print("Warning: Captum not installed. Falling back to Input x Gradient.")
        return compute_input_gradient(model, x, device, head=head)

    model.eval()
    model.to(device)
    wrapped_model = DeepLiftWrapper(model, head=head)
    dl = DeepLift(wrapped_model)

    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    if wrapped_model.is_dual:
        attr = dl.attribute((x_t[:, :4, :], x_t[:, 4:, :]), target=None)
        dna_attr = attr[0][0].detach().cpu().numpy()  # (4, L)
        k4_attr  = attr[1][0].detach().cpu().numpy()  # (1, L)
        return np.concatenate([dna_attr, k4_attr], axis=0)  # (5, L)
    else:
        attr = dl.attribute(x_t[:, :4, :], target=None)
        return attr[0].detach().cpu().numpy()  # (4, L)


def compute_attributions_batch(model, loader, device="cpu", n_seqs=None,
                                min_counts=7000, method="gradient", head="profile"):
    """Compute attributions for sequences with total signal >= min_counts."""
    model.eval()
    model.to(device)
    all_ohs, all_attrs = [], []
    collected = 0

    if n_seqs is None:
        n_seqs = len(loader.dataset)

    # Select attribution function
    attr_fn = compute_deeplift if method == "deeplift" else compute_input_gradient

    for batch in loader:
        if collected >= n_seqs:
            break
        
        seqs = batch[0]
        targets = batch[-1]
        add = batch[1] if len(batch) == 3 else None

        for i in range(seqs.shape[0]):
            if collected >= n_seqs:
                break
            y = targets[i].numpy() if isinstance(targets[i], torch.Tensor) else targets[i]
            total = float(np.sum(np.maximum(y, 0)))
            if total < min_counts:
                continue

            s = seqs[i]
            if add is not None:
                a = add[i]
                if a.dim() == 1: a = a.unsqueeze(0)
                L = s.shape[-1]
                if a.shape[-1] != L:
                    a = F.interpolate(a.unsqueeze(0).float(), size=L, mode="linear", align_corners=False).squeeze(0)
                x_np = torch.cat([s, a], dim=0).numpy()
            else:
                x_np = s.numpy()

            attr = attr_fn(model, x_np, device, head=head)
            all_ohs.append(x_np[:4])
            all_attrs.append(attr)
            collected += 1

    print(f"Collected {collected} sequences with counts >= {min_counts} using {method} on {head} head")
    return np.array(all_ohs), np.array(all_attrs)


# ─── Section 2: TF-MoDISco Motif Discovery ──────────────────────────────────

def run_modisco(one_hots, attributions, window_size=21, max_seqlets=20000,
                n_leiden=50, verbose=True):
    """Run TF-MoDISco on attribution scores.
    one_hots: (N, 4, L), attributions: (N, 4, L). Returns (pos_patterns, neg_patterns)."""

    if window_size is None:
        window_size = 21

    # modisco-lite expects (N, L, 4)
    ohs = np.transpose(one_hots, (0, 2, 1)).astype(np.float32)
    attrs = np.transpose(attributions, (0, 2, 1)).astype(np.float32)

    if verbose:
        print(f"Running TF-MoDISco on {ohs.shape[0]} seqs, L={ohs.shape[1]}, window_size={window_size}...")

    pos_patterns, neg_patterns = TFMoDISco(
        one_hot=ohs,
        hypothetical_contribs=attrs,
        sliding_window_size=window_size,
        max_seqlets_per_metacluster=max_seqlets,
        n_leiden_runs=n_leiden,
        verbose=verbose
    )
    n_pos = len(pos_patterns) if pos_patterns else 0
    n_neg = len(neg_patterns) if neg_patterns else 0
    if verbose:
        print(f"Found {n_pos} positive, {n_neg} negative motif patterns")
    return pos_patterns, neg_patterns


def extract_motifs_from_patterns(patterns, sign="pos"):
    """Convert modisco SeqletSet patterns into a list of motif dicts.
    Each dict has: cwm (4,L), pfm (4,L), n_seqlets, name."""
    if patterns is None:
        return []
    motifs_list = []
    for i, p in enumerate(patterns):
        # p.contrib_scores is (L,4) CWM, p.sequence is (L,4) PFM
        cwm = p.contrib_scores.T  # -> (4, L)
        pfm = p.sequence.T        # -> (4, L)
        motifs_list.append({
            "cwm": cwm, "pfm": pfm,
            "n_seqlets": len(p.seqlets),
            "name": f"{sign}_{i}",
            "pattern": p
        })
    return motifs_list


def discover_motifs(model, loader, device="cpu", n_seqs=500, min_counts=7000,
                    window_size=21, max_seqlets=20000,
                    n_leiden=50, verbose=True, method="gradient", head="profile",
                    save_path=None, k4_window=200):
    """End-to-end pipeline: attribution -> TF-MoDISco -> motif dicts.

    For BPNetK4Model, automatically enriches each motif dict with:
      'k4_profile': mean K4 attribution centered on seqlets (k4_window bp)
      'k4_std':     std across seqlets
    k4_window: number of positions to extract around each seqlet center.
    """
    if n_seqs is None:
        n_seqs = len(loader.dataset)

    # 1. Compute Attributions — (N,4,L) for base model, (N,5,L) for K4 model
    ohs, attrs = compute_attributions_batch(
        model, loader, device, n_seqs, min_counts, method=method, head=head)

    if len(ohs) == 0:
        print("No sequences passed the min_counts filter!")
        return []

    # 2. Split DNA and K4 channels
    dna_attrs = attrs[:, :4, :]                                    # always (N,4,L)
    k4_attrs  = attrs[:, 4:, :] if attrs.shape[1] == 5 else None  # (N,1,L) or None

    # 3. Run TF-MoDISco on DNA channels only
    pos_pats, neg_pats = run_modisco(ohs, dna_attrs, window_size, max_seqlets,
                                      n_leiden, verbose)

    # 4. Optionally save to HDF5 (for 'modisco report' CLI)
    if save_path:
        print(f"Saving results to {save_path}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        io.save_hdf5(save_path, pos_pats, neg_pats, window_size=window_size)

    # 5. Extract into Python dicts
    motifs_out = extract_motifs_from_patterns(pos_pats, "pos")
    motifs_out += extract_motifs_from_patterns(neg_pats, "neg")

    # 6. Enrich with K4 profiles if this is a K4 model
    if k4_attrs is not None:
        pos_k4 = extract_k4_seqlet_profiles(pos_pats or [], k4_attrs, window=k4_window)
        neg_k4 = extract_k4_seqlet_profiles(neg_pats or [], k4_attrs, window=k4_window)
        for m, k in zip(motifs_out, pos_k4 + neg_k4):
            m.update(k)

    return motifs_out


# ─── Section 2b: K4 Seqlet Profile Extraction ───────────────────────────────

def extract_k4_seqlet_profiles(patterns, k4_attrs, window=200):
    """Compute mean K4 attribution profile centered on each pattern's seqlets.

    Args:
        patterns:  list of MoDISco pattern objects (each has .seqlets)
        k4_attrs:  (N, 1, L) K4 attribution array
        window:    number of positions to extract centered on each seqlet

    Returns:
        list of dicts, one per pattern, with keys:
          'k4_profile': (window,) mean K4 attribution across seqlets
          'k4_std':     (window,) std deviation across seqlets
    """
    half = window // 2
    L = k4_attrs.shape[2]
    results = []

    for pattern in patterns:
        tracks = []
        for s in pattern.seqlets:
            center = (s.start + s.end) // 2
            lo, hi = center - half, center + half
            lo_clip, hi_clip = max(0, lo), min(L, hi)
            track = k4_attrs[s.example_idx, 0, lo_clip:hi_clip].copy()
            # Pad if seqlet is near a sequence edge
            pad_l, pad_r = lo_clip - lo, hi - hi_clip
            if pad_l > 0 or pad_r > 0:
                track = np.pad(track, (pad_l, pad_r), mode='constant')
            if len(track) != window:
                continue  # skip malformed windows
            if s.is_revcomp:
                track = track[::-1]
            tracks.append(track)

        if tracks:
            arr = np.stack(tracks)  # (n_seqlets, window)
            results.append({'k4_profile': arr.mean(axis=0), 'k4_std': arr.std(axis=0)})
        else:
            results.append({'k4_profile': np.zeros(window), 'k4_std': np.zeros(window)})

    return results


# ─── Section 3: First-Layer Filter PWMs ──────────────────────────────────────

def extract_filter_pwms(model, loader, device="cpu", n_batches=5,
                        top_n_patches=200):
    """Extract PWMs from top-activating patches of the first conv layer."""
    model.to(device)
    model.eval()
    first_conv = model.encoder[0]
    fw = first_conv.kernel_size[0]
    nf = first_conv.out_channels

    filter_patches = [[] for _ in range(nf)]
    filter_scores = [[] for _ in range(nf)]

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            if len(batch) == 2:
                seq, _ = batch
                x_in = seq.to(device)
            else:
                seq, add, _ = batch
                L = seq.shape[-1]
                if add.dim() == 2: add = add.unsqueeze(1)
                if add.shape[-1] != L:
                    add = F.interpolate(add.float(), size=L, mode="linear", align_corners=False)
                x_in = torch.cat([seq.to(device), add.to(device)], dim=1)

            acts = model.encoder[:3](x_in)  # Conv->BN->ReLU
            for f in range(nf):
                f_acts = acts[:, f, :]
                flat = f_acts.flatten()
                k = min(50, len(flat))
                vals, idxs = torch.topk(flat, k=k)
                for val, idx in zip(vals, idxs):
                    if val <= 0: continue
                    b = idx // f_acts.shape[1]
                    pos = idx % f_acts.shape[1]
                    half = fw // 2
                    s, e = pos - half, pos + half + 1
                    if s >= 0 and e <= seq.shape[-1]:
                        patch = seq[b, :4, s:e].numpy()
                        if patch.shape[1] == fw:
                            filter_patches[f].append(patch)
                            filter_scores[f].append(val.item())

    pwms = []
    for f in range(nf):
        if not filter_patches[f]:
            pwms.append(np.zeros((4, fw)))
            continue
        combined = sorted(zip(filter_scores[f], filter_patches[f]),
                         key=lambda x: x[0], reverse=True)
        top = [p for _, p in combined[:top_n_patches]]
        pwms.append(np.mean(top, axis=0))
    return pwms


def get_consensus(pwm):
    """Get consensus DNA sequence from a PWM (4, L)."""
    bases = ['A', 'C', 'G', 'T']
    if np.max(pwm) == 0:
        return "N" * pwm.shape[1]
    return "".join([bases[i] for i in np.argmax(pwm, axis=0)])


# ─── Section 4: JASPAR Database Matching ─────────────────────────────────────

def _ensure_jaspar_db(taxon="vertebrates", db_path=None):
    """Ensures the JASPAR database file exists, downloading it to db_path if necessary."""
    # 1. If db_path is provided and already exists, we're done
    if db_path is not None and os.path.exists(db_path):
        return db_path
    
    # 2. Determine the target filename
    # If no db_path is provided, use the default name in the current directory
    target_file = db_path if db_path is not None else f"JASPAR2024_CORE_{taxon}_non-redundant_pfms_jaspar.txt"
    
    if os.path.exists(target_file):
        return target_file

    # 3. Download to the target_file location
    url = f"https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_{taxon}_non-redundant_pfms_jaspar.txt"
    print(f"Downloading JASPAR 2024 CORE ({taxon}) to {target_file}...")
    
    try:
        urllib.request.urlretrieve(url, target_file)
        return target_file
    except Exception as e:
        print(f"Primary download failed: {e}. Trying fallback...")
        # Fallback to the generic non-taxon-specific database name
        fallback_name = "JASPAR2024_CORE_non-redundant_pfms_jaspar.txt"
        # If the user provided a directory in db_path, we should try to save there
        if db_path and os.path.isdir(os.path.dirname(db_path)):
            fb_path = os.path.join(os.path.dirname(db_path), fallback_name)
        else:
            fb_path = fallback_name
            
        if os.path.exists(fb_path):
            return fb_path
            
        fb_url = "https://jaspar.elixir.no/download/data/2024/CORE/" + fallback_name
        urllib.request.urlretrieve(fb_url, fb_path)
        return fb_path


def _get_jaspar_species_ids(species):
    cache_file = "jaspar_species_cache.json"
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f: cache = json.load(f)
    if species in cache:
        return set(cache[species])
    url = (f"https://jaspar.elixir.no/api/v1/matrix/?collection=CORE"
           f"&species={species.replace(' ', '+')}&format=json&page_size=5000")
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
            ids = [r['matrix_id'] for r in data.get('results', [])]
            base_ids = [mid.split('.')[0] for mid in ids]
            all_ids = set(ids) | set(base_ids)
            cache[species] = list(all_ids)
            with open(cache_file, "w") as f: json.dump(cache, f)
            return all_ids
    except Exception as e:
        print(f"Error fetching species: {e}")
        return set()


def search_jaspar(pwm, taxon="vertebrates", species="Mus musculus", top_n=5, db_path=None):
    """Search PWM (4,L) against JASPAR CORE using Pearson correlation."""
    db_path = _ensure_jaspar_db(taxon, db_path)
    if not db_path or not os.path.exists(db_path): return []
    allowed = _get_jaspar_species_ids(species) if species else None
    if allowed is not None and not allowed:
        allowed = None

    with open(db_path) as h:
        jaspar = motifs.jaspar.read(h, "jaspar")

    Lq = pwm.shape[1]
    results = []
    for m in jaspar:
        if allowed and m.matrix_id not in allowed and m.matrix_id.split('.')[0] not in allowed:
            continue
        mp = m.counts.normalize()
        ma = np.array([mp['A'], mp['C'], mp['G'], mp['T']])
        Lt = ma.shape[1]
        best = -1.0
        for strand in [ma, ma[::-1, ::-1]]:
            overlap = min(5, Lq, Lt)
            for off in range(-Lt + overlap, Lq - overlap + 1):
                qs, qe = max(0, off), min(Lq, off + Lt)
                ts, te = max(0, -off), min(Lt, Lq - off)
                if (qe - qs) < overlap: continue
                c = np.corrcoef(pwm[:, qs:qe].flatten(), strand[:, ts:te].flatten())
                r = c[0, 1] if c.shape == (2, 2) else 0.0
                if np.isnan(r): r = 0.0
                best = max(best, r)
        results.append({"id": m.matrix_id, "name": m.name, "correlation": best, "pwm": ma})
    results.sort(key=lambda x: x["correlation"], reverse=True)
    return results[:top_n]


# ─── Section 5: Visualization ────────────────────────────────────────────────

# Standard scientific color palette (Light Mode)
_COLORS = {
    'bg': '#ffffff',        # White background
    'card': '#ffffff',      # White plot area
    'accent': '#d32f2f',    # Red for predictions
    'accent2': '#1976d2',   # Blue for true signal
    'text': '#000000',      # Black text
    'grid': '#e0e0e0',      # Light gray grid
    'A': '#228B22',         # Forest Green
    'C': '#0000FF',         # Blue
    'G': '#FFA500',         # Orange
    'T': '#FF0000',         # Red
}

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(_COLORS['card'])
    ax.set_title(title, color=_COLORS['text'], fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color=_COLORS['text'], fontsize=9)
    ax.set_ylabel(ylabel, color=_COLORS['text'], fontsize=9)
    ax.tick_params(colors=_COLORS['text'], labelsize=8)
    for s in ax.spines.values():
        s.set_color(_COLORS['grid'])


def plot_attribution_map(sequence, attribution, y_true=None, y_pred=None,
                         title="Attribution Map", figsize=(14, 5)):
    """BPNet-style attribution visualization with profile overlay."""
    n_panels = 1 + (1 if y_true is not None else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, gridspec_kw={'height_ratios': [1]*n_panels})
    fig.patch.set_facecolor(_COLORS['bg'])
    if n_panels == 1: axes = [axes]

    idx = 0
    if y_true is not None:
        ax = axes[idx]; idx += 1
        _style_ax(ax, "Binding Profile", ylabel="Signal")
        if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().numpy()
        y_true = y_true.squeeze()
        ax.fill_between(range(len(y_true)), y_true, alpha=0.4, color=_COLORS['accent2'], label='true')
        if y_pred is not None:
            if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
            y_pred = y_pred.squeeze()
            ax.plot(range(len(y_pred)), y_pred, color=_COLORS['accent'], lw=1.5, alpha=0.9, label='pred')
            ax.legend(facecolor=_COLORS['card'], edgecolor=_COLORS['grid'], labelcolor=_COLORS['text'])

    # Attribution as weighted sequence logo
    ax = axes[idx]
    _style_ax(ax, title, xlabel="Position", ylabel="Contribution")
    df = pd.DataFrame(attribution.T, columns=['A', 'C', 'G', 'T'])
    colors = {b: _COLORS[b] for b in 'ACGT'}
    logomaker.Logo(df, ax=ax, color_scheme=colors)

    plt.tight_layout()
    return fig


def plot_motif_logo(matrix, title="Motif", logo_type="cwm"):
    """Sequence logo from CWM or PFM matrix (4, L)."""
    fig, ax = plt.subplots(figsize=(max(4, matrix.shape[1] * 0.35), 2.5))
    fig.patch.set_facecolor(_COLORS['bg'])
    _style_ax(ax, title, ylabel="CWM Score" if logo_type == "cwm" else "Frequency")
    df = pd.DataFrame(matrix.T, columns=['A', 'C', 'G', 'T'])
    colors = {b: _COLORS[b] for b in 'ACGT'}
    logomaker.Logo(df, ax=ax, color_scheme=colors)
    plt.tight_layout()
    plt.show()
    return fig


def plot_all_motifs(motifs_list, n_cols=4, logo_type="cwm", title="Discovered Motifs"):
    """Grid of all discovered motifs."""
    n = len(motifs_list)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)
    fig.patch.set_facecolor(_COLORS['bg'])
    fig.suptitle(title, color=_COLORS['text'], fontsize=14, fontweight='bold', y=1.01)

    colors = {b: _COLORS[b] for b in 'ACGT'}
    for idx, m in enumerate(motifs_list):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        _style_ax(ax)
        mat = m["cwm"] if logo_type == "cwm" else m["pfm"]
        df = pd.DataFrame(mat.T, columns=['A', 'C', 'G', 'T'])
        logomaker.Logo(df, ax=ax, color_scheme=colors)
        label = f'{m["name"]} (n={m["n_seqlets"]})'
        ax.set_title(label, color=_COLORS['text'], fontsize=9, fontweight='bold')
        ax.set_xticks([])

    for i in range(idx + 1, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


def plot_motif_match(query_pwm, hit, title="JASPAR Match"):
    """Side-by-side query PWM vs JASPAR match."""
    fig, axes = plt.subplots(2, 1, figsize=(max(6, query_pwm.shape[1] * 0.4), 4.5))
    fig.patch.set_facecolor(_COLORS['bg'])
    colors = {b: _COLORS[b] for b in 'ACGT'}
    for i, (mat, lbl) in enumerate([(query_pwm, "Query Motif"),
                                     (hit["pwm"], f'{hit["name"]} ({hit["id"]}) r={hit["correlation"]:.3f}')]):
        _style_ax(axes[i], lbl)
        df = pd.DataFrame(mat.T, columns=['A', 'C', 'G', 'T'])
        logomaker.Logo(df, ax=axes[i], color_scheme=colors)
    plt.tight_layout()
    plt.show()
    return fig


def plot_attribution_heatmap(attributions, title="Attribution Heatmap", max_seqs=100):
    """Heatmap of per-base contributions across sequences."""
    # Sum across channels -> (N, L)
    summed = attributions[:max_seqs].sum(axis=1)
    fig, ax = plt.subplots(figsize=(14, max(3, len(summed) * 0.08)))
    fig.patch.set_facecolor(_COLORS['bg'])
    _style_ax(ax, title, xlabel="Position", ylabel="Sequence")
    vmax = np.percentile(np.abs(summed), 99)
    ax.imshow(summed, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
              interpolation='nearest')
    plt.tight_layout()
    plt.show()
    return fig


def plot_motif_with_k4(motif_dict, title=None, figsize=None):
    """Two-panel figure: DNA CWM logo (top) + K4 attribution track (bottom).

    Falls back to a single DNA logo panel if 'k4_profile' key is absent.
    The K4 track x-axis is centered on 0 (= motif center).
    """
    has_k4 = 'k4_profile' in motif_dict
    mat = motif_dict.get('cwm', motif_dict.get('pfm'))
    if title is None:
        title = motif_dict.get('name', 'Motif')

    n_panels = 2 if has_k4 else 1
    if figsize is None:
        w = max(4, mat.shape[1] * 0.35)
        figsize = (w, 2.5 + 1.5 * has_k4)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize,
                             gridspec_kw={'height_ratios': [2, 1] if has_k4 else [1]},
                             squeeze=False)
    fig.patch.set_facecolor(_COLORS['bg'])

    # Top: DNA logo
    ax_dna = axes[0][0]
    _style_ax(ax_dna, title, ylabel='CWM Score')
    df = pd.DataFrame(mat.T, columns=['A', 'C', 'G', 'T'])
    logomaker.Logo(df, ax=ax_dna, color_scheme={b: _COLORS[b] for b in 'ACGT'})
    ax_dna.set_xticks([])

    # Bottom: K4 importance
    if has_k4:
        ax_k4 = axes[1][0]
        _style_ax(ax_k4, '', xlabel='Position relative to motif center', ylabel='K4 Importance')
        profile = motif_dict['k4_profile']
        std     = motif_dict.get('k4_std', np.zeros_like(profile))
        x = np.arange(len(profile)) - len(profile) // 2
        k4_color = '#00897B'
        ax_k4.plot(x, profile, color=k4_color, lw=1.5)
        ax_k4.fill_between(x, profile - std, profile + std, color=k4_color, alpha=0.2)
        ax_k4.axhline(0, color=_COLORS['grid'], lw=0.8, ls='--')
        ax_k4.axvline(0, color=_COLORS['grid'], lw=0.8, ls='--')

    plt.tight_layout()
    plt.show()
    return fig


def plot_all_motifs_k4(motifs_list, n_cols=4, logo_type='cwm',
                       title='Discovered Motifs'):
    """Grid of two-panel motif figures (DNA logo + K4 track per motif).

    Falls back to single-panel logos if no motif has a 'k4_profile' key.
    """
    if not motifs_list:
        print('No motifs to plot.')
        return

    has_k4 = any('k4_profile' in m for m in motifs_list)
    n = len(motifs_list)
    n_rows = math.ceil(n / n_cols)
    panels = 2 if has_k4 else 1  # sub-rows per motif row

    fig, axes = plt.subplots(
        n_rows * panels, n_cols,
        figsize=(4 * n_cols, (2.5 + 1.5 * has_k4) * n_rows),
        gridspec_kw={'height_ratios': [2, 1] * n_rows} if has_k4 else {},
        squeeze=False
    )
    fig.patch.set_facecolor(_COLORS['bg'])
    fig.suptitle(title, color=_COLORS['text'], fontsize=14, fontweight='bold', y=1.01)

    dna_colors = {b: _COLORS[b] for b in 'ACGT'}
    k4_color = '#00897B'

    for idx, m in enumerate(motifs_list):
        col      = idx % n_cols
        logo_row = (idx // n_cols) * panels

        # DNA logo
        ax_dna = axes[logo_row][col]
        _style_ax(ax_dna)
        mat = m['cwm'] if logo_type == 'cwm' else m['pfm']
        logomaker.Logo(pd.DataFrame(mat.T, columns=['A', 'C', 'G', 'T']),
                       ax=ax_dna, color_scheme=dna_colors)
        ax_dna.set_title(f'{m["name"]} (n={m["n_seqlets"]})',
                         color=_COLORS['text'], fontsize=9, fontweight='bold')
        ax_dna.set_xticks([])

        # K4 track
        if has_k4:
            ax_k4 = axes[logo_row + 1][col]
            _style_ax(ax_k4, ylabel='K4')
            if 'k4_profile' in m:
                profile = m['k4_profile']
                std     = m.get('k4_std', np.zeros_like(profile))
                x = np.arange(len(profile)) - len(profile) // 2
                ax_k4.plot(x, profile, color=k4_color, lw=1.2)
                ax_k4.fill_between(x, profile - std, profile + std,
                                   color=k4_color, alpha=0.2)
                ax_k4.axhline(0, color=_COLORS['grid'], lw=0.6, ls='--')
            ax_k4.set_xticks([])

    # Hide unused axes
    for i in range(n, n_rows * n_cols):
        col      = i % n_cols
        logo_row = (i // n_cols) * panels
        axes[logo_row][col].set_visible(False)
        if has_k4:
            axes[logo_row + 1][col].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


# ─── Section 6: Helper Utilities ─────────────────────────────────────────────

def visualize_binding_profile(y, y2=None, span=None, title="Binding Profile",
                              label1="true", label2="pred", alpha=0.7):
    """Overlay true/predicted profiles."""
    if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
    y = y.squeeze()
    if y2 is not None:
        if isinstance(y2, torch.Tensor): y2 = y2.detach().cpu().numpy()
        y2 = y2.squeeze()
    if span:
        y = y[span[0]:span[1]]
        if y2 is not None: y2 = y2[span[0]:span[1]]
    x = np.arange(len(y)) + (span[0] if span else 0)

    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor(_COLORS['bg'])
    _style_ax(ax, title, xlabel="Position", ylabel="Signal")
    ax.fill_between(x, y, alpha=0.4, color=_COLORS['accent2'], label=label1)
    if y2 is not None:
        ax.plot(x, y2, color=_COLORS['accent'], lw=1.5, alpha=alpha, label=label2)
        ax.legend(facecolor=_COLORS['card'], edgecolor=_COLORS['grid'], labelcolor=_COLORS['text'])
    plt.tight_layout()
    plt.show()
    return fig


def visualize_sequence_logo(seq_tensor, span=None, title="Sequence"):
    """Visualize a one-hot (4,L) tensor as a sequence logo."""
    if isinstance(seq_tensor, torch.Tensor):
        seq_tensor = seq_tensor.detach().cpu().numpy()
    if seq_tensor.ndim == 3: seq_tensor = seq_tensor[0]
    if span: seq_tensor = seq_tensor[:, span[0]:span[1]]
    fig, ax = plt.subplots(figsize=(max(6, seq_tensor.shape[1] * 0.15), 2.5))
    fig.patch.set_facecolor(_COLORS['bg'])
    _style_ax(ax, title)
    df = pd.DataFrame(seq_tensor.T, columns=['A', 'C', 'G', 'T'])
    colors = {b: _COLORS[b] for b in 'ACGT'}
    logomaker.Logo(df, ax=ax, color_scheme=colors)
    plt.tight_layout()
    plt.show()
    return fig


__all__ = [
    "compute_input_gradient", "compute_attributions_batch",
    "run_modisco", "extract_motifs_from_patterns", "discover_motifs",
    "extract_k4_seqlet_profiles",
    "extract_filter_pwms", "get_consensus",
    "search_jaspar",
    "plot_attribution_map", "plot_motif_logo", "plot_all_motifs",
    "plot_motif_match", "plot_attribution_heatmap",
    "plot_motif_with_k4", "plot_all_motifs_k4",
    "visualize_binding_profile", "visualize_sequence_logo",
]
