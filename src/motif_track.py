"""
MotifTracker: Stateful motif discovery pipeline for BPNet-style models.

Wraps DeepLIFT attribution → TF-MoDISco → JASPAR matching into a single
object so intermediate results can be reused without recomputation.

Works with both BPNetModel and BPNetK4Model.
"""

import os
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from typing import Optional, List, Union

from cnn import BPNetK4Model
from cnn_track import (
    compute_attributions_batch,
    run_modisco,
    extract_motifs_from_patterns,
    search_jaspar,
    _COLORS,
    _style_ax,
)
from modiscolite import io


class MotifTracker:
    """
    Stateful motif discovery pipeline for BPNet-style models.

    Usage::

        tracker = MotifTracker(model, device='mps')
        tracker.run(loader, min_counts=model.min_profile,
                    method='deeplift', head='profile',
                    save_path='results/modisco.h5')
        tracker.match_jaspar(db_path='../data/JASPAR2024_CORE_vertebrates.txt')
        tracker.plot_motifs()
        tracker.summary()
        tracker.save('results/tracker.pkl')
    """

    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.is_k4 = isinstance(model, BPNetK4Model)

        # Populated by compute_attributions()
        self.ohs: Optional[np.ndarray] = None        # (N, 4, L)
        self.attrs: Optional[np.ndarray] = None      # (N, 4or5, L)
        self.dna_attrs: Optional[np.ndarray] = None  # (N, 4, L)
        self.k4_attrs: Optional[np.ndarray] = None   # (N, 1, L) or None

        # Populated by run_modisco()
        self.pos_patterns = None
        self.neg_patterns = None
        self.motifs: List[dict] = []

    # ── High-level entry point ────────────────────────────────────────────

    def run(self, loader, n_seqs: int = 500, min_counts: float = 7000,
            method: str = 'deeplift', head: str = 'profile',
            window_size: int = 21, max_seqlets: int = 20000,
            n_leiden: int = 50, verbose: bool = True,
            save_path: Optional[str] = None):
        """All-in-one pipeline: attributions → MoDISco → motif dicts."""
        self.compute_attributions(loader, n_seqs, min_counts, method, head)
        self.run_modisco(window_size, max_seqlets, n_leiden, verbose, save_path)
        return self

    # ── Step 1: Attributions ──────────────────────────────────────────────

    def compute_attributions(self, loader, n_seqs: int = 500,
                              min_counts: float = 7000,
                              method: str = 'deeplift',
                              head: str = 'profile'):
        """Compute attributions for high-signal sequences."""
        ohs, attrs = compute_attributions_batch(
            self.model, loader, self.device, n_seqs, min_counts,
            method=method, head=head
        )
        self.ohs = ohs
        self.attrs = attrs
        self.dna_attrs = attrs[:, :4, :]
        self.k4_attrs = (attrs[:, 4:, :]
                         if (self.is_k4 and attrs.shape[1] == 5)
                         else None)
        return self

    # ── Step 2: MoDISco ───────────────────────────────────────────────────

    def run_modisco(self, window_size: int = 21, max_seqlets: int = 20000,
                    n_leiden: int = 50, verbose: bool = True,
                    save_path: Optional[str] = None):
        """Run TF-MoDISco on DNA attributions."""
        if self.dna_attrs is None:
            raise RuntimeError("Call compute_attributions() first.")

        self.pos_patterns, self.neg_patterns = run_modisco(
            self.ohs, self.dna_attrs, window_size, max_seqlets,
            n_leiden, verbose
        )

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            io.save_hdf5(save_path, self.pos_patterns, self.neg_patterns,
                         window_size=window_size)
            if verbose:
                print(f"Saved MoDISco results to {save_path}")

        self.motifs = extract_motifs_from_patterns(self.pos_patterns or [], 'pos')
        self.motifs += extract_motifs_from_patterns(self.neg_patterns or [], 'neg')
        return self

    # ── Step 3: JASPAR ────────────────────────────────────────────────────

    def match_jaspar(self, db_path: Optional[str] = None,
                     taxon: str = 'vertebrates',
                     species: str = 'Mus musculus',
                     top_n: int = 5):
        """Match each motif CWM against JASPAR. Annotates motifs in-place."""
        if not self.motifs:
            raise RuntimeError("Call run_modisco() first.")
        for m in self.motifs:
            m['jaspar'] = search_jaspar(m['cwm'], taxon=taxon, species=species,
                                        top_n=top_n, db_path=db_path)
        return self

    # ── Visualization ─────────────────────────────────────────────────────

    def plot_motifs(self, n_cols: int = 4, logo_type: str = 'cwm',
                    title: str = 'Discovered Motifs'):
        """Grid of CWM logos with optional JASPAR annotation."""
        if not self.motifs:
            print("No motifs to plot.")
            return None

        n = len(self.motifs)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 2.5 * n_rows),
                                 squeeze=False)
        fig.patch.set_facecolor(_COLORS['bg'])
        fig.suptitle(title, color=_COLORS['text'], fontsize=14,
                     fontweight='bold', y=1.01)
        colors = {b: _COLORS[b] for b in 'ACGT'}

        for idx, m in enumerate(self.motifs):
            r, c = divmod(idx, n_cols)
            ax = axes[r][c]
            _style_ax(ax)
            mat = m['cwm'] if logo_type == 'cwm' else m['pfm']
            logomaker.Logo(pd.DataFrame(mat.T, columns=['A', 'C', 'G', 'T']),
                           ax=ax, color_scheme=colors)
            jaspar_label = ''
            if m.get('jaspar'):
                jaspar_label = f"\n{m['jaspar'][0]['name']}"
            ax.set_title(f"{m['name']} (n={m['n_seqlets']}){jaspar_label}",
                         color=_COLORS['text'], fontsize=8, fontweight='bold')
            ax.set_xticks([])

        for i in range(n, n_rows * n_cols):
            r, c = divmod(i, n_cols)
            axes[r][c].set_visible(False)

        plt.tight_layout()
        plt.show()
        return fig

    def plot_motif(self, idx: int, logo_type: str = 'cwm'):
        """Single motif logo with top JASPAR matches."""
        m = self.motifs[idx]
        mat = m['cwm'] if logo_type == 'cwm' else m['pfm']
        fig, ax = plt.subplots(figsize=(max(4, mat.shape[1] * 0.4), 3))
        fig.patch.set_facecolor(_COLORS['bg'])
        _style_ax(ax, m['name'], ylabel='CWM Score')
        colors = {b: _COLORS[b] for b in 'ACGT'}
        logomaker.Logo(pd.DataFrame(mat.T, columns=['A', 'C', 'G', 'T']),
                       ax=ax, color_scheme=colors)
        if m.get('jaspar'):
            lines = [f"{h['name']} ({h['id']})  r={h['correlation']:.3f}"
                     for h in m['jaspar'][:3]]
            ax.set_xlabel('\n'.join(lines), fontsize=8, color='#555555')
        plt.tight_layout()
        plt.show()
        return fig

    def summary(self, top_n_jaspar: int = 1):
        """Print a summary table of all discovered motifs."""
        header = f"{'Motif':<14} {'Seqlets':>8}  {'JASPAR hit':<24} {'r':>6}"
        print(header)
        print('─' * len(header))
        for m in self.motifs:
            hit, r_val = '', ''
            if m.get('jaspar') and m['jaspar']:
                best = m['jaspar'][0]
                hit = best['name'][:22]
                r_val = f"{best['correlation']:.3f}"
            print(f"{m['name']:<14} {m['n_seqlets']:>8}  {hit:<24} {r_val:>6}")

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        """Pickle this tracker (model excluded)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        model_ref, self.model = self.model, None
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.model = model_ref
        print(f"Saved MotifTracker to {path}  (re-attach model after loading)")

    @classmethod
    def load(cls, path: str, model=None):
        """Load a pickled MotifTracker. Pass model= to re-attach."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if model is not None:
            obj.model = model
        return obj
