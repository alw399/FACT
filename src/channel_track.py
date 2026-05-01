"""
ChannelTracker: K4 signal channel importance analysis for BPNetK4Model.

Requires a fitted MotifTracker as input. Analyses:
  1. Per-motif mean K4 attribution profiles (centered on seqlets)
  2. Per-seqlet K4 tracks (raw, for heatmaps showing heterogeneity)
  3. Joint DNA × K4 importance analysis per seqlet
  4. K4 dependency score per motif (Pearson between DNA and K4 attr strength)
  5. Global K4 attribution map (baseline across all sequences)
"""

import os
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logomaker
from scipy.stats import pearsonr
from typing import Optional, Dict, Union

from cnn_track import _COLORS, _style_ax

_K4_COLOR  = '#00897B'   # teal
_DNA_COLOR = '#1976d2'   # blue


class ChannelTracker:
    """
    K4 signal channel importance analysis, tied to a MotifTracker.

    Usage::

        ch = ChannelTracker(tracker, k4_window=200)
        ch.compute_k4_profiles()
        ch.compute_joint_analysis()
        ch.compute_k4_dependency()
        ch.compute_global_k4()

        ch.plot_k4_profiles()
        ch.plot_k4_dependency()
        ch.plot_seqlet_heatmap('pos_0')
        ch.plot_joint_attribution('pos_0')
        ch.plot_motif_k4_summary('pos_0')
        ch.plot_global_k4()
    """

    def __init__(self, tracker, k4_window: int = 200):
        if not tracker.is_k4 or tracker.k4_attrs is None:
            raise ValueError(
                "ChannelTracker requires a MotifTracker fitted on a "
                "BPNetK4Model. Call tracker.compute_attributions() first."
            )
        self.tracker = tracker
        self.k4_window = k4_window

        # Populated by compute_k4_profiles()
        self.k4_profiles: list = []                           # [{k4_profile, k4_std}, ...] per motif
        self.seqlet_k4_tracks: Dict[str, np.ndarray] = {}    # {name: (n_seqlets, W)}

        # Populated by compute_joint_analysis()
        self.seqlet_dna_scores: Dict[str, np.ndarray] = {}   # {name: (n_seqlets,)}
        self.seqlet_k4_strength: Dict[str, np.ndarray] = {}  # {name: (n_seqlets,)}

        # Populated by compute_k4_dependency()
        self.k4_dep_scores: Dict[str, float] = {}

        # Populated by compute_global_k4()
        self.global_k4_mean: Optional[np.ndarray] = None
        self.global_k4_std: Optional[np.ndarray] = None

    # ── Internal helpers ──────────────────────────────────────────────────

    def _motif_by_name(self, name_or_idx: Union[str, int]):
        motifs = self.tracker.motifs
        if isinstance(name_or_idx, int):
            return name_or_idx, motifs[name_or_idx]
        for i, m in enumerate(motifs):
            if m['name'] == name_or_idx:
                return i, m
        raise KeyError(f"Motif '{name_or_idx}' not found.")

    def _extract_seqlet_windows(self, pattern, window: int) -> np.ndarray:
        """Extract K4 attribution windows (n_seqlets, window) for a pattern."""
        k4 = self.tracker.k4_attrs    # (N, 1, L)
        half = window // 2
        L = k4.shape[2]
        tracks = []
        for s in pattern.seqlets:
            center = (s.start + s.end) // 2
            lo, hi = center - half, center + half
            lo_c, hi_c = max(0, lo), min(L, hi)
            track = k4[s.example_idx, 0, lo_c:hi_c].copy()
            pad_l, pad_r = lo_c - lo, hi - hi_c
            if pad_l > 0 or pad_r > 0:
                track = np.pad(track, (pad_l, pad_r), mode='constant')
            if len(track) != window:
                continue
            if s.is_revcomp:
                track = track[::-1]
            tracks.append(track)
        return np.stack(tracks) if tracks else np.zeros((0, window))

    def _all_patterns(self):
        return (self.tracker.pos_patterns or []) + (self.tracker.neg_patterns or [])

    # ── Analysis ──────────────────────────────────────────────────────────

    def compute_k4_profiles(self, window: Optional[int] = None):
        """
        Compute mean ± std K4 attribution profile per motif.
        Also stores raw per-seqlet tracks for heatmaps.
        """
        window = window or self.k4_window
        self.k4_profiles = []
        self.seqlet_k4_tracks = {}
        for pattern, m in zip(self._all_patterns(), self.tracker.motifs):
            tracks = self._extract_seqlet_windows(pattern, window)
            self.seqlet_k4_tracks[m['name']] = tracks
            if len(tracks) > 0:
                self.k4_profiles.append({
                    'k4_profile': tracks.mean(axis=0),
                    'k4_std':     tracks.std(axis=0),
                })
            else:
                self.k4_profiles.append({
                    'k4_profile': np.zeros(window),
                    'k4_std':     np.zeros(window),
                })
        return self

    def compute_joint_analysis(self, window: Optional[int] = None):
        """
        For each motif's seqlets, compute:
          - DNA attribution strength: sum |dna_attrs| at seqlet span
          - K4 importance:           mean |k4_attrs| in center window
        """
        window = window or self.k4_window
        dna = self.tracker.dna_attrs    # (N, 4, L)
        k4  = self.tracker.k4_attrs     # (N, 1, L)
        L   = k4.shape[2]
        half = window // 2

        for pattern, m in zip(self._all_patterns(), self.tracker.motifs):
            dna_scores, k4_scores = [], []
            for s in pattern.seqlets:
                n = s.example_idx
                dna_str = float(np.abs(dna[n, :, s.start:s.end]).sum())
                center  = (s.start + s.end) // 2
                lo, hi  = max(0, center - half), min(L, center + half)
                k4_str  = float(np.abs(k4[n, 0, lo:hi]).mean())
                dna_scores.append(dna_str)
                k4_scores.append(k4_str)
            self.seqlet_dna_scores[m['name']]  = np.array(dna_scores)
            self.seqlet_k4_strength[m['name']] = np.array(k4_scores)
        return self

    def compute_k4_dependency(self):
        """
        Pearson(DNA attr strength, K4 importance) per seqlet for each motif.
        High positive score → DNA and K4 are synergistic at this motif.
        Requires compute_joint_analysis() first.
        """
        if not self.seqlet_dna_scores:
            raise RuntimeError("Call compute_joint_analysis() first.")
        self.k4_dep_scores = {}
        for name in self.seqlet_dna_scores:
            x = self.seqlet_dna_scores[name]
            y = self.seqlet_k4_strength[name]
            if len(x) >= 3:
                r, _ = pearsonr(x, y)
                self.k4_dep_scores[name] = float(r) if not np.isnan(r) else 0.0
            else:
                self.k4_dep_scores[name] = 0.0
        return self

    def compute_global_k4(self):
        """
        Mean and std K4 attribution across all N sequences.
        Baseline: is K4 globally important or only near specific motifs?
        """
        k4 = self.tracker.k4_attrs[:, 0, :]   # (N, L)
        self.global_k4_mean = k4.mean(axis=0)
        self.global_k4_std  = k4.std(axis=0)
        return self

    # ── Visualizations ────────────────────────────────────────────────────

    def plot_k4_profiles(self, n_cols: int = 4,
                          title: str = 'K4 Signal Importance by Motif'):
        """Grid of mean K4 attribution profiles per motif."""
        if not self.k4_profiles:
            raise RuntimeError("Call compute_k4_profiles() first.")

        motifs = self.tracker.motifs
        order = sorted(range(len(motifs)),
                       key=lambda i: self.k4_dep_scores.get(motifs[i]['name'], 0),
                       reverse=True)

        n = len(motifs)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 2.5 * n_rows),
                                 squeeze=False, 
                                 sharey=True)
        fig.patch.set_facecolor(_COLORS['bg'])
        fig.suptitle(title, fontsize=14,
                     color=_COLORS['text'], y=1.01)

        for plot_pos, mi in enumerate(order):
            r, c = divmod(plot_pos, n_cols)
            ax = axes[r][c]
            m = motifs[mi]
            prof = self.k4_profiles[mi]
            _style_ax(ax, ylabel='K4 attr')
            x = np.arange(len(prof['k4_profile'])) - len(prof['k4_profile']) // 2
            ax.plot(x, prof['k4_profile'], color=_K4_COLOR, lw=1.5)
            ax.fill_between(x,
                            prof['k4_profile'] - prof['k4_std'],
                            prof['k4_profile'] + prof['k4_std'],
                            color=_K4_COLOR, alpha=0.2)
            ax.axhline(0, color=_COLORS['grid'], lw=0.8, ls='--')
            ax.axvline(0, color=_COLORS['grid'], lw=0.8, ls='--')
            dep = self.k4_dep_scores.get(m['name'])
            dep_str = f'  dep={dep:.2f}' if dep is not None else ''
            ax.set_title(f"{m['name']} (n={m['n_seqlets']}){dep_str}",
                         fontsize=8, color=_COLORS['text'])

        for i in range(n, n_rows * n_cols):
            r, c = divmod(i, n_cols)
            axes[r][c].set_visible(False)

        plt.tight_layout()
        plt.show()
        return fig

    def plot_seqlet_heatmap(self, name_or_idx: Union[str, int],
                             normalize: bool = False,
                             sort_by: str = 'k4_strength',
                             title: Optional[str] = None):
        """
        Heatmap of individual seqlet K4 tracks (rows = seqlets).
        Shows heterogeneity hidden in the mean profile.

        Args:
            normalize: z-score rows to compare shape regardless of magnitude.
            sort_by:   'k4_strength' | 'dna_strength' | 'none'
        """
        if not self.seqlet_k4_tracks:
            raise RuntimeError("Call compute_k4_profiles() first.")

        motif_idx, m = self._motif_by_name(name_or_idx)
        tracks = self.seqlet_k4_tracks[m['name']].copy()  # (n_seqlets, W)
        if len(tracks) == 0:
            print(f"No seqlet tracks for {m['name']}.")
            return None

        # Sort rows
        if sort_by == 'k4_strength' and m['name'] in self.seqlet_k4_strength:
            order = np.argsort(self.seqlet_k4_strength[m['name']])[::-1]
        elif sort_by == 'dna_strength' and m['name'] in self.seqlet_dna_scores:
            order = np.argsort(self.seqlet_dna_scores[m['name']])[::-1]
        else:
            order = np.arange(len(tracks))
        tracks = tracks[order]

        if normalize:
            std = tracks.std(axis=1, keepdims=True)
            std[std == 0] = 1
            tracks = (tracks - tracks.mean(axis=1, keepdims=True)) / std

        x = np.arange(tracks.shape[1]) - tracks.shape[1] // 2
        vmax = np.percentile(np.abs(tracks), 98)

        fig = plt.figure(figsize=(8, max(3, min(12, len(tracks) * 0.12 + 2))))
        fig.patch.set_facecolor(_COLORS['bg'])
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)

        ax_heat = fig.add_subplot(gs[0])
        im = ax_heat.imshow(tracks, aspect='auto', cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax,
                            extent=[x[0], x[-1], len(tracks), 0])
        plt.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02,
                     label='K4 attr (z-scored)' if normalize else 'K4 attr')
        ax_heat.set_ylabel('Seqlet', color=_COLORS['text'], fontsize=9)
        ax_heat.set_title(
            title or f"{m['name']}  —  K4 seqlet heatmap  (n={len(tracks)})",
            fontsize=11, fontweight='bold', color=_COLORS['text'])
        ax_heat.axvline(0, color='black', lw=0.8, ls='--')
        ax_heat.set_xticks([])

        ax_mean = fig.add_subplot(gs[1])
        _style_ax(ax_mean, xlabel='Position relative to motif center', ylabel='Mean K4')
        prof = self.k4_profiles[motif_idx]
        ax_mean.plot(x, prof['k4_profile'], color=_K4_COLOR, lw=1.5)
        ax_mean.fill_between(x,
                             prof['k4_profile'] - prof['k4_std'],
                             prof['k4_profile'] + prof['k4_std'],
                             color=_K4_COLOR, alpha=0.2)
        ax_mean.axhline(0, color=_COLORS['grid'], lw=0.8, ls='--')
        ax_mean.axvline(0, color=_COLORS['grid'], lw=0.8, ls='--')
        plt.show()
        return fig

    def plot_k4_dependency(self, title: str = 'K4 Dependency Score by Motif'):
        """Horizontal bar chart ranking motifs by Pearson(DNA_strength, K4_strength)."""
        if not self.k4_dep_scores:
            raise RuntimeError("Call compute_k4_dependency() first.")

        motifs = self.tracker.motifs
        items = [(m['name'], self.k4_dep_scores.get(m['name'], 0), m['n_seqlets'])
                 for m in motifs if m['name'] in self.k4_dep_scores]
        items.sort(key=lambda x: x[1])

        names   = [i[0] for i in items]
        scores  = [i[1] for i in items]
        seqlets = [i[2] for i in items]
        bar_colors = [_K4_COLOR if s > 0.1 else '#cccccc' for s in scores]

        fig, ax = plt.subplots(figsize=(6, max(3, len(names) * 0.4)))
        fig.patch.set_facecolor(_COLORS['bg'])
        _style_ax(ax, title, xlabel='Pearson r  (DNA attr strength vs K4 importance)')
        ax.barh(range(len(names)), scores, color=bar_colors, edgecolor='none')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([f"{n}  (n={ns})" for n, ns in zip(names, seqlets)],
                           fontsize=8, color=_COLORS['text'])
        ax.axvline(0, color=_COLORS['text'], lw=0.8, ls='--')
        ax.set_xlim(-1, 1)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_joint_attribution(self, name_or_idx: Union[str, int],
                                title: Optional[str] = None):
        """
        Scatter plot: DNA attr strength vs K4 importance per seqlet.
        Shows whether the two channels are correlated at the instance level.
        """
        if not self.seqlet_dna_scores:
            raise RuntimeError("Call compute_joint_analysis() first.")

        _, m = self._motif_by_name(name_or_idx)
        x = self.seqlet_dna_scores[m['name']]
        y = self.seqlet_k4_strength[m['name']]

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(_COLORS['bg'])
        _style_ax(ax, title or f"{m['name']} — DNA × K4 per seqlet",
                  xlabel='DNA attribution strength', ylabel='K4 importance')
        ax.scatter(x, y, alpha=0.5, s=18, color=_K4_COLOR, edgecolors='none')

        if len(x) >= 3:
            r, _ = pearsonr(x, y)
            ax.text(0.05, 0.93, f'r = {r:.3f}', transform=ax.transAxes,
                    fontsize=9, fontweight='bold', color=_COLORS['text'],
                    bbox=dict(facecolor='white', edgecolor='#cccccc',
                              boxstyle='round,pad=0.3'))

            # Regression line
            m_fit, b_fit = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 100)
            ax.plot(xr, m_fit * xr + b_fit, color=_COLORS['accent'],
                    lw=1.2, ls='--')

        plt.tight_layout()
        plt.show()
        return fig

    def plot_motif_k4_summary(self, name_or_idx: Union[str, int],
                               title: Optional[str] = None):
        """
        Three-panel summary for a single motif:
          1. DNA CWM logo
          2. Mean K4 profile (± std)
          3. Joint scatter (DNA attr strength vs K4 importance per seqlet)
        """
        if not self.k4_profiles:
            raise RuntimeError("Call compute_k4_profiles() first.")

        motif_idx, m = self._motif_by_name(name_or_idx)
        mat = m['cwm']
        prof = self.k4_profiles[motif_idx]

        fig = plt.figure(figsize=(12, 3.5))
        fig.patch.set_facecolor(_COLORS['bg'])
        fig.suptitle(title or m['name'], fontsize=13, fontweight='bold',
                     color=_COLORS['text'])
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # Panel 1: DNA logo
        ax_dna = fig.add_subplot(gs[0])
        _style_ax(ax_dna, 'DNA Motif (CWM)', ylabel='Score')
        colors = {b: _COLORS[b] for b in 'ACGT'}
        logomaker.Logo(pd.DataFrame(mat.T, columns=['A', 'C', 'G', 'T']),
                       ax=ax_dna, color_scheme=colors)
        ax_dna.set_xticks([])

        # Panel 2: K4 profile
        ax_k4 = fig.add_subplot(gs[1])
        _style_ax(ax_k4, 'K4 Attribution Profile',
                  xlabel='Position from motif center', ylabel='K4 attr')
        x = np.arange(len(prof['k4_profile'])) - len(prof['k4_profile']) // 2
        ax_k4.plot(x, prof['k4_profile'], color=_K4_COLOR, lw=1.5)
        ax_k4.fill_between(x,
                           prof['k4_profile'] - prof['k4_std'],
                           prof['k4_profile'] + prof['k4_std'],
                           color=_K4_COLOR, alpha=0.2)
        ax_k4.axhline(0, color=_COLORS['grid'], lw=0.8, ls='--')
        ax_k4.axvline(0, color=_COLORS['grid'], lw=0.8, ls='--')

        # Panel 3: Joint scatter
        ax_sc = fig.add_subplot(gs[2])
        if m['name'] in self.seqlet_dna_scores:
            xd = self.seqlet_dna_scores[m['name']]
            yk = self.seqlet_k4_strength[m['name']]
            _style_ax(ax_sc, 'Joint Attribution',
                      xlabel='DNA attr strength', ylabel='K4 importance')
            ax_sc.scatter(xd, yk, alpha=0.5, s=15,
                          color=_K4_COLOR, edgecolors='none')
            if len(xd) >= 3:
                r, _ = pearsonr(xd, yk)
                ax_sc.text(0.05, 0.93, f'r = {r:.3f}',
                           transform=ax_sc.transAxes, fontsize=9,
                           fontweight='bold', color=_COLORS['text'],
                           bbox=dict(facecolor='white', edgecolor='#cccccc',
                                     boxstyle='round,pad=0.3'))
                mf, bf = np.polyfit(xd, yk, 1)
                xr = np.linspace(xd.min(), xd.max(), 100)
                ax_sc.plot(xr, mf * xr + bf, color=_COLORS['accent'],
                           lw=1.2, ls='--')
        else:
            _style_ax(ax_sc, 'Joint Attribution (run compute_joint_analysis)')

        plt.show()
        return fig

    def plot_global_k4(self, title: str = 'Global K4 Attribution Profile'):
        """
        Mean K4 attribution across all N sequences.
        Baseline showing whether K4 is globally important or motif-specific.
        """
        if self.global_k4_mean is None:
            raise RuntimeError("Call compute_global_k4() first.")

        L = len(self.global_k4_mean)
        x = np.arange(L)

        fig, ax = plt.subplots(figsize=(12, 2.5))
        fig.patch.set_facecolor(_COLORS['bg'])
        _style_ax(ax, title, xlabel='Position along sequence', ylabel='K4 attr')
        ax.plot(x, self.global_k4_mean, color=_K4_COLOR, lw=1.2, alpha=0.9)
        ax.fill_between(x,
                        self.global_k4_mean - self.global_k4_std,
                        self.global_k4_mean + self.global_k4_std,
                        color=_K4_COLOR, alpha=0.15)
        ax.axhline(0, color=_COLORS['grid'], lw=0.8, ls='--')
        plt.tight_layout()
        plt.show()
        return fig

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        """Pickle this ChannelTracker (tracker reference excluded)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tracker_ref, self.tracker = self.tracker, None
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.tracker = tracker_ref
        print(f"Saved ChannelTracker to {path}  (re-attach tracker after loading)")

    @classmethod
    def load(cls, path: str, tracker=None):
        """Load a pickled ChannelTracker. Pass tracker= to re-attach."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if tracker is not None:
            obj.tracker = tracker
        return obj
