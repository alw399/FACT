"""
Data utilities for CUT&RUN prediction from mm10 sequence.

This module provides:
- One-hot encoding for DNA sequences (A/C/G/T -> 4 channels)
- A PyTorch Dataset that pairs sequence windows with CUT&RUN signal
  extracted from a bigWig file.

The code is written so that you can plug in any mm10 FASTA file path
and any bigWig file with CUT&RUN signal over the same genome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset
import pyfaidx  # for FASTA access
import pyBigWig  # for bigWig access

NUC_TO_IDX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


def one_hot_encode_sequence(seq: str) -> np.ndarray:
    """
    One-hot encode a DNA sequence string into shape (4, L).

    Unknown bases (e.g. N) are encoded as all zeros.
    """
    seq = seq.upper()
    L = len(seq)
    arr = np.zeros((4, L), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = NUC_TO_IDX.get(base)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


@dataclass
class GenomicInterval:
    """Convenience container describing a genomic window."""

    chrom: str
    start: int
    end: int


def make_uniform_intervals(
    bed: pd.DataFrame,
    window_size: int,
) -> List[GenomicInterval]:
    """
    Convenience function: tile a chromosome with uniform windows.

    Parameters
    ----------
    bed
        BED file containing genomic intervals. Assume TSS is at start 
    window_size
        Size of each window in bases.
    """

    intervals: List[GenomicInterval] = []

    bed['band_start'] = bed['start'] - window_size 
    bed['band_end'] = bed['start'] + window_size 
    
    for chrom in bed['#chrom'].unique():
        chrom_bed = bed[bed['#chrom'] == chrom]
        intervals.extend([
            GenomicInterval(chrom=chrom, start=start, end=end) 
            for start, end in zip(chrom_bed['band_start'], chrom_bed['band_end'])
        ])
    return intervals


class GenomicDatasetBase(Dataset):
    """
    Base class for datasets mapping DNA sequence to BigWig signals.
    
    Handles FASTA and BED loading, interval generation around BED regions,
    and filtering based on signal presence.
    """
    def __init__(
        self,
        fasta_path: str,
        bed_path: str,
        reference_bw_path: str,
        signal_bins: Optional[int] = None,
        window_size: int = 1000,
        binarize_signal: bool = False,
        normalize_signal: bool = True
    ) -> None:
        if pyfaidx is None:
            raise ImportError("pyfaidx is required for GenomicDatasetBase")
        if pyBigWig is None:
            raise ImportError("pyBigWig is required for GenomicDatasetBase")

        self.fasta_path = fasta_path
        self.bed_path = bed_path
        self.reference_bw_path = reference_bw_path
        self.signal_bins = signal_bins
        self.binarize_signal = binarize_signal
        self.normalize_signal = normalize_signal
        self.window_size = window_size

        self._fasta = pyfaidx.Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        self._bw_ref = pyBigWig.open(reference_bw_path)
        self._bed = pd.read_csv(self.bed_path, sep="\t")
        self._k4_scale_factor = self._get_scale_factor(self._bw_ref)

        # Tile chromosome windows around BED regions
        intervals = make_uniform_intervals(
            bed=self._bed,
            window_size=window_size,
        )

        # Filter out intervals where the reference signal is zero or NaN.
        filtered_intervals: List[GenomicInterval] = []
        for iv in intervals:
            try:
                stats = self._bw_ref.stats(iv.chrom, iv.start, iv.end, type="mean")
                total = stats[0] if stats else 0
            except Exception:
                total = 0

            if total is not None and not np.isnan(total) and total > 0.0:
                filtered_intervals.append(iv)

        self.intervals = filtered_intervals
    
    def __len__(self) -> int:
        return len(self.intervals)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-picklable file handles
        state.pop("_fasta", None)
        state.pop("_bw_ref", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-open file handles
        self._fasta = pyfaidx.Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        self._bw_ref = pyBigWig.open(self.reference_bw_path)

    def _get_signal(self, bw: pyBigWig.BigWigFile, chrom: str, start: int, end: int) -> np.ndarray:
        """Helper to extract and preprocess signal from a bigWig."""
        if self.signal_bins is None:
            vals = bw.values(chrom, start, end, numpy=True)
        else:
            stats = bw.stats(chrom, start, end, nBins=self.signal_bins, type="mean")
            vals = np.array(stats, dtype=np.float32)
        
        vals = np.nan_to_num(vals, nan=0.0).astype(np.float32)

        if self.binarize_signal:
            vals = (vals > 0).astype(np.float32)
        elif self.normalize_signal:
            max_val = np.nanmax(vals)
            if max_val > 0:
                vals = vals / max_val
        
        return np.nan_to_num(vals, nan=0.0).astype(np.float32)
    
    def _get_scale_factor(self, bw):
        max_val = 1
        for chr in self._bed['#chrom'].unique():
            stats = bw.stats(chrom=chr, type='max')[0]
            if stats > max_val:
                max_val = stats
        return max_val


class SequenceBigWigDataset(GenomicDatasetBase):
    """
    Dataset that returns (one_hot_sequence, cutrun_signal) pairs.
    """
    def __init__(
        self,
        fasta_path: str,
        bigwig_path: str,
        bed_path: str,
        signal_bins: Optional[int] = None,
        window_size: int = 1000,
        binarize_signal: bool = False,
        normalize_signal: bool = True
    ) -> None:
        super().__init__(
            fasta_path=fasta_path,
            bed_path=bed_path,
            reference_bw_path=bigwig_path,
            signal_bins=signal_bins,
            window_size=window_size,
            binarize_signal=binarize_signal,
            normalize_signal=normalize_signal
        )
        self.bigwig_path = bigwig_path
        self._bw = self._bw_ref

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_bw", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._bw = self._bw_ref

    def __getitem__(self, idx: int):
        interval = self.intervals[idx]
        chrom, start, end = interval.chrom, interval.start, interval.end

        # Sequence from FASTA
        seq = self._fasta[chrom][start:end]
        one_hot = one_hot_encode_sequence(str(seq))

        # Signal from BigWig
        vals = self._get_signal(self._bw, chrom, start, end)

        # Convert to torch tensors
        x = torch.from_numpy(one_hot)          # (4, L)
        y = torch.from_numpy(vals)     
        return x, y


class SequenceDualBigWigDataset(GenomicDatasetBase):
    """
    Dataset that returns (sequence, k4_cutrun, target_y) triplets.
    """
    def __init__(
        self,
        fasta_path: str,
        k4_bigwig_path: str,
        target_bigwig_path: str,
        bed_path: str,
        signal_bins: Optional[int] = None,
        window_size: int = 1000,
        binarize_signal: bool = False,
        normalize_signal: bool = True
    ) -> None:
        super().__init__(
            fasta_path=fasta_path,
            bed_path=bed_path,
            reference_bw_path=target_bigwig_path,
            signal_bins=signal_bins,
            window_size=window_size,
            binarize_signal=binarize_signal,
            normalize_signal=normalize_signal
        )
        self.k4_bigwig_path = k4_bigwig_path
        self.target_bigwig_path = target_bigwig_path
        self._bw_k4 = pyBigWig.open(self.k4_bigwig_path)
        self._bw_target = self._bw_ref
        self._k4_scale_factor = self._get_scale_factor(self._bw_k4)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_bw_k4", None)
        state.pop("_bw_target", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._bw_k4 = pyBigWig.open(self.k4_bigwig_path)
        self._bw_target = self._bw_ref

    def __getitem__(self, idx: int):
        interval = self.intervals[idx]
        chrom, start, end = interval.chrom, interval.start, interval.end

        # Sequence from FASTA
        seq = self._fasta[chrom][start:end]
        one_hot = one_hot_encode_sequence(str(seq))

        # k4 CUT&RUN signal (input feature)
        k4_vals = self._get_signal(self._bw_k4, chrom, start, end) / self._k4_scale_factor

        # Target CUT&RUN signal
        target_vals = self._get_signal(self._bw_target, chrom, start, end)

        # Convert to torch tensors
        sequence = torch.from_numpy(one_hot)
        k4_cutrun = torch.from_numpy(k4_vals) 
        target_y = torch.from_numpy(target_vals)

        return sequence, k4_cutrun, target_y


__all__ = [
    "one_hot_encode_sequence",
    "GenomicInterval",
    "SequenceBigWigDataset",
    "SequenceDualBigWigDataset",
    "make_uniform_intervals",
]

