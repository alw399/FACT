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


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - allows importing without torch installed
    torch = None
    Dataset = object  # type: ignore


try:
    import pyfaidx  # for FASTA access
except ImportError:  # pragma: no cover
    pyfaidx = None


try:
    import pyBigWig  # for bigWig access
except ImportError:  # pragma: no cover
    pyBigWig = None


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
    chrom: str,
    chrom_length: int,
    window_size: int,
    stride: Optional[int] = None,
) -> List[GenomicInterval]:
    """
    Convenience function: tile a chromosome with uniform windows.

    Parameters
    ----------
    chrom
        Chromosome name (e.g. 'chr3').
    chrom_length
        Length of the chromosome in bases.
    window_size
        Size of each window in bases.
    stride
        Step between window starts. Defaults to window_size
        (non-overlapping windows).
    """
    if stride is None:
        stride = window_size

    intervals: List[GenomicInterval] = []
    for start in range(0, chrom_length - window_size + 1, stride):
        end = start + window_size
        intervals.append(GenomicInterval(chrom=chrom, start=start, end=end))
    return intervals


class SequenceBigWigDataset(Dataset):
    """
    Dataset that returns (one_hot_sequence, cutrun_signal) pairs.

    - Sequences are read from an mm10 FASTA file via pyfaidx.
    - CUT&RUN signal is read from a bigWig via pyBigWig.

    Each item corresponds to a fixed-length genomic window.
    """

    def __init__(
        self,
        fasta_path: str,
        bigwig_path: str,
        signal_bins: Optional[int] = None,
        chrom: Optional[str] = None,
        window_size: int = 10000,
        stride: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        fasta_path
            Path to an mm10 FASTA file (not provided in this repo;
            download separately and supply the path).
        bigwig_path
            Path to CUT&RUN bigWig file.
        intervals
            Optional iterable of `GenomicInterval` objects; each defines a window
            to extract. If None, intervals will be automatically generated.
        signal_bins
            If not None, downsample the bigWig signal into this many bins
            per interval using pyBigWig.stats. If None, return per-base
            values with `values()` (can be large).
        chrom
            Chromosome name (e.g. 'chr3').
        window_size
            Size of each window in bases.
        stride
            Step between window starts. Defaults to window_size (non-overlapping).
        """
        if pyfaidx is None:
            raise ImportError("pyfaidx is required for SequenceBigWigDataset")
        if pyBigWig is None:
            raise ImportError("pyBigWig is required for SequenceBigWigDataset")

        self.fasta_path = fasta_path
        self.bigwig_path = bigwig_path
        self.signal_bins = signal_bins

        self._fasta = pyfaidx.Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        self._bw = pyBigWig.open(self.bigwig_path)

        # Basic validation: check that chromosomes overlap
        fasta_chroms = set(self._fasta.keys())
        bw_chroms = set(self._bw.chroms().keys())
        common = fasta_chroms & bw_chroms
        if not common:
            raise ValueError("No overlapping chromosomes between FASTA and bigWig.")

        # Generate intervals
        if chrom is None:
            raise ValueError(
                "Either 'intervals' or 'chrom' must be provided. "
                "If 'chrom' is provided, intervals will be automatically generated."
            )
        chrom_length = self._bw.chroms().get(chrom)
        if chrom_length is None:
            raise ValueError(
                f"Chromosome {chrom} not found in bigWig file. "
                f"Available chromosomes: {sorted(common)}"
            )
        if stride is None:
            stride = window_size
        # Initial tiling of the chromosome
        intervals = make_uniform_intervals(
            chrom=chrom,
            chrom_length=chrom_length,
            window_size=window_size,
            stride=stride,
        )

        # Filter out intervals where the total signal (y) is zero.
        # This avoids training on completely empty windows.
        filtered_intervals: List[GenomicInterval] = []
        for iv in intervals:
            # Use pyBigWig.stats with type="sum" to get total signal
            total = self._bw.stats(
                iv.chrom,
                iv.start,
                iv.end,
                type="sum",
            )[0]
            if total is None or np.isnan(total):
                total_val = 0.0
            else:
                total_val = float(total)

            if total_val > 0.0:
                filtered_intervals.append(iv)

        self.intervals = filtered_intervals
  
    def __len__(self) -> int:
        return len(self.intervals)

    def __getitem__(self, idx: int):
        if torch is None:
            raise ImportError("PyTorch is required to use this dataset.")

        interval = self.intervals[idx]
        chrom, start, end = interval.chrom, interval.start, interval.end

        # Sequence from FASTA
        seq = self._fasta[chrom][start:end]
        one_hot = one_hot_encode_sequence(str(seq))

        # CUT&RUN signal from bigWig
        if self.signal_bins is None:
            vals = self._bw.values(chrom, start, end, numpy=True)
            # pyBigWig uses NaN for missing; convert to 0
            vals = np.nan_to_num(vals, nan=0.0).astype(np.float32)
        else:
            stats = self._bw.stats(
                chrom,
                start,
                end,
                nBins=self.signal_bins,
                type="mean",
            )
            vals = np.array(
                [0.0 if v is None or np.isnan(v) else float(v) for v in stats],
                dtype=np.float32,
            )

        # Total counts in this window
        total_counts = float(vals.sum())
        y_count = np.array(total_counts, dtype=np.float32)

        # Normalize profile to [0, 1] (probability-like) if there is signal
        if total_counts > 0.0:
            vals_norm = vals / total_counts
        else:
            vals_norm = vals

        # Convert to torch tensors
        x = torch.from_numpy(one_hot)          # (4, L)
        y = torch.from_numpy(vals_norm)        # normalized profile (L,) or (signal_bins,)
        y_count_t = torch.from_numpy(y_count)  # total counts scalar

        return x, y, y_count_t


class SequenceDualBigWigDataset(Dataset):
    """
    Dataset that returns (sequence, additional_cutrun, target_cutrun) triplets.
    
    For multi-input models that use both sequence and an additional CUT&RUN
    experiment to predict a target CUT&RUN signal.

    - Sequences are read from an mm10 FASTA file via pyfaidx.
    - Both CUT&RUN signals are read from bigWig files via pyBigWig.

    Each item corresponds to a fixed-length genomic window.
    """

    def __init__(
        self,
        fasta_path: str,
        additional_bigwig_path: str,
        target_bigwig_path: str,
        intervals: Optional[Sequence[GenomicInterval]] = None,
        signal_bins: Optional[int] = None,
        chrom: Optional[str] = None,
        window_size: int = 10000,
        stride: Optional[int] = None,
        drop_zero_target: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        fasta_path
            Path to an mm10 FASTA file.
        additional_bigwig_path
            Path to the additional CUT&RUN bigWig file (used as input feature).
        target_bigwig_path
            Path to the target CUT&RUN bigWig file (what we want to predict).
        intervals
            Optional iterable of `GenomicInterval` objects; each defines a window
            to extract. If None, intervals will be automatically generated.
        signal_bins
            If not None, downsample the bigWig signals into this many bins
            per interval using pyBigWig.stats. If None, return per-base
            values with `values()` (can be large).
        chrom
            Chromosome name (e.g. 'chr3'). Required if intervals is None.
        window_size
            Size of each window in bases. Used only if intervals is None.
        stride
            Step between window starts. Defaults to window_size (non-overlapping).
            Used only if intervals is None.
        drop_zero_target
            If True, drop intervals whose *target* total signal is 0.
        """
        if pyfaidx is None:
            raise ImportError("pyfaidx is required for SequenceDualBigWigDataset")
        if pyBigWig is None:
            raise ImportError("pyBigWig is required for SequenceDualBigWigDataset")

        self.fasta_path = fasta_path
        self.additional_bigwig_path = additional_bigwig_path
        self.target_bigwig_path = target_bigwig_path
        self.signal_bins = signal_bins

        self._fasta = pyfaidx.Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        self._bw_additional = pyBigWig.open(self.additional_bigwig_path)
        self._bw_target = pyBigWig.open(self.target_bigwig_path)

        # Basic validation: check that chromosomes overlap
        fasta_chroms = set(self._fasta.keys())
        bw_additional_chroms = set(self._bw_additional.chroms().keys())
        bw_target_chroms = set(self._bw_target.chroms().keys())
        common = fasta_chroms & bw_additional_chroms & bw_target_chroms
        if not common:
            raise ValueError(
                "No overlapping chromosomes between FASTA and both bigWig files."
            )

        # Generate intervals if not provided
        if intervals is None:
            if chrom is None:
                raise ValueError(
                    "Either 'intervals' or 'chrom' must be provided. "
                    "If 'chrom' is provided, intervals will be automatically generated."
                )
            # Use target bigWig to get chromosome length (they should all have same chroms)
            chrom_length = self._bw_target.chroms().get(chrom)
            if chrom_length is None:
                raise ValueError(
                    f"Chromosome {chrom} not found in bigWig files. "
                    f"Available chromosomes: {sorted(common)}"
                )
            if stride is None:
                stride = window_size
            intervals = make_uniform_intervals(
                chrom=chrom,
                chrom_length=chrom_length,
                window_size=window_size,
                stride=stride,
            )
            # Optionally drop windows where target has zero total signal
            if drop_zero_target:
                filtered: List[GenomicInterval] = []
                for iv in intervals:
                    total = self._bw_target.stats(iv.chrom, iv.start, iv.end, type="sum")[0]
                    if total is None or np.isnan(total):
                        total_val = 0.0
                    else:
                        total_val = float(total)
                    if total_val > 0.0:
                        filtered.append(iv)
                self.intervals = filtered
            else:
                self.intervals = list(intervals)
        else:
            self.intervals: List[GenomicInterval] = list(intervals)

    def __len__(self) -> int:
        return len(self.intervals)

    def __getitem__(self, idx: int):
        if torch is None:
            raise ImportError("PyTorch is required to use this dataset.")

        interval = self.intervals[idx]
        chrom, start, end = interval.chrom, interval.start, interval.end

        # Sequence from FASTA
        seq = self._fasta[chrom][start:end]
        one_hot = one_hot_encode_sequence(str(seq))

        def _get_signal(bw, chrom, start, end):
            """Helper to extract signal from a bigWig."""
            if self.signal_bins is None:
                vals = bw.values(chrom, start, end, numpy=True)
                vals = np.nan_to_num(vals, nan=0.0).astype(np.float32)
            else:
                stats = bw.stats(
                    chrom,
                    start,
                    end,
                    nBins=self.signal_bins,
                    type="mean",
                )
                vals = np.array(
                    [0.0 if v is None or np.isnan(v) else float(v) for v in stats],
                    dtype=np.float32,
                )
            return vals

        # Additional CUT&RUN signal (input feature)
        additional_vals = _get_signal(self._bw_additional, chrom, start, end)
        additional_total = float(additional_vals.sum())
        if additional_total > 0.0:
            additional_norm = additional_vals / additional_total
        else:
            additional_norm = additional_vals

        # Target CUT&RUN signal (what we want to predict)
        target_vals = _get_signal(self._bw_target, chrom, start, end)
        target_total = float(target_vals.sum())
        if target_total > 0.0:
            target_norm = target_vals / target_total
        else:
            target_norm = target_vals

        # Convert to torch tensors
        sequence = torch.from_numpy(one_hot)  # (4, L)
        additional_cutrun = torch.from_numpy(additional_norm)  # normalized profile (L,) or (signal_bins,)
        target_y = torch.from_numpy(target_norm)               # normalized profile (L,) or (signal_bins,)
        target_y_count = torch.tensor(target_total, dtype=torch.float32)  # scalar

        return sequence, additional_cutrun, target_y, target_y_count


__all__ = [
    "one_hot_encode_sequence",
    "GenomicInterval",
    "SequenceBigWigDataset",
    "SequenceDualBigWigDataset",
    "make_uniform_intervals",
]

