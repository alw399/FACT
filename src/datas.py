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
import torch.nn.functional as F
from torch.utils.data import Dataset
import pyfaidx 
# pyrefly: ignore [missing-import]
import pyBigWig 
from scipy.signal import find_peaks

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
    window_left: int = 500,
    window_right: int = 500,
    exclude_kws=['random'],
    subset_dict: dict = None,
) -> List[GenomicInterval]:
    """
    Convenience function: tile a chromosome with uniform windows.

    Parameters
    ----------
    bed
        BED file containing genomic intervals. Assume TSS is at start 
    """

    intervals: List[GenomicInterval] = []

    bed['band_start'] = bed['start'] - window_left
    bed['band_end'] = bed['start'] + window_right 

    if subset_dict is not None:
        for col, keywords in subset_dict.items():
            assert col in bed.columns, f'Column {col} not found in bed file.'
            if isinstance(keywords, list):
                bed = bed[bed[col].isin(keywords)]
            else:
                bed = bed[bed[col] == keywords]
    
    for chrom in bed['#chrom'].unique():
        if any(kw in chrom for kw in exclude_kws):
            continue

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
        window_left: int = 500,
        window_right: int = 500,
        subset_dict: dict = None,
    ) -> None:

        self.fasta_path = fasta_path
        self.bed_path = bed_path
        self.reference_bw_path = reference_bw_path
        self.signal_bins = signal_bins
        self.window_left = window_left
        self.window_right = window_right

        self._fasta = pyfaidx.Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        self._bw_ref = pyBigWig.open(reference_bw_path)
        self._bed = pd.read_csv(self.bed_path, sep="\t")

        # Tile chromosome windows around BED regions
        intervals = make_uniform_intervals(
            bed=self._bed,
            window_left=window_left,
            window_right=window_right,
            subset_dict=subset_dict
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

        return np.nan_to_num(vals, nan=0.0).astype(np.float32)


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
        window_left: int = 500,
        window_right: int = 500,
        subset_dict: dict = None,
    ) -> None:
        super().__init__(
            fasta_path=fasta_path,
            bed_path=bed_path,
            reference_bw_path=bigwig_path,
            signal_bins=signal_bins,
            window_left=window_left,
            window_right=window_right,
            subset_dict=subset_dict
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

    def __getitem__(self, idx):
        interval = self.intervals[idx]
        chrom, start, end = interval.chrom, interval.start, interval.end

        seq = self._fasta[chrom][start:end]
        one_hot = one_hot_encode_sequence(str(seq))
        signal = self._get_signal(self._bw, chrom, start, end)

        return torch.from_numpy(one_hot), torch.from_numpy(signal)


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
        window_left: int = 500,
        window_right: int = 500,
        subset_dict: dict = None,
    ) -> None:
        super().__init__(
            fasta_path=fasta_path,
            bed_path=bed_path,
            reference_bw_path=target_bigwig_path,
            signal_bins=signal_bins,
            window_left=window_left,
            window_right=window_right,
            subset_dict=subset_dict,
        )
        self.k4_bigwig_path = k4_bigwig_path
        self._bw_k4 = pyBigWig.open(self.k4_bigwig_path)
        self._bw_target = self._bw_ref

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
        k4_vals = self._get_signal(self._bw_k4, chrom, start, end)

        # Target CUT&RUN signal
        target_vals = self._get_signal(self._bw_target, chrom, start, end)


        # Convert to torch tensors
        sequence = torch.from_numpy(one_hot)          # (4, L)
        k4_cutrun = torch.log1p(torch.from_numpy(k4_vals))
        target_y = torch.from_numpy(target_vals)

        return sequence, k4_cutrun, target_y



class SequenceBigWigPeaksDataset(GenomicDatasetBase):
    """
    Dataset that dynamically discovers peaks in a BigWig file, generates intervals
    centered on those summits, and returns (sequence, target_y).
    """
    def __init__(
        self,
        fasta_path: str,
        target_bw_path: str,
        peak_bw_path: Optional[str] = None,
        threshold_percentile: float = 99.9,
        min_peak_distance: int = 1000,
        signal_bins: Optional[int] = None,
        window_left: int = 500,
        window_right: int = 500,
    ) -> None:
        if pyfaidx is None or pyBigWig is None:
            raise ImportError("pyfaidx and pyBigWig are required.")
            
        self.fasta_path = fasta_path
        self.reference_bw_path = target_bw_path
        self.signal_bins = signal_bins
        self.window_left = window_left
        self.window_right = window_right
        
        self.peak_bw_path = peak_bw_path if peak_bw_path else target_bw_path
        
        self._fasta = pyfaidx.Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        self._bw_ref = pyBigWig.open(self.reference_bw_path)
        
        # Dynamically discover peaks to build self._bed
        print(f"Discovering peaks in {self.peak_bw_path}...")
        bw_peak = pyBigWig.open(self.peak_bw_path)
        
        # Calculate threshold
        sample_chroms = [c for c in ['chr1', 'chr2', 'chr3'] if c in bw_peak.chroms()]
        if not sample_chroms:
            sample_chroms = list(bw_peak.chroms().keys())[:3]
            
        sample_vals = []
        for c in sample_chroms:
            v = bw_peak.values(c, 0, bw_peak.chroms()[c], numpy=True)
            v = np.nan_to_num(v, nan=0.0)
            sample_vals.append(v[v > 0])
            
        if len(sample_vals) > 0:
            all_sample = np.concatenate(sample_vals)
            threshold = np.percentile(all_sample, threshold_percentile)
        else:
            threshold = 10.0
            
        print(f"Using peak threshold: {threshold:.2f} ({threshold_percentile}th percentile)")
        
        peaks = []
        for chrom, length in bw_peak.chroms().items():
            if '_' in chrom or 'M' in chrom or 'random' in chrom or 'Un' in chrom: 
                continue
            vals = bw_peak.values(chrom, 0, length, numpy=True)
            vals = np.nan_to_num(vals, nan=0.0)
            
            peak_idxs, _ = find_peaks(vals, height=threshold, distance=min_peak_distance)
            for idx in peak_idxs:
                peaks.append({
                    '#chrom': chrom,
                    'start': int(idx),
                    'end': int(idx + 1)
                })
        
        bw_peak.close()
        self._bed = pd.DataFrame(peaks)
        print(f"Discovered {len(self._bed)} peaks.")
        
        # Assuming peak summit is at 'start' (which it is, since start=idx and end=idx+1)
        # make_uniform_intervals will natively expand around 'start'
        intervals = make_uniform_intervals(
            bed=self._bed,
            window_left=window_left,
            window_right=window_right,
            subset_dict=None
        )
        
        filtered_intervals = []
        for iv in intervals:
            try:
                stats = self._bw_ref.stats(iv.chrom, iv.start, iv.end, type="mean")
                total = stats[0] if stats else 0
            except Exception:
                total = 0
            if total is not None and not np.isnan(total) and total > 0.0:
                filtered_intervals.append(iv)
                
        self.intervals = filtered_intervals
        self._bw = self._bw_ref

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_fasta", None)
        state.pop("_bw_ref", None)
        state.pop("_bw", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fasta = pyfaidx.Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        self._bw_ref = pyBigWig.open(self.reference_bw_path)
        self._bw = self._bw_ref

    def __len__(self) -> int:
        return len(self.intervals)

    def _get_signal(self, bw: pyBigWig.BigWigFile, chrom: str, start: int, end: int) -> np.ndarray:
        if self.signal_bins is None:
            vals = bw.values(chrom, start, end, numpy=True)
        else:
            stats = bw.stats(chrom, start, end, nBins=self.signal_bins, type="mean")
            vals = np.array(stats, dtype=np.float32)
        vals = np.nan_to_num(vals, nan=0.0).astype(np.float32)
        return vals

    def __getitem__(self, idx: int):
        interval = self.intervals[idx]
        chrom, start, end = interval.chrom, interval.start, interval.end
        seq = self._fasta[chrom][start:end]
        one_hot = one_hot_encode_sequence(str(seq))
        signal = self._get_signal(self._bw, chrom, start, end)
        return torch.from_numpy(one_hot), torch.from_numpy(signal)




def get_dataset_composition(dataset: Dataset, n_samples: Optional[int] = None, only_peaks: bool = False):
    """
    Computes the frequency of A, C, G, T and GC content across the dataset.
    
    Parameters
    ----------
    dataset
        The PyTorch dataset to analyze.
    n_samples
        Maximum number of sequences to analyze.
    only_peaks
        If True, only analyze sequences where the target signal (y) is > 0.
    """
    counts = np.zeros(4)
    total_bases = 0
    analyzed_count = 0

    for i in range(len(dataset)):
        if n_samples is not None and analyzed_count >= n_samples:
            break
            
        item = dataset[i]
        x = item[0] 
        y = item[-1] # Target signal is always the last element

        # Filter for peaks if requested
        if only_peaks:
            y_sum = y.sum() if isinstance(y, torch.Tensor) else np.sum(y)
            if y_sum == 0:
                continue

        if isinstance(x, torch.Tensor):
            x = x.numpy()
        
        counts += x.sum(axis=1)
        total_bases += x.shape[1]
        analyzed_count += 1

    if analyzed_count == 0:
        print("No sequences found matching the criteria.")
        return

    freqs = counts / total_bases
    gc_content = freqs[1] + freqs[2]

    print("-" * 30)
    print(f"Nucleotide Frequencies (Analyzed {analyzed_count} sequences):")
    print(f"  A: {freqs[0]:.2%}")
    print(f"  C: {freqs[1]:.2%}")
    print(f"  G: {freqs[2]:.2%}")
    print(f"  T: {freqs[3]:.2%}")
    print("-" * 30)
    print(f"  GC Content: {gc_content:.2%}")
    print("-" * 30)


class SequenceBigWigClusterDataset(SequenceBigWigDataset):
    def __init__(
        self,
        fasta_path: str,
        bed_path: str,
        bigwig_path: str,
        signal_bins: Optional[int] = None,
        window_left: int = 500,
        window_right: int = 500,
        subset_dict: dict = None,
    ) -> None:

        self.fasta_path = fasta_path
        self.bed_path = bed_path
        self.bigwig_path = bigwig_path
        self.signal_bins = signal_bins
        self.window_left = window_left
        self.window_right = window_right

        self._fasta = pyfaidx.Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        self._bw = pyBigWig.open(bigwig_path)
        self._bed = pd.read_csv(self.bed_path, sep="\t")

        # Tile chromosome windows around BED regions
        intervals, chrom_labels = self.make_uniform_intervals(
            bed=self._bed,
            window_left=window_left,
            window_right=window_right,
            subset_dict=subset_dict
        )

        # Filter out intervals where the reference signal is zero or NaN.
        filtered_intervals: List[GenomicInterval] = []
        filtered_labels = []
        for iv, label in zip(intervals, chrom_labels):
            try:
                stats = self._bw.stats(iv.chrom, iv.start, iv.end, type="mean")
                total = stats[0] if stats else 0
            except Exception:
                total = 0

            if total is not None and not np.isnan(total) and total > 0.0:
                filtered_intervals.append(iv)
                filtered_labels.append(label)
        
        self.intervals = filtered_intervals
        self.chrom_labels = filtered_labels

        assert len(self.chrom_labels) == len(self.intervals)
    
    def make_uniform_intervals(self, bed, window_left, window_right, exclude_kws=['random'], subset_dict=None):
        intervals: List[GenomicInterval] = []

        bed['band_start'] = bed['start'] - window_left
        bed['band_end'] = bed['start'] + window_right 

        if subset_dict is not None:
            for col, keywords in subset_dict.items():
                assert col in bed.columns, f'Column {col} not found in bed file.'
                if isinstance(keywords, list):
                    bed = bed[bed[col].isin(keywords)]
                else:
                    bed = bed[bed[col] == keywords]
        
        
        cluster_labels = []
        for chrom in bed['#chrom'].unique():
            if any(kw in chrom for kw in exclude_kws):
                continue

            chrom_bed = bed[bed['#chrom'] == chrom]

            chrom_intervals = [
                GenomicInterval(chrom=chrom, start=start, end=end) 
                for start, end in zip(chrom_bed['band_start'], chrom_bed['band_end'])
            ]
            intervals.extend(chrom_intervals)
            cluster_labels.extend(chrom_bed['deepTools_group'].str.replace('cluster_', '').astype(int))

        # minimize the label
        cluster_labels = np.array(cluster_labels) - min(cluster_labels)
        return intervals, cluster_labels
    
    def __getitem__(self, idx):
        interval = self.intervals[idx]
        chrom, start, end = interval.chrom, interval.start, interval.end

        seq = self._fasta[chrom][start:end]
        one_hot = torch.from_numpy(one_hot_encode_sequence(str(seq)))
        signal = torch.tensor(self._get_signal(self._bw, chrom, start, end), dtype=torch.float32)
        label = torch.tensor(self.chrom_labels[idx], dtype=torch.long)

        return one_hot, label, signal
    
    def __len__(self):
        return len(self.intervals)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-picklable file handles
        state.pop("_fasta", None)
        state.pop("_bw", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-open file handles
        self._fasta = pyfaidx.Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        self._bw = pyBigWig.open(self.bigwig_path)



__all__ = [
    "one_hot_encode_sequence",
    "GenomicInterval",
    "SequenceBigWigDataset",
    "SequenceDualBigWigDataset",
    "SequenceBigWigPeaksDataset",
    "SequenceBigWigClusterDataset",
    "make_uniform_intervals",
    "get_dataset_composition"
]

