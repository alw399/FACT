# Motif analysis for CUT&RUN data

Runs HOMER motif discovery and known motif enrichment on CUT&RUN peaks, and outputs PWMs.

## Requirements

- **HOMER** installed and on your `PATH` (e.g. `findMotifsGenome.pl`).
- If needed, install a genome:  
  `perl /path/to/homer/configureHomer.pl -install mm10`
- For peak calling from bigWig: **bigWigToBedGraph** (UCSC) and **MACS** on `PATH`.  
  On this cluster: `module load macs/3.0.3` before running.

## Usage

You can pass either:

1. **A BED file** of peak regions (e.g. from MACS2 or another peak caller), or  
2. **A directory** containing CUT&RUN data. If it has `.bed` files, the first one is used. If it only has bigWig (`.bw`) and you use `--call-peaks`, peaks are called from a bigWig and then HOMER is run.

### Examples

Using a **directory** of CUT&RUN data (bigWig), with peak calling:

```bash
python run_motif_analysis.py \
  --data /ix/djishnu/alw399/FACT/data/bws_for_predictor \
  --results /ix/djishnu/alw399/FACT/motif/results_sox2 \
  --call-peaks
```

Using a **BED file** of peaks:

```bash
python run_motif_analysis.py \
  --data /path/to/peaks.bed \
  --results /path/to/motif_output
```

Optional arguments:

- `--genome mm10` (default) or e.g. `hg38`
- `--size 200` or `--size given` (HOMER region size)
- `--bigwig path/to/file.bw` — when `--data` is a directory, which bigWig to use for `--call-peaks`
- `--peak-threshold 1.0` — MACS score cutoff for peak calling from bigWig
- `--preparsed-dir /path` — HOMER `-preparsedDir` (useful on shared clusters)

## Output

Results are written under `--results`:

- **knownResults.txt** — known motif enrichment
- **homerResults/** — de novo motifs and PWM files (e.g. `motif1.motif`, `motif2.motif`)

## Peak calling only (no HOMER)

To only generate a BED from a bigWig (bigWigToBedGraph + MACS bdgpeakcall):

```bash
module load macs/3.0.3   # on this cluster
python peak_calling.py /path/to/signal.bw -o peaks.bed -c 1.0
```

Then use `peaks.bed` as `--data` in `run_motif_analysis.py`.
