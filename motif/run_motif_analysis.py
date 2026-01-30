#!/usr/bin/env python3
"""
Run motif analysis (PWM / HOMER) on CUT&RUN data.

Accepts --data as a BED file of peak regions or a directory containing BED files.
Peak calling (bigWig -> BED) is done from submit.sh; use those BEDs as --data here.

Requires HOMER (findMotifsGenome.pl) to be installed and in PATH.
Genome must be prepared with HOMER: run "perl /path/to/homer/configureHomer.pl -install mm10"
(or your genome) if needed.

Example:
  python run_motif_analysis.py \\
    --data /ix/djishnu/alw399/FACT/data/bws_for_predictor \\
    --results /ix/djishnu/alw399/FACT/motif/results_sox2
  python run_motif_analysis.py \\
    --data /path/to/peaks.bed \\
    --results /path/to/motif_output
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_bed_in_dir(path: Path) -> Path | None:
    """Return first .bed or .bed.gz file in directory, or None."""
    for ext in ("*.bed", "*.bed.gz"):
        for f in path.glob(ext):
            return f
    return None


def run_homer(
    bed_path: Path,
    genome: str,
    output_dir: Path,
    size: str | int = 200,
    preparsed_dir: Path | None = None,
) -> None:
    """Run HOMER findMotifsGenome.pl."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "findMotifsGenome.pl",
        str(bed_path),
        genome,
        str(output_dir),
        "-size",
        str(size),
    ]
    if preparsed_dir is not None:
        cmd.extend(["-preparsedDir", str(preparsed_dir)])

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "Error: findMotifsGenome.pl not found. Install HOMER and ensure it is in your PATH, "
            "e.g. configureHomer.pl -install mm10",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"HOMER exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run motif analysis (HOMER) on CUT&RUN peaks; optionally call peaks from bigWig first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to BED file of peaks or directory containing BED files.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Directory where motif analysis results (and PWMs) will be written.",
    )
    parser.add_argument(
        "--genome",
        type=str,
        default="mm10",
        help="Reference genome name for HOMER (e.g. mm10, hg38). Must be installed in HOMER.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="200",
        help='HOMER -size: region size in bp (e.g. 200) or "given" to use peak sizes.',
    )
    parser.add_argument(
        "--preparsed-dir",
        type=Path,
        default=None,
        help="HOMER -preparsedDir if genome parsing should be written here (e.g. on shared cluster).",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    results_path = Path(args.results)

    if not data_path.exists():
        print(f"Error: --data path does not exist: {data_path}", file=sys.stderr)
        sys.exit(1)

    bed_to_use: Path | None = None
    effective_results_path = results_path

    if data_path.is_file():
        if data_path.suffix in (".bed", "") or ".bed." in data_path.name:
            bed_to_use = data_path
        else:
            print(
                f"Error: --data must be a BED file or a directory. Got: {data_path}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        bed_in_dir = find_bed_in_dir(data_path)
        if bed_in_dir is not None:
            bed_to_use = bed_in_dir
        else:
            print(
                f"Error: no BED file found in {data_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    if bed_to_use is None:
        sys.exit(1)

    print(f"Using peaks: {bed_to_use}")
    print(f"Results will be written to: {effective_results_path}")
    run_homer(
        bed_to_use,
        args.genome,
        effective_results_path,
        size=args.size,
        preparsed_dir=args.preparsed_dir,
    )
    print("Done. Check knownResults.txt and homerResults/ for PWMs and motif locations.")


if __name__ == "__main__":
    main()
