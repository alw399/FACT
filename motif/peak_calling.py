"""
Convert MACS narrowPeak output to 6-column BED for HOMER.

Peak calling (bigWig -> bedGraph -> macs2 bdgpeakcall) is done from
submit.sh. This script is for standalone conversion of a narrowPeak
file to 6-col BED if needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def narrowpeak_to_bed(narrowpeak_path: str | Path, out_bed_path: str | Path) -> int:
    """
    Write 6-column BED (chr, start, end, name, score, strand) from narrowPeak.
    Returns number of lines written.
    """
    narrowpeak_path = Path(narrowpeak_path)
    out_bed_path = Path(out_bed_path)
    out_bed_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(narrowpeak_path) as f_in, open(out_bed_path, "w") as f_out:
        for line in f_in:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) >= 6:
                f_out.write("\t".join(parts[:6]) + "\n")
                n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert narrowPeak to 6-col BED for HOMER.",
    )
    parser.add_argument("narrowpeak", type=Path, help="Path to narrowPeak file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output BED path")
    args = parser.parse_args()
    if not args.narrowpeak.exists():
        print(f"Error: {args.narrowpeak} not found", file=sys.stderr)
        sys.exit(1)
    n = narrowpeak_to_bed(args.narrowpeak, args.output)
    print(f"Wrote {n} peaks to {args.output}")


if __name__ == "__main__":
    main()
