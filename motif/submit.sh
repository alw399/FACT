#!/bin/bash
#SBATCH --output="motif_DMSO.out"
#SBATCH --cluster=htc
#SBATCH --job-name="motif_DMSO"
#SBATCH --cpus-per-task 1
#SBATCH --time 0-6:00:00
#SBATCH --mem 50G

set -e


conda init
source ~/.bashrc
conda activate scGPT
echo "Conda env: $CONDA_DEFAULT_ENV"

module load macs/3.0.3

# Use absolute motif dir so Python finds run_motif_analysis.py when job runs from SLURM spool dir
MOTIF_DIR="/ix/djishnu/alw399/FACT/motif"
cd "$MOTIF_DIR"

# Paths
BW_DIR="/ix/djishnu/alw399/FACT/data/bws_for_predictor"
BED_DIR="/ix/djishnu/alw399/FACT/data/beds_for_motif"
RESULTS_DIR="/ix/djishnu/alw399/FACT/motif_results"

mkdir -p "$BED_DIR"
mkdir -p "$RESULTS_DIR"

# SAMPLE="SPT16_4D_2HdTAG_Sox2_S46_1_120_sorted"
# SAMPLE="SPT16_4D_6HdTAG_Sox2_S49_1_120_sorted"
# SAMPLE="SPT16_4D_10dTAG_Sox2_rep2_S58_1_120_sorted"
# SAMPLE="SPT16_4D_DMSO_Sox2_S40_1_120_sorted"

# SAMPLE="SPT16_4D_2HdTAG_K4_S49_150_500_scaled"
# SAMPLE="SPT16_4D_6HdTAG_K4_S55_150_500_scaled"
# SAMPLE="SPT16_4D_10dTAG_K4_rep2_S55_150_500_scaled"
# SAMPLE="SPT16_4D_30dTAG_K4_S43_150_500_scaled"
SAMPLE="SPT16_4D_DMSO_K4_S31_150_500_scaled"

CUTOFF=0.5
BW="${BW_DIR}/${SAMPLE}.bw"
BED="${BED_DIR}/${SAMPLE}_peaks.bed"

if [[ ! -f "$BW" ]]; then
  echo "Error: bigWig not found: $BW"
  exit 1
fi

# --- Step 1: Peak calling ---

echo "=== Step 1: Peak calling $SAMPLE -> $BED ==="
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

bigWigToBedGraph "$BW" "$TMP/signal.bedGraph"
macs3 bdgpeakcall \
  -i "$TMP/signal.bedGraph" \
  -o "$TMP/peaks.narrowPeak" \
  -c "$CUTOFF" \
  -l 200 \
  -g 30

# MACS3 may write peaks.narrowPeak or <prefix>_peaks.narrowPeak
NARROW="$TMP/peaks.narrowPeak"
[[ ! -f "$NARROW" ]] && NARROW="$TMP/peaks_peaks.narrowPeak"
[[ ! -f "$NARROW" ]] && NARROW=$(find "$TMP" -name '*.narrowPeak' -print -quit)
if [[ -f "$NARROW" ]] && [[ -s "$NARROW" ]]; then
  grep -v '^track' "$NARROW" | cut -f1-6 > "$BED"
  PEAK_COUNT=$(wc -l < "$BED")
  echo "Peaks written to $BED ($PEAK_COUNT peaks)"
else
  echo "Error: macs3 produced no peaks for $SAMPLE (cutoff=$CUTOFF). Try lowering CUTOFF (e.g. 0.1) at top of script."
  echo "Contents of $TMP: $(ls -la "$TMP" 2>/dev/null || true)"
  exit 1
fi

# --- Step 2: Motif analysis (needs conda for Python/HOMER) ---

echo "=== Step 2: Motif analysis $SAMPLE -> ${RESULTS_DIR}/${SAMPLE} ==="
python "${MOTIF_DIR}/run_motif_analysis.py" \
  --data "$BED" \
  --results "${RESULTS_DIR}/${SAMPLE}"

echo "Done. Results in ${RESULTS_DIR}/${SAMPLE}/"
