#!/bin/bash
set -euo pipefail

# ====== Env ======
# Setup environment for Local 
source ~/.bashrc
conda activate knapspec
echo "[ENV] Activated conda env"

# ====== Dirs ======
mkdir -p bench_out
mkdir -p bench_out/logs
echo "[DIR] Created output dirs"

# ====== Config ======
# MODEL="Qwen/Qwen3-32B"
# SKIP_BUDGET=20

# MODEL="Qwen/Qwen3-14B"
# SKIP_BUDGET=10

MODEL="Qwen/Qwen3-8B"
SKIP_BUDGET=8

# MODEL="Qwen/Qwen3-4B"
# SKIP_BUDGET=8



# MODEL="meta-llama/Meta-Llama-3-70B"
# SKIP_BUDGET=20

# MODEL="meta-llama/Meta-Llama-3-8B"
# SKIP_BUDGET=8

# MODEL="meta-llama/Llama-3.1-8B"
# SKIP_BUDGET=8

# MODEL="meta-llama/Llama-3.2-3B"
# SKIP_BUDGET=8

# MODEL="meta-llama/Llama-3.2-1B"
# SKIP_BUDGET=4

DATASET="aime24"
NUM_SAMPLES=30
MAX_LENGTH=32768

# DATASET="aime25"
# NUM_SAMPLES=30
# MAX_LENGTH=32768

# DATASET="mmlu_pro"
# NUM_SAMPLES=70
# MAX_LENGTH=4096

# DATASET="gpqa"
# NUM_SAMPLES=30
# MAX_LENGTH=32768

# DATASET="pg19"
# NUM_SAMPLES=20
# MAX_LENGTH=1024

# DATASET="govreport"
# NUM_SAMPLES=20
# MAX_LENGTH=1024

# DATASET="booksum"
# NUM_SAMPLES=20
# MAX_LENGTH=1024

# DATASET="specbench"
# NUM_SAMPLES=3
# MAX_LENGTH=128

GAMMA=10
OPTIMIZE_INTERVAL=64
OUT_DIR="bench_out"

# Allow passing start and end index as arguments
# Usage: ./run_comparison.sh 0 15
START_IDX=${1:-0}
END_IDX=${2:-""} # Optional

echo "Running benchmark comparison (AR -> CLASP -> KnapSpec...)"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Slice: ${START_IDX} to ${END_IDX:-${NUM_SAMPLES} (length)}"

# Construct the command array
CMD=(
    python benchmark.py
    --model "${MODEL}"
    --dataset "${DATASET}"
    --max-length "${MAX_LENGTH}"
    --gamma "${GAMMA}"
    --skip-budget "${SKIP_BUDGET}"
    --optimize-interval "${OPTIMIZE_INTERVAL}"
    --out-dir "${OUT_DIR}"
    --compare-strategies
    --start-idx "${START_IDX}"
)

# If END_IDX is provided, use it. Otherwise use NUM_SAMPLES (which acts as length)
if [ -n "${END_IDX}" ]; then
    CMD+=(--end-idx "${END_IDX}")
    LOG_SUFFIX="${START_IDX}_to_${END_IDX}"
else
    CMD+=(--num-samples "${NUM_SAMPLES}")
    LOG_SUFFIX="${START_IDX}_len_${NUM_SAMPLES}"
fi

MODEL_TAG="${MODEL//\//_}"
LOG_FILE="${OUT_DIR}/logs/comparison_$(date +%Y%m%d_%H%M%S)_${MODEL_TAG}_${DATASET}_${LOG_SUFFIX}.log"

echo "[RUN] Logging to: ${LOG_FILE}"

# Run
set -x
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
set +x

echo "[DONE] Comparison complete."
