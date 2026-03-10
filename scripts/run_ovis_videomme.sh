#!/bin/bash
# Run Ovis evaluation on VideoMME benchmark (multi-GPU via torchrun)
# Videos located at: /home/ubuntu/.cache/huggingface/videomme/data/
# Runs the following configurations sequentially:
#   1.  8f  max_clip=1  fps=2  →  clips:8:1:2
#   2. 16f  max_clip=1  fps=2  →  clips:16:1:2
#   3. 16f  max_clip=2  fps=2  →  clips:16:2:2
#   4. 16f  max_clip=4  fps=2  →  clips:16:4:2
#   5. 16f  max_clip=8  fps=2  →  clips:16:8:2

set -e

# ── Config ────────────────────────────────────────────────────────────────────
MODEL="ovis"
VIDEO_DIR="/home/ubuntu/.cache/huggingface/videomme/data"
MAX_TOKENS=10
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"  # auto-detect all GPUs
TEST_FLAG=""
# ─────────────────────────────────────────────────────────────────────────────

# Configurations to run sequentially: "label|mode"
CONFIGS=(
    "8f  max_clip=1 fps=2|clips:8:1:2"
    "16f max_clip=1 fps=2|clips:16:1:2"
    "16f max_clip=2 fps=2|clips:16:2:2"
    "16f max_clip=4 fps=2|clips:16:4:2"
    "16f max_clip=8 fps=2|clips:16:8:2"
    "16f max_clip=16 fps=2|clips:16:16:2"
)

# Parse optional CLI flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)      TEST_FLAG="--test"; shift ;;
        --num_gpus)  NUM_GPUS="$2";      shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."   # run from repo root

TORCHRUN="$(pdm run which torchrun)"

TOTAL=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
    IFS="|" read -r LABEL MODE <<< "${CONFIGS[$i]}"
    RUN_NUM=$((i + 1))

    echo ""
    echo "============================================================"
    echo " Ovis  ×  VideoMME  —  Run ${RUN_NUM}/${TOTAL}"
    echo "============================================================"
    echo "  Config     : $LABEL"
    echo "  Mode       : $MODE"
    echo "  Video dir  : $VIDEO_DIR"
    echo "  Max tokens : $MAX_TOKENS"
    echo "  GPUs       : $NUM_GPUS"
    echo "  Test mode  : ${TEST_FLAG:-off}"
    echo "============================================================"

    "$TORCHRUN" \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29501 \
        scripts/vlm_VideoMME.py \
            --model      "$MODEL"      \
            --video_dir  "$VIDEO_DIR"  \
            --mode       "$MODE"       \
            --max_tokens "$MAX_TOKENS" \
            $TEST_FLAG

    echo "✅  Run ${RUN_NUM}/${TOTAL} finished: $LABEL"
done

echo ""
echo "============================================================"
echo " All ${TOTAL} configurations completed successfully!"
echo "============================================================"