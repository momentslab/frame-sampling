#!/bin/bash
# Run Qwen2.5 evaluation on VideoMME benchmark (multi-GPU via torchrun)

set -e

MODEL="qwen2_5"
VIDEO_DIR="/home/ubuntu/.cache/huggingface/videomme/data"
MAX_TOKENS=10
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"
TEST_FLAG=""

CONFIGS=(
    #"fps=2 min=4 max=96|fps:2:4:96"
    #"8f  max_clip=1 fps=2|clips:8:1:2"
    #"16f max_clip=1 fps=2|clips:16:1:2"
    #"16f max_clip=2 fps=2|clips:16:2:2"
    #"16f max_clip=4 fps=2|clips:16:4:2"
    #"16f max_clip=8 fps=2|clips:16:8:2"
    "16f max_clip=16 fps=2|clips:16:16:2"
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)      TEST_FLAG="--test"; shift ;;
        --num_gpus)  NUM_GPUS="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."

TORCHRUN="$(pdm run which torchrun)"

TOTAL=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
    IFS="|" read -r LABEL MODE <<< "${CONFIGS[$i]}"
    RUN_NUM=$((i + 1))

    echo ""
    echo "============================================================"
    echo " Qwen2.5  ×  VideoMME  —  Run ${RUN_NUM}/${TOTAL}"
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