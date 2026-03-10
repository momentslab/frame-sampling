#!/bin/bash
# Run InternVL then Qwen2.5 on VideoMME with clips:16:16:2 sequentially.

set -e

MODE="clips:16:16:2"
VIDEO_DIR="/home/ubuntu/.cache/huggingface/videomme/data"
MAX_TOKENS=10
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"
TEST_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)      TEST_FLAG="--test"; shift ;;
        --num_gpus)  NUM_GPUS="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."

TORCHRUN="$(pdm run which torchrun)"

run_eval() {
    local model="$1"
    local label="$2"
    local master_port="$3"

    echo ""
    echo "============================================================"
    echo " ${label}  ×  VideoMME"
    echo "============================================================"
    echo "  Mode       : $MODE"
    echo "  Video dir  : $VIDEO_DIR"
    echo "  Max tokens : $MAX_TOKENS"
    echo "  GPUs       : $NUM_GPUS"
    echo "  Test mode  : ${TEST_FLAG:-off}"
    echo "============================================================"

    "$TORCHRUN" \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$master_port" \
        scripts/vlm_VideoMME.py \
            --model      "$model"      \
            --video_dir  "$VIDEO_DIR"  \
            --mode       "$MODE"       \
            --max_tokens "$MAX_TOKENS" \
            $TEST_FLAG

    echo "✅ Finished: ${label} (${MODE})"
}

run_eval "intern" "InternVL" 29501
run_eval "qwen2_5" "Qwen2.5" 29502

echo ""
echo "============================================================"
echo " Both sequential runs completed successfully!"
echo "============================================================"