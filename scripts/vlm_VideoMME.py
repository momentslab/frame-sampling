import argparse
import sys
import os
import time
import json
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

# ── Distributed setup ─────────────────────────────────────────────────────────
RANK       = int(os.environ.get("RANK",       0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

# Init process group so transformers doesn't try to do it itself
if WORLD_SIZE > 1:
    torch.distributed.init_process_group(backend="nccl")

# Pin each rank to its own GPU
torch.cuda.set_device(LOCAL_RANK)

sys.path.append('src')

from datasets import load_dataset
from models.utils import get_model, cleanup_model
from utils.videomme_evaluation import eval_your_results
from utils.common_eval_utils import create_video_config, extract_answer, save_frame_indices_json
from utils.videomme_utils import build_video_id_map, find_video_file_by_id, create_videomme_prompt

IS_MAIN = (RANK == 0)


# ── Arguments ─────────────────────────────────────────────────────────────────
def setup_arguments():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on VideoMME (multi-GPU via torchrun).")
    parser.add_argument("--model", type=str, required=True,
                        choices=["ovis", "smolvlm", "qwen2", "qwen2_5", "qwen3", "intern", "apollo"])
    parser.add_argument("--model_path", type=str, default=None,
                        help="Optional path to model weights (e.g. for Apollo local checkpoint).")
    parser.add_argument("--video_dir", type=str, default="./data/VideoMME")
    parser.add_argument("--mode", type=str, default="fps:2:4:8",
                        help="Frame mode: 'first'|'center'|'fps:fps:min:max'|'maxinfo:input:max'|'csta:input:max'|'ufp:N'|'clips:fpc:max_clips:target_fps[:ratio]'")
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--test", action="store_true",
                        help="Test mode: first N questions only")
    parser.add_argument("--test_samples", type=int, default=4,
                        help="Number of questions to use in --test mode (default: 4 = 1 video)")
    return parser.parse_args()


# ── Build flat question list ──────────────────────────────────────────────────
def build_flat_questions(dataset):
    """Flatten dataset into one entry per question (not per video)."""
    questions = []
    for example in dataset:
        questions.append({
            "video_id":    example["video_id"],
            "videoID":     example["videoID"].lower(),
            "duration":    example.get("duration",     "Unknown"),
            "domain":      example.get("domain",       "Unknown"),
            "sub_category":example.get("sub_category", "Unknown"),
            "question_id": example["question_id"],
            "question":    example["question"],
            "options":     example["options"],
            "answer":      example.get("answer",    "Unknown"),
            "task_type":   example.get("task_type", "Unknown"),
        })
    return questions


# ── Inference ─────────────────────────────────────────────────────────────────
def run_model_inference(model, video_path, prompt, args, question_id=None, video_id=None):
    video_items = create_video_config(args.model, video_path, args.mode)
    try:
        return model.predict(video_items=video_items, prompt=prompt, max_tokens=args.max_tokens)
    except Exception as exc:
        item_parts = []
        if question_id is not None:
            item_parts.append(f"question_id={question_id}")
        if video_id is not None:
            item_parts.append(f"video_id={video_id}")
        item_desc = ", ".join(item_parts) or "unknown item"
        print(
            f"[rank {RANK}] ❌ Inference failed for {item_desc} "
            f"model={args.model} mode={args.mode} path={video_path}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return None


# ── File-based barrier: rank 0 waits for all partial files ───────────────────
def wait_for_partials(results_dir, world_size, mode_filename, timeout=7200):
    """Rank 0 polls until every rank has written its partial file."""
    import time as _time
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        missing = [r for r in range(world_size)
                   if not (results_dir / f"partial_rank{r}_{mode_filename}.json").exists()]
        if not missing:
            return
        _time.sleep(5)
    raise TimeoutError(f"Timed out waiting for partial results from ranks: {missing}")


# ── Merge partials → video-grouped format for eval ───────────────────────────
def merge_partial_results(results_dir, world_size, mode_filename):
    all_questions = []
    for r in range(world_size):
        path = results_dir / f"partial_rank{r}_{mode_filename}.json"
        with open(path) as f:
            all_questions.extend(json.load(f))
        path.unlink()

    # Re-group by video_id (required by eval_your_results)
    video_map = {}
    for q in all_questions:
        vid = q["video_id"]
        if vid not in video_map:
            video_map[vid] = {
                "video_id":    vid,
                "duration":    q["duration"],
                "domain":      q["domain"],
                "sub_category":q["sub_category"],
                "questions":   [],
            }
        video_map[vid]["questions"].append({
            "question_id":     q["question_id"],
            "task_type":       q["task_type"],
            "question":        q["question"],
            "options":         q["options"],
            "answer":          q["answer"],
            "response":        q["response"],
            "predicted_answer":q["predicted_answer"],
            "correct":         q["correct"],
        })
    return list(video_map.values())


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    total_start = time.time()   # wall-clock start: before everything
    args = setup_arguments()

    if IS_MAIN:
        print("=" * 60)
        print(f"🎯 VideoMME Evaluation  |  {WORLD_SIZE} GPU(s)")
        print(f"   Model : {args.model}  |  Mode : {args.mode}")
        print(f"   Ranks : {WORLD_SIZE}  (this is rank {RANK})")
        print("=" * 60)

    # ── Load dataset & build flat question list ───────────────────
    if IS_MAIN:
        print("📦 Loading dataset...")
    dataset = load_dataset("lmms-lab/Video-MME", split="test")
    flat_questions = build_flat_questions(dataset)

    if args.test:
        flat_questions = flat_questions[:args.test_samples]

    # ── Shard by question index: rank r takes indices r, r+W, r+2W … ─
    shard = flat_questions[RANK::WORLD_SIZE]

    if IS_MAIN:
        print(f"📊 Total questions: {len(flat_questions)} | Per GPU: ~{len(shard)}")

    # ── Build video path index & load model ──────────────────────
    video_id_map = build_video_id_map(args.video_dir)

    if IS_MAIN:
        print("🤖 Loading model...")
    model_kwargs = {}
    if args.model_path:
        model_kwargs["model_path"] = args.model_path
    if args.model == "apollo":
        model_kwargs["mode"] = args.mode
    model = get_model(args.model, **model_kwargs)

    results_dir = Path("results/VideoMME") / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference loop ────────────────────────────────────────────
    partial_results = []
    video_path_cache: dict = {}   # avoid repeated filesystem scans per rank

    pbar = tqdm(
        shard,
        desc=f"GPU {RANK}",
        unit="q",
        position=0,
        leave=True,
        dynamic_ncols=True,
        disable=not IS_MAIN,
    )

    for item in pbar:
        videoID = item["videoID"]

        # Resolve video path (cached)
        if videoID not in video_path_cache:
            try:
                video_path_cache[videoID] = find_video_file_by_id(videoID, video_id_map)
            except Exception:
                video_path_cache[videoID] = None
        video_path = video_path_cache[videoID]

        prompt = create_videomme_prompt(item)

        if video_path is None:
            response, predicted, correct = None, None, False
        else:
            response  = run_model_inference(
                model,
                video_path,
                prompt,
                args,
                question_id=item["question_id"],
                video_id=item["video_id"],
            )
            predicted = extract_answer(response)
            correct   = (predicted == item["answer"])

        torch.cuda.empty_cache()

        partial_results.append({
            "video_id":        item["video_id"],
            "videoID":         item["videoID"],
            "duration":        item["duration"],
            "domain":          item["domain"],
            "sub_category":    item["sub_category"],
            "question_id":     item["question_id"],
            "task_type":       item["task_type"],
            "question":        item["question"],
            "options":         item["options"],
            "answer":          item["answer"],
            "response":        response,
            "predicted_answer":predicted,
            "correct":         correct,
        })
        pbar.set_postfix({"qid": item["question_id"], "pred": predicted or "?"})

    elapsed = time.time() - total_start
    if IS_MAIN:
        print(f"\n[GPU {RANK}] ✅ {len(partial_results)} questions in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Save partial results ──────────────────────────────────────
    mode_filename = args.mode.replace(":", "_").replace("/", "_")
    partial_path = results_dir / f"partial_rank{RANK}_{mode_filename}.json"
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(partial_results, f, indent=2, ensure_ascii=False)

    cleanup_model(model)

    # ── Rank 0: wait → merge → evaluate ──────────────────────────
    if IS_MAIN:
        print(f"\n⏳ Waiting for all {WORLD_SIZE} GPU(s) to finish...")
        wait_for_partials(results_dir, WORLD_SIZE, mode_filename)

        print("🔀 Merging results from all GPUs...")
        results = merge_partial_results(results_dir, WORLD_SIZE, mode_filename)

        save_frame_indices_json(benchmark_name="VideoMME", mode=args.mode, model_name=args.model)

        video_types = [duration for duration in ["short", "medium", "long"] if any(item["duration"] == duration for item in results)]
        figures_dir  = results_dir / "figures" / mode_filename
        figures_dir.mkdir(parents=True, exist_ok=True)

        temp_path = results_dir / f"temp_{mode_filename}.json"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n📊 Evaluating results...")
        eval_results = eval_your_results(
            str(temp_path),
            video_types=video_types,
            skip_missing=args.test,
            return_categories_accuracy=True,
            return_sub_categories_accuracy=True,
            return_task_types_accuracy=True,
            save_plots=True,
            plot_output_dir=str(figures_dir),
            model_name=args.model,
            return_dict=True,
        )
        temp_path.unlink()

        total_elapsed = time.time() - total_start
        final_results = {
            "overall_performance": eval_results.get("overall_performance", 0),
            "video_durations":     eval_results.get("video_durations",     {}),
            "video_categories":    eval_results.get("video_categories",    {}),
            "video_sub_categories":eval_results.get("video_sub_categories",{}),
            "task_categories":     eval_results.get("task_categories",     {}),
            "model":  args.model,
            "mode":   args.mode,
            "num_gpus": WORLD_SIZE,
            "inference_time_seconds": round(total_elapsed, 1),
            "inference_time_minutes": round(total_elapsed / 60, 2),
            "detailed_results": results,
        }
        final_path = results_dir / f"{mode_filename}.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Final results saved to: {final_path}")


if __name__ == "__main__":
    main()
