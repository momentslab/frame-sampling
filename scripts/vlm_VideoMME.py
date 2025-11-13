import argparse
import sys
import os
from pathlib import Path
import json
import time
import torch

sys.path.append('src')

from datasets import load_dataset
from models.utils import get_model, cleanup_model
from utils.videomme_evaluation import eval_your_results
from utils.common_eval_utils import create_video_config, extract_answer, save_frame_indices_json
from utils.videomme_utils import build_video_id_map, find_video_file_by_id, create_videomme_prompt


def setup_arguments():
    parser = argparse.ArgumentParser(description="Evaluate video language models on Video_MME dataset (combined mode).")
    parser.add_argument("--model", type=str, required=True,
                        choices=["ovis", "smolvlm", "qwen2", "qwen2_5", "intern"],
                        help="Model to use for evaluation")
    parser.add_argument("--video_dir", type=str, default="./data/VideoMME",
                        help="Directory containing Video_MME videos")
    parser.add_argument("--mode", type=str, default="fps:2:4:96",
                       help="Frame mode: 'first', 'center' | 'fps:fps:min:max' | 'maxinfo:input:max' | 'csta:input:max'")
    parser.add_argument("--max_tokens", type=int, default=10,
                        help="Max tokens to generate (multiple choice)")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode with limited samples (first 3 videos)")
    return parser.parse_args()


def run_model_inference(model, video_path, prompt, args):
    video_items = create_video_config(args.model, video_path, args.mode)
    try:
        response = model.predict(video_items=video_items, prompt=prompt, max_tokens=args.max_tokens)
        print(f"🤖 Response: {response}")
        return response
    except Exception as e:
        print(f"❌ Error during model inference: {e}")
        return None


def main():
    args = setup_arguments()

    print("🎯 Video_MME Evaluation - Combined Mode")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Video directory: {args.video_dir}")
    print(f"Frame mode: {args.mode}")
    print("=" * 60)

    print("📦 Loading Video_MME dataset...")
    dataset = load_dataset("lmms-lab/Video-MME", split="test")

    print("🤖 Loading model...")
    model = get_model(args.model)
    video_id_map = build_video_id_map(args.video_dir)
    print(len(video_id_map))

    # Group questions by video_id (vid) but store videoID for matching
    video_map = {}
    for example in dataset:

        video_id = example["video_id"]    # used in results and output
        videoID = example["videoID"].lower()      # used to match actual filename
        qid = example["question_id"]
        qid_number = int(qid.split("-")[-1])

        entry = {
            "question_id": qid,
            "question": example["question"],
            "options": example["options"],
            "answer": example.get("answer", "Unknown"),
            "task_type": example.get("task_type", "Unknown"),
            "domain": example.get("domain", "Unknown"),
            "sub_category": example.get("sub_category", "Unknown"),
            "qid_number": qid_number
        }

        video_map.setdefault(video_id, {
            "videoID": videoID,
            "duration": example.get("duration", "Unknown"),
            "domain": example.get("domain", "Unknown"),
            "sub_category": example.get("sub_category", "Unknown"),
            "questions": []
        })["questions"].append(entry)

    results = []
    total_start = time.time()

    # Test mode: limit to first 3 videos
    if args.test:
        print("\n🧪 TEST MODE: Limited to first 3 videos")
        video_map = dict(list(video_map.items())[:3])

    for video_id, metadata in video_map.items():
        videoID = metadata["videoID"]
        try:
            video_path = find_video_file_by_id(videoID, video_id_map)
            video_missing = False
            print(f"\n🎬 Processing video {video_id} | File: {Path(video_path).name}")
        except Exception as e:
            print(f"\n🎬 Processing video {video_id}...")
            print(f"⚠️ Skipping video {video_id} due to error: {e}")
            video_missing = True
            video_path = None

        video_result = {
            "video_id": video_id,
            "duration": metadata["duration"],
            "domain": metadata["domain"],
            "sub_category": metadata["sub_category"],
            "questions": []
        }

        for q in metadata["questions"]:
            prompt = create_videomme_prompt(q)
            print(f"\n🔍 Q: {q['question_id']} | {q['question']}")
            if video_missing:
                response = None
                predicted = None
                correct = False
            else:
                response = run_model_inference(model, video_path, prompt, args)
                predicted = extract_answer(response)
                correct = (predicted == q["answer"])
            torch.cuda.empty_cache()

            question_entry = {
                "question_id": q["question_id"],
                "task_type": q["task_type"],
                "question": q["question"],
                "options": q["options"],
                "answer": q["answer"],
                "response": response,
                "predicted_answer": predicted,
                "correct": correct
            }
            video_result["questions"].append(question_entry)

        results.append(video_result)
        torch.cuda.empty_cache()

    # Create results directory with model-specific subdirectory
    results_dir = Path("results/VideoMME") / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    cleanup_model(model)

    print(f"\n✅ Done! Processing results...")
    print(f"⏱️ Total evaluation time: {time.time() - total_start:.2f} seconds")

    # Save frame indices
    save_frame_indices_json(benchmark_name="VideoMME", mode=args.mode, model_name=args.model)

    # Evaluate results
    print("\n" + "="*60)
    print("📊 EVALUATING RESULTS")
    print("="*60)

    # Determine video types from results
    video_types = list(set([item["duration"] for item in results]))

    # Create figures directory with mode-specific name
    mode_filename = args.mode.replace(":", "_").replace("/", "_")
    figures_dir = results_dir / "figures" / mode_filename
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary file path for evaluation
    temp_results_path = results_dir / f"temp_{mode_filename}.json"
    with open(temp_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    eval_results = eval_your_results(
        str(temp_results_path),
        video_types=video_types,
        skip_missing=args.test,  # Skip missing check in test mode
        return_categories_accuracy=True,
        return_sub_categories_accuracy=True,
        return_task_types_accuracy=True,
        save_plots=True,
        plot_output_dir=str(figures_dir),
        model_name=args.model,
        return_dict=True
    )

    # Remove temporary file
    temp_results_path.unlink()

    # Create final results file with summary at the top
    final_results = {
        "overall_performance": eval_results.get("overall_performance", 0),
        "video_categories": eval_results.get("video_categories", {}),
        "video_sub_categories": eval_results.get("video_sub_categories", {}),
        "task_categories": eval_results.get("task_categories", {}),
        "model": args.model,
        "mode": args.mode,
        "detailed_results": results
    }

    # Save final results with summary at top
    final_out_path = results_dir / f"{mode_filename}.json"
    with open(final_out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Final results saved to: {final_out_path}")


if __name__ == "__main__":
    main()
