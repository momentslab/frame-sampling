import argparse
import sys
import os
from pathlib import Path
import json
import time
import re
import torch

sys.path.append('src')

from datasets import load_dataset
from video_model_research.bench_utils import get_model, parse_frame_mode, cleanup_model


def setup_arguments():
    parser = argparse.ArgumentParser(description="Evaluate video language models on Video_MME dataset (combined mode).")
    parser.add_argument("--model", type=str, required=True,
                        choices=["ovis", "smolvlm", "qwen2", "qwen2_5", "intern"],
                        help="Model to use for evaluation")
    parser.add_argument("--video_dir", type=str, default="./Video_MME",
                        help="Directory containing Video_MME videos")
    parser.add_argument("--mode", type=str, default="fps:1:4:96",
                        help="Frame mode")
    parser.add_argument("--max_tokens", type=int, default=10,
                        help="Max tokens to generate (multiple choice)")
    return parser.parse_args()


def build_video_id_map(video_dir):
    video_map = {}
    video_dir = Path(video_dir)

    for subdir in video_dir.iterdir():
        if subdir.is_dir():
            for ext in ['mp4', 'avi', 'mov', 'mkv']:
                for file in subdir.glob(f"*.{ext}"):
                    match = re.search(r'^(\d+)_(.+?)\.(mp4|avi|mov|mkv)$', file.name)
                    if match:
                        video_id = match.group(2).lower()
                        video_map[video_id] = str(file)
                    else:
                        print(f"⚠️ Skipping file (no matching pattern): {file}")

    return video_map


def find_video_file_by_id(video_id, video_id_map):
    stripped_id = video_id.strip()
    if stripped_id in video_id_map:
        print("✅ Found match!")
        return video_id_map[stripped_id]
    else:
        raise FileNotFoundError(f"❌ Video file not found for ID: '{video_id}'")


def create_prompt(question_data):
    question = question_data["question"]
    options = "\n".join(question_data["options"])
    return (
        "Select the best answer to the following multiple-choice question based on the video.\n"
        "Respond with only the letter (A, B, C, or D) of the correct option.\n\n"
        f"{question}\n{options}\n\nThe best answer is:"
    )


def run_model_inference(model, video_path, prompt, args):
    frame_config = parse_frame_mode(args.mode)
    video_items = {
        "video": video_path,
        "return_extra": True if args.model == "smolvlm" else False,
        **frame_config
    }
    try:
        response = model.predict(video_items=video_items, prompt=prompt, max_tokens=args.max_tokens)
        print(f"🤖 Response: {response}")
        return response
    except Exception as e:
        print(f"❌ Error during model inference: {e}")
        return None


def extract_answer(response):
    if not response:
        return None
    response_upper = response.upper().strip()
    for option in ['A', 'B', 'C', 'D']:
        if response_upper.startswith(option):
            return option
    for option in ['A', 'B', 'C', 'D']:
        if option in response_upper:
            return option
    print(f"⚠️ Could not extract valid answer from: {response}")
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

    for video_id, metadata in video_map.items():
        print(f"\n🎬 Processing video {video_id}...")
        videoID = metadata["videoID"]
        try:
            video_path = find_video_file_by_id(videoID, video_id_map)
            video_missing = False
        except Exception as e:
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
            prompt = create_prompt(q)
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

    out_path = Path(args.video_dir) / f"results_{args.model}_{args.mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    cleanup_model(model)

    print(f"\n✅ Done! Results saved to {out_path}")
    print(f"⏱️ Total evaluation time: {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
