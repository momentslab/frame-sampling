#!/usr/bin/env python3
"""
MVBench Evaluation Script using utils.py functions.

This script uses the functions from utils.py:
- mvbench_doc_to_text(): Format prompt with question and options
- mcq_acc(): Calculate accuracy using MVBench's official method
- mvbench_process_results(): Process individual results
- mvbench_aggregate_results(): Aggregate results across all samples
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import string

sys.path.append('src')
sys.path.append('.')

from models.utils import get_model, cleanup_model
from utils.mvbench_utils import MVBENCH_TASKS, MVBENCH_DATA_MAPPING
from utils.common_eval_utils import create_video_config, save_frame_indices_json
from utils.utils import mvbench_doc_to_text, mcq_acc, mvbench_aggregate_results, get_gt_option_letter 

from datasets import load_dataset


def setup_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MVBench evaluation using utils.py functions.")
    parser.add_argument("--model", type=str, required=True,
                       choices=["ovis", "smolvlm", "qwen2", "qwen2_5", "intern"],
                       help="Model to use for evaluation")
    parser.add_argument("--data_dir", type=str, default="./data/MVBench",
                       help="Path to MVBench data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save evaluation results")
    parser.add_argument("--mode", type=str, default="fps:2:4:96",
                       help="Frame mode configuration")
    parser.add_argument("--max_tokens", type=int, default=16,
                       help="Maximum tokens to generate")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit samples per task for testing")
    parser.add_argument("--test", action="store_true",
                       help="Test mode with limited samples")

    args = parser.parse_args()

    # Set output directory based on model if not specified
    if args.output_dir is None:
        args.output_dir = f"./results/MVBench/{args.model}/"

    if args.test:
        args.limit = 5
        print("🧪 TEST MODE: Limited to 5 samples per task")
    else:
        print("🚀 FULL MODE: Processing all samples")

    return args


def process_sample_with_utils_py(model, doc: Dict, task_name: str, args) -> Dict[str, Any]:
    """
    Process a single MVBench sample using utils.py functions.
    
    Uses:
    - mvbench_doc_to_text(): Format the prompt
    - mcq_acc(): Calculate accuracy
    """
    try:
        video_name = doc.get("video", "")
        if not video_name:
            return None

        # Construct video path
        mvbench_base = args.data_dir
        dataset_folder = MVBENCH_DATA_MAPPING.get(task_name, "")
        if not dataset_folder:
            return None

        video_path = os.path.join(mvbench_base, dataset_folder, video_name)
        if not os.path.exists(video_path):
            return None
        
        # USE utils.py: Format prompt using mvbench_doc_to_text
        lmms_eval_kwargs = {
            "post_prompt": "Only give the best option."}
        prompt = mvbench_doc_to_text(doc, lmms_eval_kwargs)
        
        # Create video config
        video_config = create_video_config(args.model, video_path, args.mode)
        
        # Run inference
        start_time = time.time()
        response = model.predict(
            video_items=video_config,
            prompt=prompt,
            max_tokens=args.max_tokens
        )
        inference_time = time.time() - start_time
        
        # Get ground truth option letter
        gt_option_letter = get_gt_option_letter(doc)
        
        # USE utils.py: Calculate accuracy using mcq_acc
        score = mcq_acc(gt_option_letter, response)
        
        result = {
            "video": video_name,
            "question": doc.get("question", ""),
            "candidates": doc.get("candidates", []),
            "prompt": prompt,
            "response": response,
            "ground_truth": doc.get("answer", ""),
            "gt_option_letter": gt_option_letter,
            "score": score,
            "inference_time": inference_time,
            "task": task_name,
            "video_path": video_path
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing sample: {e}")
        return None


def main():
    args = setup_arguments()
    
    print("\n" + "=" * 60)
    print("🎬 MVBENCH EVALUATION USING utils.py")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Using functions from utils.py:")
    print(f"  - mvbench_doc_to_text(): Format prompt")
    print(f"  - mcq_acc(): Calculate accuracy")
    print(f"  - mvbench_aggregate_results(): Aggregate scores")
    print("=" * 60 + "\n")
    
    # Load model
    print(f"🤖 Loading model: {args.model}")
    model = get_model(args.model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process tasks
    all_results = []
    task_metrics = {}
    
    for task_name in MVBENCH_TASKS.keys():  # Process all tasks
        print(f"\n🎯 Evaluating {task_name}")
        print("-" * 60)
        
        try:
            dataset = load_dataset(args.data_dir, name=task_name, split="train")
            
            if args.limit:
                dataset = dataset.select(range(min(args.limit, len(dataset))))
            
            task_results = []
            total_samples = len(dataset)

            for idx, sample in enumerate(dataset, 1):
                print(f"Processing sample {idx}/{total_samples}: {sample.get('video', 'unknown')}")

                result = process_sample_with_utils_py(model, sample, task_name, args)
                if result:
                    task_results.append(result)
                    all_results.append(result)

            # Calculate missing samples
            missing_samples = total_samples - len(task_results)

            # USE utils.py: Aggregate results using mvbench_aggregate_results
            accuracy = mvbench_aggregate_results([{"pred_answer": r["response"], "score": r["score"]} for r in task_results])

            task_metrics[task_name] = {
                "accuracy": accuracy,
                "samples": len(task_results),
                "missing_samples": missing_samples,
                "avg_inference_time": sum(r["inference_time"] for r in task_results) / len(task_results) if task_results else 0
            }

            print(f"\n📊 {task_name} Results:")
            print(f"   Accuracy: {accuracy:.1f}%")
            print(f"   Samples: {len(task_results)}")
            print(f"   Missing Samples: {missing_samples}")
            print(f"   Avg Inference Time: {task_metrics[task_name]['avg_inference_time']:.2f}s")
            
        except Exception as e:
            print(f"❌ Error processing task {task_name}: {e}")
    
    # Calculate summary statistics
    accuracies = [metrics["accuracy"] for metrics in task_metrics.values()]
    total_samples_all = sum(metrics["samples"] for metrics in task_metrics.values())
    total_missing_samples = sum(metrics["missing_samples"] for metrics in task_metrics.values())

    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    max_accuracy = max(accuracies) if accuracies else 0
    min_accuracy = min(accuracies) if accuracies else 0

    # Find tasks with max and min accuracy
    max_task = max(task_metrics.items(), key=lambda x: x[1]["accuracy"])[0] if task_metrics else None
    min_task = min(task_metrics.items(), key=lambda x: x[1]["accuracy"])[0] if task_metrics else None

    summary_stats = {
        "average_accuracy": round(avg_accuracy, 2),
        "max_accuracy": round(max_accuracy, 2),
        "max_accuracy_task": max_task,
        "min_accuracy": round(min_accuracy, 2),
        "min_accuracy_task": min_task,
        "total_samples": total_samples_all,
        "total_missing_samples": total_missing_samples,
        "total_tasks": len(task_metrics)
    }

    # Save results with mode-based filename
    # Convert mode to filename format (e.g., "fps:2:4:96" -> "fps_2_4_96")
    mode_filename = args.mode.replace(":", "_")
    results_file = os.path.join(args.output_dir, f"{mode_filename}.json")
    with open(results_file, 'w') as f:
        json.dump({
            "summary_stats": summary_stats,
            "model": args.model,
            "mode": args.mode,
            "task_metrics": task_metrics,
            "detailed_results": all_results
        }, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("📊 SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Max Accuracy: {max_accuracy:.2f}% ({max_task})")
    print(f"Min Accuracy: {min_accuracy:.2f}% ({min_task})")
    print(f"Total Samples: {total_samples_all}")
    print(f"Total Missing Samples: {total_missing_samples}")
    print(f"Total Tasks: {len(task_metrics)}")
    print("=" * 60)
    
    # Save frame indices
    save_frame_indices_json(benchmark_name="MVBench", mode=args.mode, model_name=args.model)
    
    # Cleanup
    cleanup_model(model)
    
    print("\n🎉 Evaluation completed!")


if __name__ == "__main__":
    main()

