#!/usr/bin/env python3
"""
Clean and simplified video model benchmark script.

This script uses utility functions from bench_utils.py to keep the main logic
clean and focused on orchestration.
"""

import os
import pandas as pd

from video_model_research.bench_utils import (
    set_seed,
    load_video_files,
    load_metadata_csv,
    create_output_directory,
    log_arguments,
    finalize_results_dataframe,
    process_model,
    parse_frame_mode,
    setup_logging
)
from video_model_research.metrics import plt_fig
import argparse


def setup_arguments():
    """Parse and validate command-line arguments for standard benchmark."""
    parser = argparse.ArgumentParser(description="Run video language models benchmark.")
    parser.add_argument("--folder_path", type=str, default="./Asharq_data/videos/",
                       help="Path to the folder containing video files.")
    parser.add_argument("--output_path", type=str, default="./Asharq_data/benchmark_models/",
                       help="Path to the output")
    parser.add_argument("--prompt", type=str, default="Give a short description of the video:",
                       help="Prompt to provide to the model.")
    parser.add_argument("--max_tokens", type=int, default=70,
                       help="Maximum number of tokens to generate.")
    parser.add_argument("--mode", type=str, default="fps:1:4:96",
                       help="Frame mode: 'first', 'center' | 'fps:fps:min:max' | 'maxinfo:input:max' | 'csta:input:max'")

    args = parser.parse_args()
    create_output_directory(args.output_path)
    log_arguments(args, "Benchmark Arguments")
    return args


def load_benchmark_data(args):
    """Load and prepare benchmark data for model comparison."""
    PROGRESS_REPORT_INTERVAL = 40
    SUPPORTED_MODELS = ["smolvlm", "qwen2", "qwen2_5", "intern", "ovis"]
    EVALUATION_METRICS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "BERTScore", "CIDEr", "inference_time"]

    frame_config = parse_frame_mode(args.mode)
    base_video_config = {
        **frame_config,
        "return_extra": False
    }

    video_paths = load_video_files(args.folder_path)
    df = load_metadata_csv()

    return {
        'base_video_config': base_video_config,
        'video_paths': video_paths,
        'df': df,
        'metrics': EVALUATION_METRICS,
        'models': SUPPORTED_MODELS,
        'progress_interval': PROGRESS_REPORT_INTERVAL
    }


def main():
    """Main benchmark function - clean and focused on orchestration."""
    set_seed(42)

    args = setup_arguments()

    logger = setup_logging(args.output_path, "benchmark.log")
    logger.info("Starting video model benchmark")

    data_config = load_benchmark_data(args)
    
    models = data_config['models']
    metrics = data_config['metrics']
    results_df = pd.DataFrame(index=models, columns=metrics)
    
    logger.info(f"Benchmarking {len(models)} models on {len(data_config['video_paths'])} videos")
    
    for model_name in models:
        logger.info(f"Processing model: {model_name}")
        try:
            scores = process_model(model_name, data_config, args)
            if scores:
                results_df.loc[model_name] = [scores[m] for m in metrics]
            else:
                logger.warning(f"No scores returned for {model_name}")
        except Exception as e:
            logger.error(f"Failed to process model {model_name}: {e}")
            logger.info(f"Continuing with next model...")
    
    results_df = finalize_results_dataframe(results_df, metrics, exclude_from_mean=['inference_time'])

    final_results_path = os.path.join(args.output_path, "final_results.csv")
    results_df.to_csv(final_results_path)
    logger.info(f"Final results saved to {final_results_path}")

    plot_df = results_df.drop(columns=['inference_time'])  # Remove inference_time for plotting
    fig_path = os.path.join(args.output_path, "fig.png")
    plt_fig(plot_df, fig_path)
    logger.info(f"Results visualization saved to {fig_path}")

    logger.info("Benchmark completed successfully!")
    logger.info("Results summary:")
    logger.info(f"\n{results_df}")


if __name__ == "__main__":
    main()
