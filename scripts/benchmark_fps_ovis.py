#!/usr/bin/env python3
"""
Ovis FPS Benchmark Script.

This script benchmarks the Ovis model with different FPS configurations
to analyze the impact of frame sampling rate on performance and accuracy.
"""

import logging
import sys
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
    setup_logging
)
from video_model_research.metrics import plt_fig
import argparse

def setup_ovis_arguments():
    """Parse and validate command-line arguments for Ovis FPS benchmark."""
    parser = argparse.ArgumentParser(description="Run Ovis model FPS benchmark.")
    parser.add_argument("--folder_path", type=str, default="./Asharq_data/videos/",
                       help="Path to the folder containing video files.")
    parser.add_argument("--output_path", type=str, default="./Asharq_data/ovis_fps_benchmark/",
                       help="Path to the output")
    parser.add_argument("--prompt", type=str, default="Give a short description of the video:",
                       help="Prompt to provide to the model.")
    parser.add_argument("--max_tokens", type=int, default=70,
                       help="Maximum number of tokens to generate.")
    parser.add_argument("--min_frames", type=int, default=1,
                       help="Minimum number of frames to sample from the video.")
    parser.add_argument("--max_frames", type=int, default=96,
                       help="Maximum number of frames to sample from the video.")
    parser.add_argument("--fps_configs", type=str, default="0.1,0.2,0.5,1,1.5,2,2.5,3,4",
                       help="Comma-separated list of FPS values to test (default: 0.2,0.5,1,1.5,2,2.5,3,4)")

    args = parser.parse_args()

    try:
        args.fps_list = [float(fps.strip()) for fps in args.fps_configs.split(',')]
    except ValueError:
        raise ValueError(f"Invalid FPS configuration: {args.fps_configs}")

    create_output_directory(args.output_path)
    log_arguments(args, "Ovis FPS Benchmark Arguments")
    return args


def load_ovis_benchmark_data(args):
    """Load and prepare benchmark data for Ovis FPS testing."""
    PROGRESS_REPORT_INTERVAL = 50
    EVALUATION_METRICS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "BERTScore", "CIDEr"]

    video_paths = load_video_files(args.folder_path)
    df = load_metadata_csv()

    return {
        'video_paths': video_paths,
        'df': df,
        'metrics': EVALUATION_METRICS,
        'progress_interval': PROGRESS_REPORT_INTERVAL,
        'fps_list': args.fps_list
    }





def main():
    """Main Ovis FPS benchmark function."""
    set_seed(42)

    args = setup_ovis_arguments()

    logger = setup_logging(args.output_path, "ovis_fps_benchmark.log")
    logger.info("Starting Ovis FPS benchmark")

    data_config = load_ovis_benchmark_data(args)

    fps_list = data_config['fps_list']
    metrics = data_config['metrics'] + ['inference_time']  # Add timing metric
    results_df = pd.DataFrame(index=[f"ovis_fps_{str(fps).replace('.', '_')}" for fps in fps_list],
                             columns=metrics)

    logger.info(f"Benchmarking Ovis with {len(fps_list)} FPS configurations on {len(data_config['video_paths'])} videos")
    logger.info(f"FPS configurations: {fps_list}")

    for fps_value in fps_list:
        logger.info(f"Processing FPS configuration: {fps_value}")
        try:
            video_config = {
                "fps": fps_value,
                "max_frames": args.max_frames,
                "min_frames": args.min_frames,
                "return_extra": False
            }

            fps_str = str(fps_value).replace('.', '_')
            scores = process_model("ovis", data_config, args, video_config, f"fps_{fps_str}")

            if scores:
                scores['fps'] = fps_value
                fps_key = f"ovis_fps_{fps_str}"
                results_df.loc[fps_key] = [scores[m] for m in metrics]
            else:
                logger.warning(f"No scores returned for FPS={fps_value}")
        except Exception as e:
            logger.error(f"Failed to process FPS={fps_value}: {e}")
            logger.info(f"Continuing with next FPS configuration...")

    results_df = finalize_results_dataframe(results_df, metrics, exclude_from_mean=['inference_time'])

    final_results_path = os.path.join(args.output_path, "ovis_fps_final_results.csv")
    results_df.to_csv(final_results_path)
    logger.info(f"Final FPS benchmark results saved to {final_results_path}")

    plot_df = results_df.drop(columns=['inference_time'])  # Remove inference_time for plotting
    fig_path = os.path.join(args.output_path, "ovis_fps_comparison.png")
    plt_fig(plot_df, fig_path)
    logger.info(f"FPS comparison visualization saved to {fig_path}")

    logger.info("Ovis FPS benchmark completed successfully!")
    logger.info("FPS Benchmark Results Summary:")
    logger.info(f"\n{results_df}")

    best_fps_idx = results_df["mean_scores"].idxmax()
    best_fps = fps_list[list(results_df.index).index(best_fps_idx)]
    best_score = results_df.loc[best_fps_idx, "mean_scores"]
    logger.info(f"\n🏆 Best FPS configuration: {best_fps} (Mean Score: {best_score:.3f})")


if __name__ == "__main__":
    main()
