#!/usr/bin/env python3
"""
Single Frame Benchmark Script.

This script benchmarks all available models using only single frame extraction modes
('first' and 'center') to analyze how models perform with minimal frame input.
This is useful for understanding model efficiency and performance with reduced data.
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

def setup_single_frame_arguments():
    """Parse and validate command-line arguments for single frame benchmark."""
    parser = argparse.ArgumentParser(description="Run single frame benchmark across all models.")
    parser.add_argument("--folder_path", type=str, default="./Asharq_data/videos/",
                       help="Path to the folder containing video files.")
    parser.add_argument("--output_path", type=str, default="./Asharq_data/single_frame_benchmark/",
                       help="Path to the output directory.")
    parser.add_argument("--prompt", type=str, default="Give a short description of the video:",
                       help="Prompt to provide to the model.")
    parser.add_argument("--max_tokens", type=int, default=70,
                       help="Maximum number of tokens to generate.")
    parser.add_argument("--models", type=str, default="smolvlm,qwen2,qwen2_5,intern,ovis",
                       help="Comma-separated list of models to test (default: all models)")
    parser.add_argument("--frame_modes", type=str, default="first,center",
                       help="Comma-separated list of single frame modes to test (default: first,center)")

    args = parser.parse_args()

    try:
        args.model_list = [model.strip() for model in args.models.split(',')]
    except ValueError:
        raise ValueError(f"Invalid models configuration: {args.models}")

    try:
        args.frame_mode_list = [mode.strip() for mode in args.frame_modes.split(',')]
        for mode in args.frame_mode_list:
            if mode not in ["first", "center"]:
                raise ValueError(f"Invalid frame mode: {mode}. Only 'first' and 'center' are supported.")
    except ValueError as e:
        raise ValueError(f"Invalid frame mode configuration: {e}")

    create_output_directory(args.output_path)
    log_arguments(args, "Single Frame Benchmark Arguments")
    return args


def load_single_frame_benchmark_data(args):
    """Load and prepare benchmark data for single frame testing."""
    PROGRESS_REPORT_INTERVAL = 50
    EVALUATION_METRICS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "BERTScore", "CIDEr"]

    video_paths = load_video_files(args.folder_path)
    df = load_metadata_csv()

    return {
        'video_paths': video_paths,
        'df': df,
        'metrics': EVALUATION_METRICS,
        'progress_interval': PROGRESS_REPORT_INTERVAL,
        'model_list': args.model_list,
        'frame_mode_list': args.frame_mode_list
    }


def main():
    """Main single frame benchmark function."""
    set_seed(42)

    args = setup_single_frame_arguments()

    logger = setup_logging(args.output_path, "single_frame_benchmark.log")
    logger.info("Starting Single Frame Benchmark across all models")
    logger.info(f"Testing frame modes: {args.frame_mode_list}")

    data_config = load_single_frame_benchmark_data(args)

    model_list = data_config['model_list']
    frame_mode_list = data_config['frame_mode_list']
    metrics = data_config['metrics'] + ['inference_time']  # Add timing metric
    
    index_names = []
    for model in model_list:
        for frame_mode in frame_mode_list:
            index_names.append(f"{model}_{frame_mode}")
    
    results_df = pd.DataFrame(index=index_names, columns=metrics)

    logger.info(f"Benchmarking {len(model_list)} models with {len(frame_mode_list)} frame modes")
    logger.info(f"Models: {model_list}")
    logger.info(f"Frame modes: {frame_mode_list}")
    logger.info(f"Total combinations: {len(index_names)}")

    for model_name in model_list:
        for frame_mode in frame_mode_list:
            combination_key = f"{model_name}_{frame_mode}"
            logger.info(f"Processing: {model_name} with {frame_mode} frame")
            
            try:
                video_config = {
                    **parse_frame_mode(frame_mode),
                    "return_extra": True if model_name == "smolvlm" else False
                }

                scores = process_model(model_name, data_config, args, video_config, combination_key)

                if scores:
                    scores['model'] = model_name
                    scores['frame_mode'] = frame_mode
                    results_df.loc[combination_key] = [scores[m] for m in metrics]
                    logger.info(f"✅ Completed {combination_key}")
                else:
                    logger.warning(f"❌ No scores returned for {combination_key}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {combination_key}: {e}")
                logger.info(f"Continuing with next combination...")

    results_df = finalize_results_dataframe(results_df, metrics, exclude_from_mean=['inference_time'])

    final_results_path = os.path.join(args.output_path, "single_frame_final_results.csv")
    results_df.to_csv(final_results_path)
    logger.info(f"Final single frame benchmark results saved to {final_results_path}")

    plot_df = results_df.drop(columns=['inference_time'])
    fig_path = os.path.join(args.output_path, "single_frame_comparison.png")
    plt_fig(plot_df, fig_path)
    logger.info(f"Single frame comparison visualization saved to {fig_path}")

    logger.info("Single Frame Benchmark completed successfully!")
    logger.info("Single Frame Benchmark Results Summary:")
    logger.info(f"\n{results_df}")

    best_combination_idx = results_df["mean_scores"].idxmax()
    best_score = results_df.loc[best_combination_idx, "mean_scores"]
    logger.info(f"\n🏆 Best combination: {best_combination_idx} (Mean Score: {best_score:.3f})")

    logger.info("\n📊 Analysis by Frame Mode:")
    for frame_mode in frame_mode_list:
        frame_mode_results = results_df[results_df.index.str.endswith(f"_{frame_mode}")]
        if not frame_mode_results.empty:
            best_model_for_mode = frame_mode_results["mean_scores"].idxmax()
            best_score_for_mode = frame_mode_results.loc[best_model_for_mode, "mean_scores"]
            logger.info(f"  {frame_mode.upper()} frame - Best: {best_model_for_mode} (Score: {best_score_for_mode:.3f})")

    logger.info("\n📊 Analysis by Model:")
    for model in model_list:
        model_results = results_df[results_df.index.str.startswith(f"{model}_")]
        if not model_results.empty:
            best_mode_for_model = model_results["mean_scores"].idxmax()
            best_score_for_model = model_results.loc[best_mode_for_model, "mean_scores"]
            logger.info(f"  {model.upper()} - Best: {best_mode_for_model} (Score: {best_score_for_model:.3f})")


if __name__ == "__main__":
    main()
