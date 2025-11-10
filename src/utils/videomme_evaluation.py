"""
VideoMME evaluation utilities for analyzing model performance on Video-MME dataset.
"""

import json
from typing import List, Dict, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CATEGORIES = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual"
]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual"
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]


def create_plots(v_type_dict, v_sub_type_dict, q_type_dict, video_types, output_dir, model_name,
                 plot_categories=True, plot_sub_categories=True, plot_task_types=True):
    """Create and save plots for evaluation results"""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    saved_plots = []

    # 1. Plot Video Domains (Categories) - only if requested
    if plot_categories:
        fig, ax = plt.subplots(figsize=(12, 6))
        categories = CATEGORIES
        accuracies = []
        for v_type in categories:
            total_correct = sum([v_type_dict[video_type][v_type]["correct"] for video_type in video_types])
            total_answered = sum([v_type_dict[video_type][v_type]["answered"] for video_type in video_types])
            acc = 100 * total_correct / total_answered if total_answered > 0 else 0
            accuracies.append(acc)

        bars = ax.bar(range(len(categories)), accuracies, color='steelblue', alpha=0.8)
        ax.set_xlabel('Video Domain', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance by Video Domain - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot_file = output_path / f'{model_name}_video_domains.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(f'{model_name}_video_domains.png')

    # 2. Plot Task Categories - only if requested
    if plot_task_types:
        fig, ax = plt.subplots(figsize=(14, 6))
        task_categories = TASK_CATEGORIES
        accuracies = []
        for q_type in task_categories:
            total_correct = sum([q_type_dict[video_type][q_type]["correct"] for video_type in video_types])
            total_answered = sum([q_type_dict[video_type][q_type]["answered"] for video_type in video_types])
            acc = 100 * total_correct / total_answered if total_answered > 0 else 0
            accuracies.append(acc)

        bars = ax.bar(range(len(task_categories)), accuracies, color='coral', alpha=0.8)
        ax.set_xlabel('Task Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance by Task Category - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(task_categories)))
        ax.set_xticklabels(task_categories, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plot_file = output_path / f'{model_name}_task_categories.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(f'{model_name}_task_categories.png')

    # 3. Plot Video Sub Categories - only if requested
    if plot_sub_categories:
        fig, ax = plt.subplots(figsize=(16, 8))
        sub_categories = SUB_CATEGORIES
        accuracies = []
        for v_sub_type in sub_categories:
            total_correct = sum([v_sub_type_dict[video_type][v_sub_type]["correct"] for video_type in video_types])
            total_answered = sum([v_sub_type_dict[video_type][v_sub_type]["answered"] for video_type in video_types])
            acc = 100 * total_correct / total_answered if total_answered > 0 else 0
            accuracies.append(acc)

        bars = ax.bar(range(len(sub_categories)), accuracies, color='mediumseagreen', alpha=0.8)
        ax.set_xlabel('Video Sub Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance by Video Sub Category - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(sub_categories)))
        ax.set_xticklabels(sub_categories, rotation=90, ha='right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=7, rotation=0)

        plt.tight_layout()
        plot_file = output_path / f'{model_name}_video_sub_categories.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(f'{model_name}_video_sub_categories.png')

    if saved_plots:
        print(f"\n📊 Plots saved to: {output_path}/")
        for plot in saved_plots:
            print(f"  - {plot}")


def eval_your_results(
        your_results_path: str,
        video_types: Optional[Union[List[str], str]] = None,
        skip_missing: Optional[bool] = False,
        return_categories_accuracy: Optional[bool] = True,
        return_sub_categories_accuracy: Optional[bool] = False,
        return_task_types_accuracy: Optional[bool] = False,
        gt_answer_key: Optional[str] = "answer",
        your_answer_key: Optional[str] = "response",
        save_plots: Optional[bool] = True,
        plot_output_dir: Optional[str] = "./results/VideoMME/Figures",
        model_name: Optional[str] = "model",
        return_dict: Optional[bool] = False
    ):
    """
    Evaluate your results against the ground truth

    Args:
    - your_results_path (str): Path to your results file
    - video_types (Optional[List[str], str]): List of video types to evaluate. 
    - skip_missing (Optional[bool]): If True, missing files will be skipped. If False, an error will be raised if there are missing files.
    - return_categories_accuracy (Optional[bool]): If True, the accuracy for each video category will be returned.
    - return_sub_categories_accuracy (Optional[bool]): If True, the accuracy for each video sub category will be returned.
    - return_task_types_accuracy (Optional[bool]): If True, the accuracy for each task category will be returned.
    - gt_answer_key (Optional[str]): Key to access the ground truth answer in the results file.
    - your_answer_key (Optional[str]): Key to access your answer in the results file.
    """

    # Load your results
    with open(your_results_path, 'r') as f:
        your_results = json.load(f)

    if isinstance(video_types, str):
        video_types = video_types.split(",")

    q_type_dict = {}
    v_type_dict = {}
    v_sub_type_dict = {}

    for video_type in video_types:

        # Filter your results based on video types
        your_results_video_type = [item for item in your_results if item["duration"] == video_type]

        # Task Categories
        q_type_dict[video_type] = {}
        for q_type in TASK_CATEGORIES:
            q_type_dict[video_type][q_type] = {"correct": 0, "answered": 0}

        # Video categories
        v_type_dict[video_type] = {}
        for v_type in CATEGORIES:
            v_type_dict[video_type][v_type] = {"correct": 0, "answered": 0}
        
        v_sub_type_dict[video_type] = {}
        for v_sub_type in SUB_CATEGORIES:
            v_sub_type_dict[video_type][v_sub_type] = {"correct": 0, "answered": 0}

        if not skip_missing:
            # Check if the number of files in your results and ground truth are the same
            assert len(your_results_video_type) == 300, f"Number of files in {video_type} is not 300. Check if there are missing files."

        for item in your_results_video_type:

            if skip_missing and item.get("missing"):
                continue

            # Get the video category, sub category and question category
            video_category = item["domain"]
            video_sub_category = item["sub_category"]
            
            questions = item["questions"]

            for question in questions:
                q_type = question["task_type"]

                # Get the ground truth and your response
                gt_answer = question[gt_answer_key]
                response = question[your_answer_key]
                correct = question["correct"]

                # Extract the answer from the response
                if not response:
                    continue
                else:
                    q_type_dict[video_type][q_type]["answered"] += 1
                    q_type_dict[video_type][q_type]["correct"] += correct

                    v_type_dict[video_type][video_category]["answered"] += 1
                    v_type_dict[video_type][video_category]["correct"] += correct

                    v_sub_type_dict[video_type][video_sub_category]["answered"] += 1
                    v_sub_type_dict[video_type][video_sub_category]["correct"] += correct

    # Print the results for each video type
    for video_type in video_types:

        print("=====================================")
        print(f"Evaluation on video Type: {video_type}")
        print("=====================================")
        if return_categories_accuracy:
            print("-------------------------------------")
            print("Video Categories")
            print("-------------------------------------")
            for v_type in v_type_dict[video_type]:
                print(f"{v_type}: {100 * v_type_dict[video_type][v_type]['correct'] / v_type_dict[video_type][v_type]['answered'] if v_type_dict[video_type][v_type]['answered'] > 0 else 0 : .1f}%")
        if return_sub_categories_accuracy:
            print("-------------------------------------")
            print("Video Sub Categories")
            print("-------------------------------------")
            for v_sub_type in v_sub_type_dict[video_type]:
                print(f"{v_sub_type}: {100 * v_sub_type_dict[video_type][v_sub_type]['correct'] / v_sub_type_dict[video_type][v_sub_type]['answered'] if v_sub_type_dict[video_type][v_sub_type]['answered'] > 0 else 0 : .1f}%")
        if return_task_types_accuracy:
            print("-------------------------------------")
            print("Task Categories")
            print("-------------------------------------")
            for q_type in q_type_dict[video_type]:
                print(f"{q_type}: {100 * q_type_dict[video_type][q_type]['correct'] / q_type_dict[video_type][q_type]['answered'] if q_type_dict[video_type][q_type]['answered'] > 0 else 0 : .1f}%")
        
        print("-------------------------------------")
        print("Overall Performance")
        print("-------------------------------------")
        total_correct = sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES])
        total_answered = sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES])
        print(f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

        print("\n")

    # Print the results for the entire dataset
    print("=====================================")
    print("Evaluation on the entire dataset")
    print("=====================================")

    if return_categories_accuracy:
        print("-------------------------------------")
        print("Video Domains")
        print("-------------------------------------")
        for v_type in CATEGORIES:
            total_correct = sum([v_type_dict[video_type][v_type]["correct"] for video_type in video_types])
            total_answered = sum([v_type_dict[video_type][v_type]["answered"] for video_type in video_types])
            print(f"{v_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    
    if return_sub_categories_accuracy:
        print("-------------------------------------")
        print("Video Sub Categories")
        print("-------------------------------------")

        for v_sub_type in SUB_CATEGORIES:
            total_correct = sum([v_sub_type_dict[video_type][v_sub_type]["correct"] for video_type in video_types])
            total_answered = sum([v_sub_type_dict[video_type][v_sub_type]["answered"] for video_type in video_types])
            print(f"{v_sub_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    if return_task_types_accuracy:
        print("-------------------------------------")
        print("Task Categories")
        print("-------------------------------------")
        for q_type in TASK_CATEGORIES:

            total_correct = sum([q_type_dict[video_type][q_type]["correct"] for video_type in video_types])
            total_answered = sum([q_type_dict[video_type][q_type]["answered"] for video_type in video_types])
            print(f"{q_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    print("-------------------------------------")
    print("Overall Performance")
    print("-------------------------------------")
    total_correct = sum([sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES]) for video_type in video_types])
    total_answered = sum([sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES]) for video_type in video_types])
    overall_accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
    print(f"Overall: {overall_accuracy : .1f}%")

    # Generate plots if requested (based on the same flags as text output)
    if save_plots:
        create_plots(v_type_dict, v_sub_type_dict, q_type_dict, video_types, plot_output_dir, model_name,
                     plot_categories=return_categories_accuracy,
                     plot_sub_categories=return_sub_categories_accuracy,
                     plot_task_types=return_task_types_accuracy)

    # Return results as dictionary if requested
    if return_dict:
        results_dict = {
            "overall_performance": round(overall_accuracy, 2)
        }

        if return_categories_accuracy:
            results_dict["video_categories"] = {}
            for v_type in CATEGORIES:
                total_correct = sum([v_type_dict[video_type][v_type]["correct"] for video_type in video_types])
                total_answered = sum([v_type_dict[video_type][v_type]["answered"] for video_type in video_types])
                accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
                results_dict["video_categories"][v_type] = round(accuracy, 2)

        if return_sub_categories_accuracy:
            results_dict["video_sub_categories"] = {}
            for v_sub_type in SUB_CATEGORIES:
                total_correct = sum([v_sub_type_dict[video_type][v_sub_type]["correct"] for video_type in video_types])
                total_answered = sum([v_sub_type_dict[video_type][v_sub_type]["answered"] for video_type in video_types])
                accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
                results_dict["video_sub_categories"][v_sub_type] = round(accuracy, 2)

        if return_task_types_accuracy:
            results_dict["task_categories"] = {}
            for q_type in TASK_CATEGORIES:
                total_correct = sum([q_type_dict[video_type][q_type]["correct"] for video_type in video_types])
                total_answered = sum([q_type_dict[video_type][q_type]["answered"] for video_type in video_types])
                accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
                results_dict["task_categories"][q_type] = round(accuracy, 2)

        return results_dict

