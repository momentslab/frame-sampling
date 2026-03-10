#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ["intern", "smolvlm", "qwen2_5", "ovis", "ours"]
MODEL_LABELS = {
    "intern": "InternVL",
    "smolvlm": "SmolVLM",
    "qwen2_5": "Qwen2.5",
    "ovis": "Ovis",
    "ours": "Ours",
}
MODEL_MARKERS = {
    "intern": "o",
    "smolvlm": "s",
    "qwen2_5": "^",
    "ovis": "D",
    "ours": "P",
}
DURATION_ORDER = ["short", "medium", "long"]
METRICS = {
    "global": "Overall",
    "short": "Short",
    "medium": "Medium",
    "long": "Long",
}
CONFIG_ORDER = [
    "clips_8_1_2",
    "clips_16_1_2",
    "clips_16_2_2",
    "clips_16_4_2",
    "clips_16_8_2",
    "clips_16_16_2",
]
CONFIG_LABELS = {
    "clips_8_1_2": "8x1@2",
    "clips_16_1_2": "16x1@2",
    "clips_16_2_2": "16x2@2",
    "clips_16_4_2": "16x4@2",
    "clips_16_8_2": "16x8@2",
    "clips_16_16_2": "16x16@2",
}
CONFIG_AXIS_LABELS = {
    "clips_8_1_2": "8f / 1c\n8 total",
    "clips_16_1_2": "16f / 1c\n16 total",
    "clips_16_2_2": "16f / 2c\n32 total",
    "clips_16_4_2": "16f / 4c\n64 total",
    "clips_16_8_2": "16f / 8c\n128 total",
    "clips_16_16_2": "16f / 16c\n256 total",
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OURS_EVAL_DIRS = {
    "clips_8_1_2": REPO_ROOT / "outputs" / "stage3-siglip-perceiver-256" / "eval_results" / "outputs_8f_1clips_stage3-siglip-perceiver-256",
    "clips_16_1_2": REPO_ROOT / "outputs" / "stage3-siglip-perceiver-256" / "eval_results" / "outputs_16f_1clips_stage3-siglip-perceiver-256",
    "clips_16_2_2": REPO_ROOT / "outputs" / "stage3-siglip-perceiver-256" / "eval_results" / "outputs_16f_2clips_stage3-siglip-perceiver-256",
    "clips_16_4_2": REPO_ROOT / "outputs" / "stage3-siglip-perceiver-256" / "eval_results" / "outputs_16f_4clips_stage3-siglip-perceiver-256",
    "clips_16_8_2": REPO_ROOT / "outputs" / "stage3-siglip-perceiver-256" / "eval_results" / "outputs_16f_8clips_stage3-siglip-perceiver-256",
    "clips_16_16_2": REPO_ROOT / "outputs" / "stage3-siglip-perceiver-256" / "eval_results" / "outputs_16f_16clips_stage3-siglip-perceiver-256",
}


def compute_accuracy(correct: int, answered: int) -> float:
    return 100.0 * correct / answered if answered > 0 else 0.0


def compute_metrics_from_detailed_results(detailed_results: list[dict]) -> dict[str, dict | float]:
    duration_stats = {duration: {"correct": 0, "answered": 0} for duration in DURATION_ORDER}
    category_stats: dict[str, dict[str, int]] = {}
    sub_category_stats: dict[str, dict[str, int]] = {}
    task_stats: dict[str, dict[str, int]] = {}

    for item in detailed_results:
        if item.get("missing"):
            continue

        duration = item.get("duration")
        domain = item.get("domain")
        sub_category = item.get("sub_category")

        for question in item.get("questions", []):
            correct = int(bool(question.get("correct")))
            task_type = question.get("task_type")

            if duration in duration_stats:
                duration_stats[duration]["answered"] += 1
                duration_stats[duration]["correct"] += correct

            if domain is not None:
                category_stats.setdefault(domain, {"correct": 0, "answered": 0})
                category_stats[domain]["answered"] += 1
                category_stats[domain]["correct"] += correct

            if sub_category is not None:
                sub_category_stats.setdefault(sub_category, {"correct": 0, "answered": 0})
                sub_category_stats[sub_category]["answered"] += 1
                sub_category_stats[sub_category]["correct"] += correct

            if task_type is not None:
                task_stats.setdefault(task_type, {"correct": 0, "answered": 0})
                task_stats[task_type]["answered"] += 1
                task_stats[task_type]["correct"] += correct

    total_correct = sum(values["correct"] for values in task_stats.values())
    total_answered = sum(values["answered"] for values in task_stats.values())

    return {
        "overall_performance": round(compute_accuracy(total_correct, total_answered), 2),
        "video_durations": {
            duration: round(compute_accuracy(values["correct"], values["answered"]), 2)
            for duration, values in duration_stats.items()
        },
        "video_categories": {
            name: round(compute_accuracy(values["correct"], values["answered"]), 2)
            for name, values in category_stats.items()
        },
        "video_sub_categories": {
            name: round(compute_accuracy(values["correct"], values["answered"]), 2)
            for name, values in sub_category_stats.items()
        },
        "task_categories": {
            name: round(compute_accuracy(values["correct"], values["answered"]), 2)
            for name, values in task_stats.items()
        },
    }


def find_single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under {directory}")
    return matches[-1]


def compute_duration_metrics_from_samples(samples_path: Path) -> dict[str, float]:
    stats = {duration: {"correct": 0, "answered": 0} for duration in DURATION_ORDER}

    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            score = record.get("videomme_perception_score", {})
            duration = score.get("duration")
            if duration not in stats:
                continue
            stats[duration]["answered"] += 1
            stats[duration]["correct"] += int(score.get("pred_answer") == score.get("answer"))

    return {duration: round(compute_accuracy(values["correct"], values["answered"]), 2) for duration, values in stats.items()}


def enrich_result_json(result_path: Path) -> dict:
    data = json.loads(result_path.read_text())
    changed = False
    model_name = result_path.parent.name
    config_name = result_path.stem

    if "detailed_results" in data:
        recomputed_metrics = compute_metrics_from_detailed_results(data["detailed_results"])
        for key, value in recomputed_metrics.items():
            if data.get(key) != value:
                data[key] = value
                changed = True

    if "overall_performance" not in data:
        videomme = data.get("results", {}).get("videomme", {})
        accuracy = next(
            (float(value) for key, value in videomme.items() if key.startswith("videomme_perception_score")),
            None,
        )
        total_time = data.get("total_evaluation_time_seconds")
        if accuracy is not None:
            data["overall_performance"] = round(float(accuracy), 2)
            changed = True
        if total_time is not None:
            time_seconds = float(total_time)
            data["inference_time_seconds"] = round(time_seconds, 1)
            data["inference_time_minutes"] = round(time_seconds / 60.0, 2)
            changed = True

    if "video_durations" not in data:
        durations = None
        if model_name == "ours":
            eval_dir = OURS_EVAL_DIRS.get(config_name)
            if eval_dir and eval_dir.exists():
                samples_path = find_single_file(eval_dir, "*_samples_videomme.jsonl")
                durations = compute_duration_metrics_from_samples(samples_path)
                data["source_samples_videomme"] = str(samples_path)
        if durations is not None:
            data["video_durations"] = durations
            changed = True

    if changed:
        result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    return data


def extract_metrics(result_path: Path, metric_name: str) -> tuple[float, float]:
    data = enrich_result_json(result_path)
    if metric_name == "global":
        accuracy = data.get("overall_performance")
    else:
        accuracy = data.get("video_durations", {}).get(metric_name)

    total_time = data.get("inference_time_seconds")
    if accuracy is None or total_time is None:
        raise ValueError(f"Could not extract {metric_name} accuracy/time from {result_path}")
    return float(accuracy), float(total_time)


def collect_points(results_dir: Path, metric_name: str) -> tuple[dict[str, list[dict]], list[dict]]:
    series: dict[str, list[dict]] = {}
    rows: list[dict] = []

    for model in MODELS:
        points: list[dict] = []
        for config in CONFIG_ORDER:
            result_path = results_dir / model / f"{config}.json"
            if not result_path.exists():
                break
            accuracy, time_seconds = extract_metrics(result_path, metric_name)
            point = {
                "model": model,
                "config": config,
                "config_label": CONFIG_LABELS[config],
                "metric": metric_name,
                "accuracy": accuracy,
                "time_seconds": time_seconds,
                "time_minutes": time_seconds / 60.0,
            }
            points.append(point)
            rows.append(point)
        series[model] = points

    return series, rows


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "model", "config", "config_label", "accuracy", "time_seconds", "time_minutes"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def attach_within_group_x_positions(series: dict[str, list[dict]]) -> None:
    base_positions = {config: idx for idx, config in enumerate(CONFIG_ORDER)}
    max_offset = 0.30

    grouped_points: dict[str, list[dict]] = {config: [] for config in CONFIG_ORDER}
    for points in series.values():
        for point in points:
            grouped_points[point["config"]].append(point)

    for config, points in grouped_points.items():
        if not points:
            continue

        times = [point["time_seconds"] for point in points]
        min_time = min(times)
        max_time = max(times)
        center_x = base_positions[config]

        for point in points:
            if max_time == min_time:
                normalized = 0.5
            else:
                normalized = (point["time_seconds"] - min_time) / (max_time - min_time)

            point["x_position"] = center_x - max_offset + normalized * (2 * max_offset)


def plot_pareto(series: dict[str, list[dict]], output_path: Path, metric_name: str) -> None:
    attach_within_group_x_positions(series)

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = plt.get_cmap("tab10").colors

    for index, model in enumerate(MODELS):
        points = series.get(model, [])
        if not points:
            continue
        color = colors[index % len(colors)]
        marker = MODEL_MARKERS[model]
        xs = [point["x_position"] for point in points]
        ys = [point["accuracy"] for point in points]
        ax.scatter(xs, ys, s=80, color=color, marker=marker, label=MODEL_LABELS[model], zorder=3)
        for point in points:
            x = point["x_position"]
            y = point["accuracy"]
            ax.annotate(
                f"{point['time_minutes']:.1f}m",
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=color,
                zorder=4,
            )

    ax.set_title(f"VideoMME {METRICS[metric_name].lower()} accuracy with within-group inference-time placement")
    ax.set_xlabel("Frame / clip configuration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(len(CONFIG_ORDER)))
    ax.set_xticklabels([CONFIG_AXIS_LABELS[config] for config in CONFIG_ORDER])
    ax.set_xlim(-0.5, len(CONFIG_ORDER) - 0.5)
    for separator in range(len(CONFIG_ORDER) - 1):
        ax.axvline(separator + 0.5, color="0.68", linestyle="--", linewidth=1.0, zorder=1)
    all_accuracies = [point["accuracy"] for points in series.values() for point in points]
    if all_accuracies:
        y_min = min(all_accuracies) - 2.0
        y_max = max(all_accuracies) + 2.0
        ax.set_ylim(y_min, y_max)
        tick_start = math.floor(y_min / 2.0) * 2
        tick_end = math.ceil(y_max / 2.0) * 2
        ax.set_yticks(range(int(tick_start), int(tick_end) + 1, 2))
    ax.grid(True, alpha=0.3)
    ax.text(
        0.01,
        0.98,
        "Within each total-frame group:\nleft = faster, right = slower\nspacing is normalized per group",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(0.0, 0.88))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    default_results_dir = SCRIPT_DIR.parent / "results" / "VideoMME"

    parser = argparse.ArgumentParser(description="Plot VideoMME inference-time vs accuracy curves.")
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    parser.add_argument("--output-dir", type=Path, default=default_results_dir / "figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric_name in METRICS:
        series, rows = collect_points(args.results_dir, metric_name)
        plot_path = args.output_dir / f"videomme_pareto_{metric_name}_accuracy_vs_time.png"
        csv_path = args.output_dir / f"videomme_pareto_{metric_name}_accuracy_vs_time.csv"
        plot_pareto(series, plot_path, metric_name)
        write_summary_csv(rows, csv_path)
        print(f"Saved plot: {plot_path}")
        print(f"Saved summary: {csv_path}")
        for model in MODELS:
            configs = ", ".join(point["config"] for point in series.get(model, [])) or "<none>"
            print(f"[{metric_name}] {model}: {configs}")


if __name__ == "__main__":
    main()