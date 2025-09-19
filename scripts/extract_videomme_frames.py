"""Utilities to extract frame selections for Video-MME videos."""

from __future__ import annotations

import argparse
import ast
import csv
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
from datasets import load_dataset
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames for Video-MME videos based on pre-computed indices. "
            "The CSV is expected to contain a header named 'values' where each row "
            "is a Python-style list of frame indices."),
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Path to the CSV file containing frame indices",
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        type=Path,
        help="Directory that contains the 900 Video-MME videos",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/frame_exports"),
        type=Path,
        help="Destination directory for the exported frames",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help=(
            "Reserved for future parallel extraction. Currently unused; passing a "
            "value other than 0 will raise an error."),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing frames that already exist on disk",
    )
    return parser.parse_args()


def load_frame_indices(csv_path: Path) -> List[List[int]]:
    """Parse a CSV file containing literal frame lists."""
    rows: List[List[int]] = []
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if "values" not in (reader.fieldnames or []):
            raise ValueError("CSV file must contain a 'values' column")
        for line_number, row in enumerate(reader, start=2):
            raw_value = row.get("values", "").strip()
            if not raw_value:
                LOGGER.warning("Skipping empty row at line %s", line_number)
                continue
            try:
                parsed = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Failed to parse frame list at line {line_number}: {raw_value}") from exc
            if not isinstance(parsed, Iterable):
                raise ValueError(f"Frame specification at line {line_number} is not iterable: {raw_value}")
            frames = [int(idx) for idx in parsed]
            rows.append(frames)
    if not rows:
        raise ValueError(f"No frame indices parsed from {csv_path}")
    return rows


def build_video_id_map(video_dir: Path) -> Dict[str, Path]:
    """Map normalized video identifiers to file paths."""
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_map: Dict[str, Path] = {}
    pattern = re.compile(r"^(?:\d+_)?(.+?)\.(mp4|avi|mov|mkv)$", re.IGNORECASE)

    for file in video_dir.rglob("*"):
        if not file.is_file():
            continue
        suffix = file.suffix.lower()
        if suffix not in valid_extensions:
            continue
        match = pattern.match(file.name)
        if not match:
            LOGGER.debug("Skipping file without Video-MME pattern: %s", file)
            continue
        video_key = match.group(1).lower()
        if video_key in video_map:
            LOGGER.warning("Duplicate video key '%s' detected; keeping the first occurrence", video_key)
            continue
        video_map[video_key] = file
    if not video_map:
        raise FileNotFoundError(f"No video files found under {video_dir}")
    LOGGER.info("Indexed %s video files from %s", len(video_map), video_dir)
    return video_map


def collect_video_requests(frame_rows: Sequence[Sequence[int]]) -> List[Dict[str, object]]:
    """Load the Video-MME dataset metadata and align it with frame indices."""
    LOGGER.info("Loading Video-MME test split metadata")
    dataset = load_dataset("lmms-lab/Video-MME", split="test")
    if len(dataset) != len(frame_rows):
        raise ValueError(
            "Mismatch between dataset entries and CSV rows: "
            f"{len(dataset)} dataset items vs {len(frame_rows)} CSV rows"
        )

    video_requests: Dict[str, Dict[str, object]] = {}
    for example, frames in zip(dataset, frame_rows):
        video_id = example["video_id"]
        video_key = example["videoID"].lower()
        entry = video_requests.setdefault(
            video_id,
            {"video_key": video_key, "frames": set()},
        )
        entry["frames"].update(int(idx) for idx in frames)

    consolidated: List[Dict[str, object]] = []
    for video_id, payload in video_requests.items():
        consolidated.append(
            {
                "video_id": video_id,
                "video_key": payload["video_key"],
                "frames": sorted(payload["frames"]),
            }
        )
    LOGGER.info("Prepared frame extraction requests for %s unique videos", len(consolidated))
    return consolidated


def extract_frames_from_video(
    video_path: Path,
    frame_indices: Sequence[int],
    destination: Path,
    *,
    skip_existing: bool,
) -> Dict[str, object]:
    """Extract specific frames from a video file."""
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video at {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    written = 0
    skipped = 0
    missing: List[int] = []

    for index in frame_indices:
        if index < 0 or index >= total_frames:
            missing.append(index)
            continue
        output_path = destination / f"frame_{index:06d}.jpg"
        if skip_existing and output_path.exists():
            skipped += 1
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = capture.read()
        if not success or frame is None:
            missing.append(index)
            continue
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to write frame {index} to {output_path}")
        written += 1

    capture.release()
    return {
        "written": written,
        "skipped": skipped,
        "missing": missing,
        "total_requested": len(frame_indices),
    }


def main() -> None:
    args = parse_args()
    if args.max_workers:
        raise NotImplementedError("Parallel extraction is not implemented yet")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    frame_rows = load_frame_indices(args.csv)
    video_requests = collect_video_requests(frame_rows)
    video_map = build_video_id_map(args.video_dir)

    output_root = args.output_dir / args.csv.stem
    total_written = 0
    total_skipped = 0
    total_missing = 0

    for request in tqdm(video_requests, desc="Extracting videos", unit="video"):
        video_id = request["video_id"]
        video_key = request["video_key"]
        frame_indices = request["frames"]
        video_path = video_map.get(video_key)
        if not video_path:
            LOGGER.error("Video file for key '%s' not found. Skipping %s", video_key, video_id)
            total_missing += len(frame_indices)
            continue

        destination = output_root / video_id
        result = extract_frames_from_video(
            video_path,
            frame_indices,
            destination,
            skip_existing=args.skip_existing,
        )
        total_written += result["written"]
        total_skipped += result["skipped"]
        total_missing += len(result["missing"])
        if result["missing"]:
            LOGGER.warning(
                "Video %s (%s): missing %s frames out of %s",
                video_id,
                video_path.name,
                len(result["missing"]),
                result["total_requested"],
            )

    LOGGER.info(
        "Finished extraction | written=%s skipped=%s missing=%s",
        total_written,
        total_skipped,
        total_missing,
    )


if __name__ == "__main__":
    main()
