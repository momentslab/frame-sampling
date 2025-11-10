#!/usr/bin/env python3
"""Download Video-MME or MVBench dataset videos from Hugging Face."""

import os
import sys
import argparse
from huggingface_hub import snapshot_download

# Enable faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def download_videomme():
    """Download Video-MME dataset videos."""
    print("\n" + "="*60)
    print("📥 DOWNLOADING VIDEO-MME DATASET")
    print("="*60)
    print("📊 Size: ~101 GB of video files")
    print("📁 Destination: ./data/VideoMME")
    print()

    # Download all video chunks and subtitles
    snapshot_download(
        repo_id="lmms-lab/Video-MME",
        repo_type="dataset",
        allow_patterns=["videos_chunked_*.zip", "subtitle.zip"],
        local_dir="./data/VideoMME",
        local_dir_use_symlinks=False,
    )

    print()
    print("✅ Download complete!")
    print("📁 Files saved to: ./data/VideoMME")
    print()
    print("Next steps:")
    print("  1. Unzip the video chunks:")
    print("     cd data/VideoMME")
    print("     for z in videos_chunked_*.zip; do unzip -n \"$z\"; done")
    print("     unzip -n subtitle.zip -d subtitles")


def download_mvbench():
    """Download MVBench dataset videos."""
    print("\n" + "="*60)
    print("📥 DOWNLOADING MVBENCH DATASET")
    print("="*60)
    print("📊 MVBench contains videos from multiple datasets:")
    print("   - Charades, CLEVRER, NTU RGB+D, Perception Test")
    print("   - VLN-QA, TVQA, Moments in Time, Scene QA")
    print("   - Something-Something V2, FunQA")
    print("📁 Destination: ./data/MVBench")
    print()

    # Download MVBench videos from the 'video' revision
    snapshot_download(
        repo_id="OpenGVLab/MVBench",
        repo_type="dataset",
        revision="video",
        local_dir="./data/MVBench",
        local_dir_use_symlinks=False,
    )

    print()
    print("✅ Download complete!")
    print("📁 Files saved to: ./data/MVBench")


def main():
    parser = argparse.ArgumentParser(
        description="Download Video-MME or MVBench dataset videos from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Video-MME dataset
  python download_data.py VideoMME

  # Download MVBench dataset
  python download_data.py MVBench
        """
    )

    parser.add_argument(
        "dataset",
        choices=["VideoMME", "MVBench"],
        help="Dataset to download: VideoMME or MVBench"
    )

    args = parser.parse_args()

    if args.dataset == "VideoMME":
        download_videomme()
    elif args.dataset == "MVBench":
        download_mvbench()
    else:
        print(f"❌ Unknown dataset: {args.dataset}")
        sys.exit(1)


if __name__ == "__main__":
    main()

