import argparse
from video_model_research.bench_utils import get_model, cleanup_model, set_seed, parse_frame_mode

import time
from transformers.utils import logging
logging.set_verbosity_error()

set_seed(42)


def main():
    parser = argparse.ArgumentParser(description="Run SmolVLM on a video with a prompt.")
    parser.add_argument("--video_path", type=str, default="./videos/fragrance_on_the_trail_of_coco_mademoiselle_mp4.mp4", help="Path to the video file.")
    parser.add_argument("--prompt", type=str, default="Give a short description of the video:", help="Prompt to provide to the model.")
    parser.add_argument("--max_tokens", type=int, default=70, help="Maximum number of tokens to generate.")
    parser.add_argument("--number_of_inference", type=int, default=1, help="Number of inference runs.")
    parser.add_argument("--model_name", type=str, default="ovis", help="Name of the vision-language model to use")

    parser.add_argument("--mode", type=str, default="fps:1:4:96",
                       help="Frame mode: 'first', 'center' | 'fps:fps:min:max' | 'maxinfo:input:max' | 'csta:input:max'")

    args = parser.parse_args()

    start_loading = time.time()
    model = get_model(args.model_name)
    end_loading = time.time()

    start_inference = time.time()

    frame_config = parse_frame_mode(args.mode)

    video_items = {
        "video": args.video_path,
        "return_extra": True if args.model_name == "smolvlm" else False,
        **frame_config  # All parameters are embedded in the mode string
    }
    
    for _ in range(args.number_of_inference):
        print(f"Running inference {_ + 1}...")
        result = model.predict(video_items=video_items, prompt=args.prompt, max_tokens=args.max_tokens)
        print("Prediction:", result)

    end_inference = time.time()

    print(f"Model loaded in {end_loading - start_loading:.2f} seconds.")
    print(f"{args.number_of_inference} inferences completed in {end_inference - start_inference:.2f} seconds.")
    print(f"Average inference time: {(end_inference - start_inference) / args.number_of_inference:.2f} seconds.")

    cleanup_model(model)


if __name__ == "__main__":
    main()
