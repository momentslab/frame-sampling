# Reference code: https://github.com/li-plus/DSNet/blob/1804176e2e8b57846beb063667448982273fca89/src/make_dataset.py#L4
# Reference code: https://github.com/e-apostolidis/PGL-SUM/blob/81d0d6d0ee0470775ad759087deebbce1ceffec3/model/configs.py#L10
import os
import cv2
import torch

from pathlib import Path
from tqdm import tqdm

from .config import get_config
from .generate_summary import generate_summary, generate_frame_level_summary
from .model import set_model
from .video_helper import VideoPreprocessor
from .config import get_config
from .model import set_model
from .video_helper import VideoPreprocessor
from torchcodec.decoders import VideoDecoder

from torchinfo import summary
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def pick_frames(video_path, selections):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    n_frames = 0

    with tqdm(total = len(selections), ncols=90, desc = "selecting frames", unit='frame', leave = False) as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            if selections[n_frames]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            n_frames += 1

            pbar.update(1)
        
    cap.release()

    return frames

def produce_video(save_path, frames, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
    for frame in tqdm(frames, total = len(frames), ncols=90, desc = "generating videos", leave = False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()

def main(frame_tensors: list[torch.Tensor], fps: float = None, frame_width: int = None, frame_height: int = None, config = None):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.cuda.set_per_process_memory_fraction(0.9, device=0)

    # Load config
    if config is None:
        config = get_config()

    # create output directory
    # out_dir = Path(config.save_path)
    # out_dir.mkdir(parents=True, exist_ok=True)

    # feature extractor
    video_proc = VideoPreprocessor(
        sample_rate=config.sample_rate,
        device=config.device
    )

    # Load CSTA weights
    model = set_model(
        model_name=config.model_name,
        Scale=config.Scale,
        Softmax_axis=config.Softmax_axis,
        Balance=config.Balance,
        Positional_encoding=config.Positional_encoding,
        Positional_encoding_shape=config.Positional_encoding_shape,
        Positional_encoding_way=config.Positional_encoding_way,
        Dropout_on=config.Dropout_on,
        Dropout_ratio=config.Dropout_ratio,
        Classifier_on=config.Classifier_on,
        CLS_on=config.CLS_on,
        CLS_mix=config.CLS_mix,
        key_value_emb=config.key_value_emb,
        Skip_connection=config.Skip_connection,
        Layernorm=config.Layernorm
    )
    model.load_state_dict(torch.load(config.weight_path, map_location='cuda:0'))
    model = model.to(config.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    
    
    # Generate summarized videos
    with torch.no_grad():
        n_frames, features, cps, pick = video_proc.run(frame_tensors, fps, frame_width, frame_height)
        #summary(model, input_size=(1,3,features.shape[0],features.shape[1]))

        inputs = features.to(config.device)
        inputs = inputs.unsqueeze(0).expand(3,-1,-1).unsqueeze(0)
        outputs = model(inputs)
        predictions = outputs.squeeze().clone().detach().cpu().numpy().tolist()
        predictions = [p / max(predictions) for p in predictions]

        # print(cps.shape, len(predictions), n_frames, pick.shape)


        #selections = generate_summary([cps], [predictions], [n_frames], [pick])[0]

        selections = generate_frame_level_summary([predictions], [n_frames])[0]


        indices = [i for i, val in enumerate(selections) if val == 1]

        #indices = [i for i, val in enumerate(predictions) if val >= 0.7]
        #print(indices)
            
        return indices
        # frames = pick_frames(video_path=video_path, selections=selections)
        # produce_video(
        #     save_path=f'{config.save_path}/{video_name}.mp4',
        #     frames=frames,
        #     fps=video_proc.fps,
        #     frame_size=(video_proc.frame_width,video_proc.frame_height)
        # )

# if __name__=='__main__':
#     main()