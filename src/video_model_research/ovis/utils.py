from PIL import Image
import torchvision.transforms as T
from video_model_research.custom_read_video import my_custom_read_video_torchvision


def load_video(video_items):
    video, _ = my_custom_read_video_torchvision(video_items)
    frames = [T.ToPILImage()(frame).convert("RGB") for frame in video]
    return frames
