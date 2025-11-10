#from moviepy.editor import VideoFileClip
#import cv2
#import numpy as np
from PIL import Image
import torchvision.transforms as T
from video_manager.custom_read_video import my_custom_read_video_torchvision


def load_video(video_items):
    video, _ = my_custom_read_video_torchvision(video_items)
    frames = [T.ToPILImage()(frame).convert("RGB") for frame in video]
    return frames

"""
def load_video(video_path,num_frames=12,max_partition=1):
    with VideoFileClip(video_path) as clip:
        total_frames = int(clip.fps * clip.duration)
        if total_frames <= num_frames:
            sampled_indices = range(total_frames)
        else:
            stride = total_frames / num_frames
            sampled_indices = [min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(num_frames)]
        frames = [clip.get_frame(index / clip.fps) for index in sampled_indices]
        frames = [Image.fromarray(frame, mode='RGB') for frame in frames]
    return frames
"""
