# Reference code: https://github.com/li-plus/DSNet/blob/1804176e2e8b57846beb063667448982273fca89/src/helpers/video_helper.py
import cv2
import numpy as np
import torch
import torch.nn as nn

from os import PathLike
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from torchvision.models import GoogLeNet_Weights
from tqdm import tqdm
from torchcodec.decoders import VideoDecoder
import os
import torchvision.transforms.functional as TF

from .kts.cpd_auto import cpd_auto

class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.transforms = weights.transforms()
        self.model = models.googlenet(weights=weights)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model.to(self.device)
        self.model.eval()

    def run(self, pil_imgs: list[Image.Image]):
        tensors = [self.transforms(img) for img in pil_imgs]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            feats = self.model(batch)
            feats = feats.squeeze(-1).squeeze(-1)
            feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-10)

        return feats

    
class VideoPreprocessor(object):
    def __init__(self, sample_rate: int, device: str):
        self.model = FeatureExtractor(device)
        self.sample_rate = sample_rate

    def get_features(self, video_path: PathLike):
        video_path = Path(video_path)
        TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
        cap = cv2.VideoCapture(str(video_path))
        assert cap is not None, f'Cannot open video: {video_path}'

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        features = []
        n_frames = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total = total_frames, ncols=90, desc = "getting features", unit='frame', leave=False) as pbar:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feat = self.model.run(frame)
                features.append(feat)
                    
                n_frames += 1
                pbar.update(1)

        cap.release()
        features = torch.stack(features)
        return n_frames, features


    def get_features_video_decoder(self, frame_tensors: list[Image.Image], fps=None, frame_width=None, frame_height=None, batch_size=32):
        total_frames = len(frame_tensors)
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        feature_batches = []

        with tqdm(total=total_frames, ncols=90, desc="getting features", unit='frame', leave=False) as pbar:
            for i in range(0, total_frames, batch_size):
                batch_imgs = frame_tensors[i:i + batch_size]

                feats = self.model.run(batch_imgs)

                feature_batches.append(feats)
                pbar.update(len(batch_imgs))

        features = torch.cat(feature_batches, dim=0)
        return total_frames, features



    

    def kts(self, n_frames, features):
        seq_len = len(features)
        picks = np.arange(0, seq_len)
        # compute change points using KTS
        kernel = np.matmul(features.clone().detach().cpu().numpy(), features.clone().detach().cpu().numpy().T)
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1, verbose=False)
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T
        return change_points, picks

    def run(self, frame_tensors, fps, frame_width, frame_height):
        n_frames, features= self.get_features_video_decoder(frame_tensors, fps, frame_width, frame_height)
        cps, picks = self.kts(n_frames, features)
        return n_frames, features[::self.sample_rate,:], cps, picks[::self.sample_rate]
