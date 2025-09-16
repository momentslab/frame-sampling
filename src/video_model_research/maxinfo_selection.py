import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import clip

from concurrent.futures import ThreadPoolExecutor
from .maxvolpy.maxvolpy.maxvol import rect_maxvol
import random
import numpy as np

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



def get_clip_embeddings(frames, device, model, preprocess, batch_size=16):
    """
    GPU-optimized CLIP embeddings with CPU output for compatibility.

    Args:
        frames ('list[PIL.Image.Image]'): Frames from the video.
        device ('str'): The device on which to run the model ('cuda' or 'cpu').
        model ('torch.nn.Module'): The CLIP image encoder model.
        preprocess ('Callable'): A preprocessing function for CLIP.
        batch_size (int): Number of frames to process at once.

    Returns:
        numpy.ndarray: An array of embeddings of image frames.
    """
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(frames), batch_size), desc="Extracting CLIP embeddings"):
            batch_frames = frames[i:i+batch_size]

            batch_inputs = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
            if isinstance(model, torch.nn.DataParallel):
                batch_embeddings = model.module.encode_image(batch_inputs)
            else:
                batch_embeddings = model.encode_image(batch_inputs)

            all_embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def truncated_svd(Q, R = 8, normalize_output=None):
    """
    CPU-based Truncated SVD function using sklearn.

    Args:
        Q ('numpy.ndarray'):
            Matrix of embeddings.
        R ('int'):
            The rank of the Truncated SVD matrix.
        normalize_output ('str' or None):
            The way of normalizing SVD output.

    Returns:
        numpy.ndarray:
            Transformed matrix of embeddings.
    """
    svd = TruncatedSVD(n_components=R, random_state=42)
    Qs = svd.fit_transform(Q)

    if normalize_output == 'standard':
        Qs = StandardScaler().fit_transform(Qs)
    elif normalize_output == 'minmax':
        Qs = MinMaxScaler().fit_transform(Qs)
    elif normalize_output == 'l2':
        Qs = normalize(Qs, norm='l2')


    return Qs


def rectangular_maxvol(Qs, max_n, tol = 0.3):
    """
    CPU-based MaxVol selection function using NumPy.

    Args:
        Qs ('numpy.ndarray'):
            Input matrix (typically embeddings).
        max_n ('int'):
            Maximum number of frames to sample.
        tol ('float'):
            Tolerance threshold for improvement in volume.

    Returns:
        list[int]:
            Indices of the selected rows from the input matrix.
    """
    selected_indices, _ = rect_maxvol(Qs, maxK=max_n, tol = tol)

    return selected_indices.tolist()


def process_video(frames, device, model, preprocess, R, tol, normalize_svd_output, max_n):
    """
    Video processing function which:
      - Performs a MaxInfo algorithm.
      - Embeds all frames using CLIP.
      - Applies the Truncated SVD algorithm.
      - Applies the MaxVolume algorithm.

    Args:
        frames ('list[PIL.Image.Image]'):
            All frames from the video.
        device ('str'):
            The device on which to run the model ('cuda' or 'cpu').
        model ('torch.nn.Module'):
            The CLIP image encoder model.
        preprocess ('Callable[[PIL.Image.Image], torch.Tensor]'):
            A preprocessing function that transforms a PIL image into a tensor suitable for CLIP.
        R ('int'):
            Rank used for Truncated SVD.
        tol ('float'):
            Tolerance threshold for the MaxVolume algorithm.
        normalize_svd_output ('str' or None):
            The way of normalizing SVD output.
        max_n ('int'):
            Maximum number of frames to sample.

    Returns:
        numpy.ndarray:
            An array of indices of the selected frames from the shot.
    """
    Q = get_clip_embeddings(frames, device, model, preprocess)
    Qs = truncated_svd(Q, R, normalize_svd_output)
    Qs = Qs.astype(np.float64)
    indices = rectangular_maxvol(Qs, max_n, tol)
    return indices


def maxinfo_frame_selection(frames, R=8, tol=0.3, max_n=64, normalize_svd_output=None):
    """
    MaxInfo function which: 
      - Performs the MaxInfo algorithm on the whole video.
      - Defines an embedding CLIP ViT-B/32 model.

    Args:
        frames (`'list[PIL.Image.Image]'`):
            Initial frames presampled from a video, could be all frames.
        R ('int'):
            Rank used for Truncated SVD.
        tol ('float'):
            Tolerance threshold for the MaxVolume algorithm.
        max_n ('int'):
            Maximum number of frames to sample.
        normalize_svd_output ('str' or None):
            The way of normalizing SVD output.

    Returns:
        numpy.ndarray:
            An array of unique frame indices to sample.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    all_indices = process_video(frames, device, model, preprocess, R, tol, normalize_svd_output, max_n)
    del model      
    torch.cuda.empty_cache()
    all_indices = np.array(sorted(all_indices))
    return all_indices



