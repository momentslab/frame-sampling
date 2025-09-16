import numpy as np
from .knapsack_implementation import knapSack

# Generate summary videos
def generate_summary(all_shot_bound, all_scores, all_nframes, all_positions):
    all_summaries = []
    for video_index in range(len(all_scores)):
        shot_bound = all_shot_bound[video_index]
        frame_init_scores = all_scores[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]

   

        frame_scores = np.zeros(n_frames, dtype=np.float32)
        if positions.dtype != int:
            positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.concatenate([positions, [n_frames]])
        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i + 1]
            if i == len(frame_init_scores):
                frame_scores[pos_left:pos_right] = 0
            else:
                frame_scores[pos_left:pos_right] = frame_init_scores[i]

        shot_imp_scores = []
        shot_lengths = []
        for shot in shot_bound:
            shot_lengths.append(shot[1] - shot[0] + 1)
            shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())

        final_shot = shot_bound[-1]
        final_max_length = int((final_shot[1] + 1) * 0.15)

        selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1

        all_summaries.append(summary)

    return all_summaries


def generate_frame_level_summary(all_frame_scores, all_nframes):
    """
    Generate summaries by selecting important frames using knapsack.
    
    Args:
        all_frame_scores: List of np.arrays, each containing importance scores per frame for each video.
        all_nframes: List of int, number of frames in each video.
    
    Returns:
        all_summaries: List of np.arrays of 0/1 indicating selected frames for each video.
    """
    all_summaries = []
    
    for video_index in range(len(all_frame_scores)):
        frame_scores = all_frame_scores[video_index]
        n_frames = all_nframes[video_index]
        
        
        # Each frame has weight 1
        frame_lengths = [1] * n_frames
        
        # Max summary length: 15% of video frames (adjust as needed)
        capacity = int(n_frames * 0.15)
        
        # Run knapsack to select frames
        selected = knapSack(capacity, frame_lengths, frame_scores, n_frames)
        
        summary = np.zeros(n_frames, dtype=np.int8)
        for frame_idx in selected:
            summary[frame_idx] = 1
        
        all_summaries.append(summary)
        

    return all_summaries