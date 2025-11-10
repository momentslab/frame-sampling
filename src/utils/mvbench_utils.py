"""
MVBench utility constants for video language model evaluation.

This module contains task definitions and data mappings for evaluating models
on the MVBench (Multi-task Video Benchmark).
"""

# MVBench task definitions
MVBENCH_TASKS = {
    "action_sequence": "Action Sequence",
    "moving_count": "Moving Count",
    "action_prediction": "Action Prediction",
    "episodic_reasoning": "Episodic Reasoning",
    "action_antonym": "Action Antonym",
    "action_count": "Action Count",
    "scene_transition": "Scene Transition",
    "object_shuffle": "Object Shuffle",
    "object_existence": "Object Existence",
    "fine_grained_pose": "Fine-grained Pose",
    "unexpected_action": "Unexpected Action",
    "moving_direction": "Moving Direction",
    "state_change": "State Change",
    "object_interaction": "Object Interaction",
    "character_order": "Character Order",
    "action_localization": "Action Localization",
    "counterfactual_inference": "Counterfactual Inference",
    "fine_grained_action": "Fine-grained Action",
    "moving_attribute": "Moving Attribute",
    "egocentric_navigation": "Egocentric Navigation"
}

# MVBench video path mappings for different tasks
MVBENCH_DATA_MAPPING = {
    "object_interaction": "star/Charades_segment",
    "action_sequence": "star/Charades_segment",
    "action_prediction": "star/Charades_segment",
    "action_localization": "sta/sta_video_segment",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd_convert",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}

