from abc import ABC, abstractmethod


class VideoModel(ABC):
    """Abstract class for video models."""

    @abstractmethod
    def predict(self, video_items: dict, prompt: str, max_tokens: int) -> str:
        """Predict the action in the video."""
        pass


