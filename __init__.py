from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .video_save_node import VideoSaveWithAlpha

NODE_CLASS_MAPPINGS["VideoSaveWithAlpha"] = VideoSaveWithAlpha
NODE_DISPLAY_NAME_MAPPINGS["VideoSaveWithAlpha"] = "Video Save (Alpha Support)"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
