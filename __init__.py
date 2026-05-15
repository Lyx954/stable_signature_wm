"""
Stable Signature Watermark Detection Module
============================================
Self-contained watermark detection & extraction for A-ST platform.

Based on facebookresearch/stable_signature.
Supports images and videos (via frame sampling).

Usage:
    from stable_signature_wm import WatermarkDetector

    detector = WatermarkDetector()
    result = detector.detect("photo.jpg")
    print(result.has_watermark, result.bits, result.confidence)

    video_result = detector.detect_video("video.mp4", num_frames=5)
"""

from .detector import WatermarkDetector, WatermarkResult, VideoWatermarkResult
from .api import (
    detect_image, extract_watermark, verify_watermark,
    detect_video, batch_detect_images, batch_detect_videos,
    generate_watermarked_image, get_module_info,
)

__version__ = "1.0.0"
__all__ = [
    "WatermarkDetector", "WatermarkResult", "VideoWatermarkResult",
    "detect_image", "extract_watermark", "verify_watermark",
    "detect_video", "batch_detect_images", "batch_detect_videos",
    "generate_watermarked_image", "get_module_info",
]
