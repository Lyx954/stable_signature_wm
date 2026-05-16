"""
A-ST Watermark API — clean interface for system integration.

Usage:
    from stable_signature_wm.api import detect_image, detect_video

    result = detect_image("photo.jpg")
    print(result["has_watermark"], result["confidence"])

    video_result = detect_video("clip.mp4", num_frames=5)
"""

from pathlib import Path
from typing import List

from .detector import WatermarkDetector, get_detector


def detect_image(
    image_path: str,
    key_str: str = "111010110101000001010111010011010100010000100111",
    img_size: int = 256,
) -> dict:
    """Detect and extract watermark from a single image."""
    detector = get_detector(key_str=key_str, img_size=img_size)
    return detector.detect(image_path).to_dict()


def extract_watermark(image_path: str, **kwargs) -> dict:
    """Extract watermark bits from an image. Alias for detect_image."""
    return detect_image(image_path, **kwargs)


def verify_watermark(
    image_path: str,
    key_str: str = "111010110101000001010111010011010100010000100111",
    threshold: float = 0.85,
) -> dict:
    """Check if image contains the expected watermark."""
    result = detect_image(image_path, key_str=key_str)
    return {
        "verified": result["bit_accuracy"] >= threshold,
        "bit_accuracy": result["bit_accuracy"],
        "word_accuracy": result["word_accuracy"],
        "confidence": result["confidence"],
        "extracted_bits": result["bits"],
        "expected_key": key_str,
    }


def detect_video(
    video_path: str,
    num_frames: int = 5,
    key_str: str = "111010110101000001010111010011010100010000100111",
    save_frames: bool = False,
    output_dir: str = None,
) -> dict:
    """Detect watermark in a video by sampling frames."""
    detector = get_detector(key_str=key_str)
    return detector.detect_video(
        video_path, num_frames=num_frames,
        save_frames=save_frames, output_dir=output_dir,
    ).to_dict()


def batch_detect_images(
    image_paths: List[str],
    key_str: str = "111010110101000001010111010011010100010000100111",
    img_size: int = 256,
) -> List[dict]:
    """Batch detect watermark in multiple images."""
    detector = get_detector(key_str=key_str, img_size=img_size)
    results = []
    for path in image_paths:
        if not Path(path).exists():
            results.append({"image_path": path, "has_watermark": False,
                           "error": "File not found", "confidence": 0.0})
            continue
        results.append(detector.detect(path).to_dict())
    return results


def batch_detect_videos(
    video_paths: List[str],
    num_frames: int = 5,
    key_str: str = "111010110101000001010111010011010100010000100111",
) -> List[dict]:
    """Batch detect watermark in multiple videos."""
    detector = get_detector(key_str=key_str)
    results = []
    for path in video_paths:
        if not Path(path).exists():
            results.append({"video_path": path, "has_watermark": False,
                           "error": "File not found", "confidence": 0.0})
            continue
        results.append(detector.detect_video(path, num_frames=num_frames).to_dict())
    return results


def get_module_info() -> dict:
    """Return module metadata for system integration."""
    from . import __version__
    detector = get_detector()
    return {
        "module": "Stable Signature Watermark Detection",
        "version": __version__,
        "method": "facebookresearch/stable_signature",
        "bits": 48,
        "default_key": "111010110101000001010111010011010100010000100111",
        "device": str(detector.device),
        "supports": ["image", "video"],
        "api_functions": [
            "detect_image", "extract_watermark", "verify_watermark",
            "detect_video", "batch_detect_images", "batch_detect_videos",
            "get_module_info",
        ],
        "mode": "detection-only",
    }
