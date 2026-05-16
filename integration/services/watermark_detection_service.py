"""
Watermark detection service wrapping stable_signature_wm.
Copy to: backend/app/services/watermark_detection_service.py

This file auto-locates stable_signature_wm by going up from integration/.
If you move this file elsewhere, update _WM_DIR accordingly.
"""

import sys
import os
import base64
import tempfile
from pathlib import Path
from typing import Optional

# ---- Locate stable_signature_wm ----
# When this file is at: stable_signature_wm/integration/services/
# The module root is: stable_signature_wm/
_INTEGRATION_DIR = Path(__file__).resolve().parent.parent  # integration/
_WM_DIR = _INTEGRATION_DIR.parent  # stable_signature_wm/

if str(_WM_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_WM_DIR.parent))

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from stable_signature_wm import WatermarkDetector
        _detector = WatermarkDetector()
        _detector._ensure_models()
    return _detector


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_watermark(
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    key_str: str = "111010110101000001010111010011010100010000100111",
    file_type: str = "image",
    num_frames: int = 5,
) -> dict:
    """Detect watermark in an image or video (detection-only)."""
    detector = _get_detector()

    if image_base64:
        img_data = base64.b64decode(image_base64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_data)
            tmp_path = tmp.name
        source_path = tmp_path
        fname = "base64_upload"
        cleanup = True
    elif image_path:
        source_path = image_path
        fname = os.path.basename(image_path)
        cleanup = False
    else:
        return {"error": "No image_path or image_base64 provided"}

    try:
        if file_type == "video":
            result = detector.detect_video(source_path, num_frames=num_frames)
            resp = result.to_dict()
        else:
            result = detector.detect(source_path)
            resp = result.to_dict()

        return {
            "filename": fname,
            "file_type": file_type,
            "has_watermark": resp.get("has_watermark", False),
            "confidence": resp.get("confidence", 0.0),
            "bit_accuracy": resp.get("bit_accuracy", 0.0),
            "word_accuracy": resp.get("word_accuracy", 0.0),
            "extracted_bits": resp.get("bits", ""),
            "expected_key": key_str,
            "dimensions": resp.get("dimensions"),
            "size_kb": resp.get("size_kb"),
            "error": None,
        }
    except Exception as e:
        return {"error": str(e), "filename": fname}
    finally:
        if cleanup and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def detect_watermark_batch(
    file_paths: list,
    key_str: str = "111010110101000001010111010011010100010000100111",
) -> dict:
    """Batch watermark detection on multiple files."""
    results = []
    watermarked_count = 0
    for fp in file_paths:
        ext = Path(fp).suffix.lower()
        ftype = "video" if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm") else "image"
        r = detect_watermark(image_path=fp, key_str=key_str, file_type=ftype)
        results.append(r)
        if r.get("has_watermark"):
            watermarked_count += 1
    return {"total": len(results), "watermarked_count": watermarked_count, "results": results}
