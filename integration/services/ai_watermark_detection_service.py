"""
Entry point called by the detection coordinator.
Copy to: backend/app/services/ai_watermark_detection_service.py
"""

from app.services.watermark_detection_service import detect_watermark_batch


def ai_watermark_detection(file_paths: list, files_num: int) -> dict:
    """Called by detection coordinator. Returns aggregated watermark results."""
    result = detect_watermark_batch(file_paths)
    return {
        "module": "ai_watermark_detection",
        "total": result["total"],
        "watermarked_count": result["watermarked_count"],
        "details": result["results"],
    }
