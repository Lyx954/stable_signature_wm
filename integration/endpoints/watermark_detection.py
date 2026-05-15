"""Watermark detection API endpoint.
Copy to: backend/app/api/endpoints/watermark_detection.py
"""

from fastapi import APIRouter, HTTPException
from app.schemas.watermark_detection import (
    WatermarkDetectionRequest,
    WatermarkDetectionResponse,
    WatermarkBatchRequest,
    WatermarkBatchResponse,
)
from app.services.watermark_detection_service import detect_watermark, detect_watermark_batch

router = APIRouter(prefix="/api/watermark", tags=["ai_watermark_detection"])


@router.post("/detect", response_model=WatermarkDetectionResponse)
def detect_watermark_endpoint(request: WatermarkDetectionRequest):
    if not request.image_path and not request.image_base64:
        raise HTTPException(status_code=400, detail="image_path or image_base64 required")
    result = detect_watermark(
        image_path=request.image_path,
        image_base64=request.image_base64,
        key_str=request.key_str,
        file_type=request.file_type,
        num_frames=request.num_frames,
    )
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/batch", response_model=WatermarkBatchResponse)
def detect_watermark_batch_endpoint(request: WatermarkBatchRequest):
    return detect_watermark_batch(file_paths=request.file_paths, key_str=request.key_str)


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "ai_watermark_detection"}
