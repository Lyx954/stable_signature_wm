"""Pydantic schemas for AI watermark detection. Copy to backend/app/schemas/"""

from pydantic import BaseModel, Field
from typing import Optional, List


class WatermarkDetectionRequest(BaseModel):
    image_path: Optional[str] = Field(None, description="Path to image/video file on disk")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image data")
    key_str: Optional[str] = Field(
        "111010110101000001010111010011010100010000100111",
        description="48-bit watermark key"
    )
    file_type: Optional[str] = Field("image", description="'image' or 'video'")
    num_frames: Optional[int] = Field(5, description="Frames to sample for video", ge=1, le=20)


class AttackResult(BaseModel):
    attack_name: str
    bit_accuracy: float


class WatermarkDetectionResponse(BaseModel):
    filename: str
    file_type: str
    has_watermark: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    bit_accuracy: float = Field(..., ge=0.0, le=1.0)
    word_accuracy: float = Field(..., ge=0.0, le=1.0)
    extracted_bits: str
    expected_key: str
    dimensions: Optional[str] = None
    size_kb: Optional[float] = None
    attack_results: List[AttackResult] = Field(default_factory=list)
    video_info: Optional[dict] = None
    error: Optional[str] = None


class WatermarkBatchRequest(BaseModel):
    file_paths: List[str] = Field(..., min_length=1)
    key_str: Optional[str] = "111010110101000001010111010011010100010000100111"


class WatermarkBatchResponse(BaseModel):
    total: int
    watermarked_count: int
    results: List[WatermarkDetectionResponse]
