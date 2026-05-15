"""
Detection coordinator — orchestrates services and merges results.
Copy to: backend/app/services/detection_coordinator.py
"""

import os
from typing import List, Dict, Optional
from datetime import datetime


class DetectionCoordinator:

    def comprehensive_detect(
        self,
        file_ids: List[str],
        weights: Optional[Dict[str, int]] = None,
    ) -> dict:
        from app.services.ai_watermark_detection_service import ai_watermark_detection

        UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
        file_paths = [
            os.path.join(UPLOAD_DIR, fid)
            for fid in file_ids
            if os.path.exists(os.path.join(UPLOAD_DIR, fid))
        ]

        if weights is None:
            weights = {"source": 25, "entity": 25, "aigc": 25, "modification": 25}

        # Run watermark detection
        wm_result = None
        if file_paths:
            try:
                wm_result = ai_watermark_detection(file_paths, len(file_paths))
            except Exception as e:
                wm_result = {"error": str(e)}

        # Per-file results
        file_results = []
        if wm_result and "details" in wm_result:
            for detail in wm_result["details"]:
                file_results.append({
                    "filename": detail.get("filename", "unknown"),
                    "has_watermark": detail.get("has_watermark", False),
                    "confidence": detail.get("confidence", 0),
                    "bit_accuracy": detail.get("bit_accuracy", 0),
                    "extracted_bits": detail.get("extracted_bits", ""),
                    "dimensions": detail.get("dimensions", ""),
                    "size_kb": detail.get("size_kb", 0),
                })

        return {
            "success": True,
            "file_ids": file_ids,
            "timestamp": datetime.now().isoformat(),
            "weights": weights,
            "summary": {
                "total_files": len(file_ids),
                "watermarked_count": wm_result.get("watermarked_count", 0) if wm_result else 0,
            },
            "files": file_results,
            "modules": {
                "ai_watermark_detection": wm_result or {"status": "no_files"},
                "explicit_detection": {"status": "not_implemented"},
                "metadata_detection": {"status": "not_implemented"},
                "ai_detection": {"status": "not_implemented"},
            },
        }


detection_coordinator = DetectionCoordinator()
