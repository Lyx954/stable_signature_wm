"""
Detection-only watermark extraction engine.

Uses the pretrained msg_decoder (torchscript, ~1.2 MB) to extract
48-bit Stable Signature watermarks from images and videos.
No SD2.1 VAE or finetuned decoder required.
"""

import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Path setup: module root for utils.py / utils_img.py
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class WatermarkResult:
    image_path: str
    has_watermark: bool
    bits: str
    bit_accuracy: float
    word_accuracy: float
    confidence: float
    psnr: Optional[float] = None
    attack_results: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "has_watermark": self.has_watermark,
            "bits": self.bits,
            "bit_accuracy": self.bit_accuracy,
            "word_accuracy": self.word_accuracy,
            "confidence": self.confidence,
            "psnr": self.psnr,
            "attack_results": self.attack_results,
        }


@dataclass
class VideoWatermarkResult:
    video_path: str
    has_watermark: bool
    frames_checked: int
    frames_with_watermark: int
    consensus_bits: str
    consensus_accuracy: float
    confidence: float
    frame_results: List[WatermarkResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "has_watermark": self.has_watermark,
            "frames_checked": self.frames_checked,
            "frames_with_watermark": self.frames_with_watermark,
            "consensus_bits": self.consensus_bits,
            "consensus_accuracy": self.consensus_accuracy,
            "confidence": self.confidence,
            "frame_results": [f.to_dict() for f in self.frame_results],
        }


# ======================================================================
# Detection-only detector (no SD / LDM dependency)
# ======================================================================

class WatermarkDetector:
    """
    Stable Signature watermark detector (detection / extraction only).

    Only requires the 48-bit msg_decoder (~1.2 MB torchscript model).
    Embedding and attack robustness testing are NOT available in this mode.
    """

    def __init__(
        self,
        msg_decoder_path: str = None,
        key_str: str = "111010110101000001010111010011010100010000100111",
        img_size: int = 256,
        device_str: str = None,
    ):
        self.img_size = img_size
        self.key_str = key_str
        self.num_bits = len(key_str)

        _m = _MODULE_DIR
        if msg_decoder_path:
            self.msg_decoder_path = Path(msg_decoder_path)
        else:
            self.msg_decoder_path = _m / "models" / "dec_48b_whit.torchscript.pt"

        self.device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy model loading (msg_decoder only)
    # ------------------------------------------------------------------

    def _ensure_models(self):
        if self._loaded:
            return
        print(f"[WatermarkDetector] Loading msg_decoder on {self.device}...")
        self.msg_decoder = torch.jit.load(str(self.msg_decoder_path)).to(self.device)
        self.msg_decoder.eval()

        self.key = torch.tensor(
            [c == "1" for c in self.key_str], dtype=torch.float32, device=self.device
        )

        # Image transform: normalize to ImageNet stats (what msg_decoder expects)
        self._imnet_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            utils_img.normalize_img,
        ])

        self._loaded = True
        print("[WatermarkDetector] Ready (detection-only mode).")

    # ------------------------------------------------------------------
    # Image detection
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> WatermarkResult:
        """Detect and extract watermark from an image."""
        self._ensure_models()

        img_pil = Image.open(image_path).convert("RGB")
        x = self._imnet_transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            decoded = self.msg_decoder(x)
            bits_binary = (decoded > 0).squeeze(0).tolist()
            bits_str = "".join("1" if b else "0" for b in bits_binary)

            key_binary = self.key.tolist()
            matches = sum(1 for i in range(self.num_bits) if bits_binary[i] == key_binary[i])
            bit_acc = matches / self.num_bits

        confidence = abs(bit_acc - 0.5) * 2.0
        has_watermark = confidence > 0.7

        return WatermarkResult(
            image_path=str(image_path),
            has_watermark=has_watermark,
            bits=bits_str,
            bit_accuracy=bit_acc,
            word_accuracy=1.0 if bit_acc == 1.0 else 0.0,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Video detection (frame sampling)
    # ------------------------------------------------------------------

    def detect_video(
        self,
        video_path: str,
        num_frames: int = 5,
        save_frames: bool = False,
        output_dir: str = None,
    ) -> VideoWatermarkResult:
        """Detect watermark in a video by sampling evenly-spaced frames."""
        self._ensure_models()

        frames = _extract_video_frames(
            video_path, num_frames=num_frames,
            save_dir=output_dir if save_frames else None,
        )

        if not frames:
            return VideoWatermarkResult(
                video_path=str(video_path), has_watermark=False,
                frames_checked=0, frames_with_watermark=0,
                consensus_bits="", consensus_accuracy=0.0, confidence=0.0,
            )

        frame_results = [self.detect(fp) for fp in frames]
        frames_with_wm = sum(1 for r in frame_results if r.has_watermark)

        all_bits = [r.bits for r in frame_results]
        consensus = ""
        for i in range(self.num_bits):
            ones = sum(1 for b in all_bits if b[i] == "1")
            consensus += "1" if ones > len(all_bits) / 2 else "0"

        agreements = []
        for bits in all_bits:
            m = sum(1 for i in range(self.num_bits) if bits[i] == consensus[i])
            agreements.append(m / self.num_bits)
        consensus_accuracy = np.mean(agreements) if agreements else 0.0

        bit_mean = np.mean([r.bit_accuracy for r in frame_results])
        confidence = max(abs(bit_mean - 0.5) * 2.0, frames_with_wm / max(len(frame_results), 1))

        if not save_frames:
            for fp in frames:
                try:
                    os.remove(fp)
                except OSError:
                    pass

        return VideoWatermarkResult(
            video_path=str(video_path),
            has_watermark=confidence > 0.7,
            frames_checked=len(frame_results),
            frames_with_watermark=frames_with_wm,
            consensus_bits=consensus,
            consensus_accuracy=consensus_accuracy,
            confidence=confidence,
            frame_results=frame_results,
        )

    def unload(self):
        if hasattr(self, "msg_decoder"):
            del self.msg_decoder
        self._loaded = False
        torch.cuda.empty_cache()


# ======================================================================
# Video frame extraction
# ======================================================================

def _extract_video_frames(
    video_path: str,
    num_frames: int = 5,
    frame_positions: List[float] = None,
    save_dir: str = None,
) -> List[str]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        duration = 60.0

    if duration <= 0:
        duration = 60.0

    if frame_positions is None:
        if num_frames == 1:
            positions = [0.5]
        else:
            margin = 0.05
            positions = np.linspace(margin, 1.0 - margin, num_frames).tolist()
    else:
        positions = frame_positions

    out_dir = Path(save_dir) if save_dir else Path(tempfile.mkdtemp(prefix="wm_frames_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for i, pos in enumerate(positions):
        timestamp = duration * pos
        out_path = out_dir / f"frame_{i:03d}_{timestamp:.1f}s.png"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
                 "-vframes", "1", "-q:v", "2", str(out_path)],
                capture_output=True, timeout=30, check=True,
            )
            if out_path.exists():
                frame_paths.append(str(out_path))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass

    return frame_paths


# Singleton
_instance: Optional[WatermarkDetector] = None

def get_detector(**kwargs) -> WatermarkDetector:
    global _instance
    if _instance is None:
        _instance = WatermarkDetector(**kwargs)
    return _instance
