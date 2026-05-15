"""
Core watermark detection and extraction engine.

Self-contained — all paths are relative to this module's directory.
Only external dependency: SD2.1 VAE checkpoint (~5GB), configurable via path.
"""

import os
import sys
import json
import io
import copy
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf

# ---- Path setup: make this module's directory the anchor ----
_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR / "src"
# Need BOTH the module dir (for utils.py etc.) and src/ (for ldm, loss)
for _p in [str(_MODULE_DIR), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import local utils (already patched to avoid augly)
import utils
import utils_img
import utils_model

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
# Main detector
# ======================================================================

class WatermarkDetector:
    """Stable Signature watermark detection and extraction.

    Parameters
    ----------
    ldm_ckpt : str or Path, optional
        Path to SD2.1 VAE checkpoint (v2-1_512-ema-pruned.ckpt, ~5GB).
        Default: looks in ./sd/stable-diffusion-2-1-base/
    msg_decoder_path : str or Path, optional
        Path to 48-bit torchscript decoder. Default: ./models/
    finetune_ckpt : str or Path, optional
        Path to finetuned decoder weights. Default: auto-detect in ./models/
    key_str : str, optional
        48-bit watermark key for verification.
    img_size : int
        Processing resolution.
    device_str : str, optional
        Torch device.
    """

    def __init__(
        self,
        ldm_ckpt: str = None,
        msg_decoder_path: str = None,
        finetune_ckpt: str = None,
        key_str: str = "111010110101000001010111010011010100010000100111",
        img_size: int = 256,
        device_str: str = None,
    ):
        self.img_size = img_size
        self.key_str = key_str
        self.num_bits = len(key_str)

        # Resolve paths relative to module directory
        _m = _MODULE_DIR

        # SD2.1 VAE checkpoint (large file, may be symlinked elsewhere)
        if ldm_ckpt:
            self.ldm_ckpt = Path(ldm_ckpt)
        else:
            self.ldm_ckpt = _m / "sd" / "stable-diffusion-2-1-base" / "v2-1_512-ema-pruned.ckpt"

        self.ldm_config = _m / "sd" / "stable-diffusion-2-1-base" / "v2-inference.yaml"

        # Message decoder
        if msg_decoder_path:
            self.msg_decoder_path = Path(msg_decoder_path)
        else:
            self.msg_decoder_path = _m / "models" / "dec_48b_whit.torchscript.pt"

        # Finetuned decoder checkpoint
        if finetune_ckpt:
            self.finetune_ckpt = Path(finetune_ckpt)
        else:
            candidate = _m / "models" / "checkpoint_000.pth"
            self.finetune_ckpt = candidate if candidate.exists() else None

        self.device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_models(self):
        if self._loaded:
            return
        print(f"[WatermarkDetector] Loading models on {self.device}...")

        # 1. Message decoder (always needed, ~1.2MB)
        print("  [1/3] Message decoder...")
        self.msg_decoder = torch.jit.load(str(self.msg_decoder_path)).to(self.device)
        self.msg_decoder.eval()

        # 2. SD2.1 VAE (~5GB, needed for embedding path and attack testing)
        print("  [2/3] SD2.1 VAE...")
        if not self.ldm_ckpt.exists():
            print(f"  WARNING: SD checkpoint not found at {self.ldm_ckpt}")
            print(f"  Detection-only mode (no embedding, no attack robustness tests)")
            self.ldm_ae = None
        else:
            config = OmegaConf.load(str(self.ldm_config))
            ldm_full = utils_model.load_model_from_config(config, str(self.ldm_ckpt))
            self.ldm_ae = ldm_full.first_stage_model
            self.ldm_ae.to(self.device)
            self.ldm_ae.eval()

        # 3. Finetuned decoder (optional, ~567MB)
        print("  [3/3] Finetuned decoder...")
        if self.finetune_ckpt and self.finetune_ckpt.exists() and self.ldm_ae is not None:
            self.ldm_decoder = copy.deepcopy(self.ldm_ae)
            self.ldm_decoder.to(self.device)
            self.ldm_decoder.encoder = nn.Identity()
            self.ldm_decoder.quant_conv = nn.Identity()
            ckpt = torch.load(str(self.finetune_ckpt), map_location="cpu")
            self.ldm_decoder.load_state_dict(ckpt["ldm_decoder"], strict=False)
            self.ldm_decoder.eval()
        else:
            self.ldm_decoder = None

        # Key tensor
        self.key = torch.tensor(
            [c == "1" for c in self.key_str], dtype=torch.float32, device=self.device
        )

        # Transforms
        self._vqgan_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            utils_img.normalize_vqgan,
        ])
        self._vqgan_to_imnet = transforms.Compose([
            utils_img.unnormalize_vqgan,
            utils_img.normalize_img,
        ])

        self._loaded = True
        print("[WatermarkDetector] Ready.")

    # ------------------------------------------------------------------
    # Image detection
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> WatermarkResult:
        """Detect and extract watermark from an image.

        Returns WatermarkResult with:
        - has_watermark: True if confidence > 0.7
        - bits: extracted 48-bit string
        - bit_accuracy: match rate vs known key
        - confidence: detection strength (0-1)
        """
        self._ensure_models()

        img_pil = Image.open(image_path).convert("RGB")
        x = self._vqgan_transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            x_imnet = self._vqgan_to_imnet(x)
            decoded = self.msg_decoder(x_imnet)
            bits_binary = (decoded > 0).squeeze(0).tolist()
            bits_str = "".join("1" if b else "0" for b in bits_binary)

            key_binary = self.key.tolist()
            matches = sum(1 for i in range(self.num_bits) if bits_binary[i] == key_binary[i])
            bit_acc = matches / self.num_bits

        confidence = abs(bit_acc - 0.5) * 2.0
        has_watermark = confidence > 0.7

        # Attack robustness (only if VAE + finetuned decoder available)
        attack_results = {}
        if self.ldm_decoder is not None:
            with torch.no_grad():
                z = self.ldm_ae.encode(x).mode()
                x_w = self.ldm_decoder.decode(z)
                for name, attack_fn in self._default_attacks().items():
                    try:
                        x_aug = attack_fn(self._vqgan_to_imnet(x_w))
                        dec = self.msg_decoder(x_aug)
                        diff = (~torch.logical_xor(dec > 0, self.key.unsqueeze(0) > 0))
                        attack_results[name] = (torch.sum(diff, dim=-1).float() / self.num_bits).item()
                    except Exception:
                        attack_results[name] = None

        return WatermarkResult(
            image_path=str(image_path),
            has_watermark=has_watermark,
            bits=bits_str,
            bit_accuracy=bit_acc,
            word_accuracy=1.0 if bit_acc == 1.0 else 0.0,
            confidence=confidence,
            attack_results=attack_results,
        )

    # ------------------------------------------------------------------
    # Video detection
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

        # Majority-vote consensus bits
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
        confidence = max(abs(bit_mean - 0.5) * 2.0, frames_with_wm / len(frame_results))

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

    # ------------------------------------------------------------------
    # Watermark embedding (requires finetuned decoder)
    # ------------------------------------------------------------------

    def embed(self, image_path: str, output_path: str) -> str:
        """Embed watermark into an image and save to output_path.
        Returns the path to the watermarked image.
        """
        self._ensure_models()
        if self.ldm_decoder is None:
            raise RuntimeError("Finetuned decoder not available. Embedding requires it.")

        img_pil = Image.open(image_path).convert("RGB")
        x = self._vqgan_transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.ldm_ae.encode(x).mode()
            x_w = self.ldm_decoder.decode(z)

        save_image(torch.clamp(utils_img.unnormalize_vqgan(x_w), 0, 1), output_path, nrow=1)
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_attacks() -> dict:
        return {
            "none": lambda x: x,
            "jpeg_80": lambda x: utils_img.jpeg_compress(x, 80),
            "jpeg_50": lambda x: utils_img.jpeg_compress(x, 50),
            "crop_05": lambda x: utils_img.center_crop(x, 0.5),
            "brightness_2": lambda x: utils_img.adjust_brightness(x, 2.0),
            "contrast_2": lambda x: utils_img.adjust_contrast(x, 2.0),
            "rot_25": lambda x: utils_img.rotate(x, 25),
            "comb": lambda x: utils_img.jpeg_compress(
                utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80
            ),
        }

    def unload(self):
        """Release GPU memory."""
        for attr in ("ldm_ae", "ldm_decoder", "msg_decoder"):
            if hasattr(self, attr):
                delattr(self, attr)
        self._loaded = False
        torch.cuda.empty_cache()


# ======================================================================
# Video frame extraction (ffmpeg)
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

    # Get duration via ffprobe
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


# ======================================================================
# Singleton
# ======================================================================

_instance: Optional[WatermarkDetector] = None

def get_detector(**kwargs) -> WatermarkDetector:
    global _instance
    if _instance is None:
        _instance = WatermarkDetector(**kwargs)
    return _instance
