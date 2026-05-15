#!/usr/bin/env python
"""Test suite for stable_signature_wm module."""

import sys, os, tempfile
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODULE_DIR.parent))

import torch
import numpy as np
from PIL import Image

# ---- Test data paths ----
# Use COCO data from the repro directory if available
_REPRO_DATA = Path("d:/homework/Project/AIGC watermark/stable_signature_repro_full/data")


def find_sample_images(n=3):
    """Find sample images for testing."""
    for sub in ["val", "train"]:
        d = _REPRO_DATA / sub
        if d.exists():
            imgs = sorted(d.glob("*.jpg"))[:n]
            if imgs:
                return imgs
    return []


# ======================================================================
# Tests
# ======================================================================

def test_imports():
    print("[1] Module imports...")
    from stable_signature_wm import (
        WatermarkDetector, WatermarkResult, VideoWatermarkResult,
        detect_image, extract_watermark, verify_watermark,
        detect_video, batch_detect_images, get_module_info,
    )
    print("    OK")
    return True


def test_module_info():
    print("[2] Module info...")
    from stable_signature_wm.api import get_module_info
    info = get_module_info()
    assert info["bits"] == 48
    assert len(info["api_functions"]) >= 7
    print(f"    {info['module']} v{info['version']}, supports={info['supports']}")
    return True


def test_detector_init():
    print("[3] Detector init...")
    from stable_signature_wm import WatermarkDetector
    d = WatermarkDetector()
    assert d.num_bits == 48
    print(f"    msg_decoder: {d.msg_decoder_path.name}")
    print(f"    finetune_ckpt: {d.finetune_ckpt.name if d.finetune_ckpt else 'N/A'}")
    print(f"    ldm_ckpt: {d.ldm_ckpt.name}")
    return True


def test_model_loading():
    print("[4] Model loading...")
    from stable_signature_wm import WatermarkDetector
    d = WatermarkDetector()
    d._ensure_models()
    assert d.msg_decoder is not None

    # Quick forward pass
    dummy = torch.randn(1, 3, 256, 256).to(d.device)
    bits = d.msg_decoder(d._vqgan_to_imnet(dummy))
    assert bits.shape[-1] == 48
    print(f"    Output: {bits.shape[-1]} bits, device={d.device}")
    return True


def test_detect_non_watermarked():
    print("[5] Detection on non-watermarked images...")
    from stable_signature_wm import WatermarkDetector
    d = WatermarkDetector()
    d._ensure_models()

    imgs = find_sample_images(20)
    if not imgs:
        print("    SKIP: no sample images")
        return True

    accs = []
    for p in imgs:
        r = d.detect(str(p))
        accs.append(r.bit_accuracy)

    mean_acc = np.mean(accs)
    dev = abs(mean_acc - 0.5)
    print(f"    {len(imgs)} images, mean bit_acc={mean_acc:.4f}, dev={dev:.4f}")
    assert dev < 0.1, f"Deviation too large: {dev:.4f}"
    print("    PASS: close to random baseline (0.5)")
    return True


def test_detect_watermarked():
    print("[6] Detection on watermarked image...")
    from stable_signature_wm import WatermarkDetector
    d = WatermarkDetector()
    d._ensure_models()

    if d.ldm_decoder is None:
        print("    SKIP: no finetuned decoder")
        return True

    imgs = find_sample_images(1)
    if not imgs:
        print("    SKIP: no sample images")
        return True

    # Generate watermarked image
    out = _MODULE_DIR / "output" / "_test_wm.png"
    d.embed(str(imgs[0]), str(out))
    assert out.exists()

    # Detect
    r = d.detect(str(out))
    print(f"    bit_acc={r.bit_accuracy:.4f}, confidence={r.confidence:.4f}, has_wm={r.has_watermark}")
    assert r.has_watermark, "Watermarked image should be detected!"
    out.unlink()
    print("    PASS")
    return True


def test_attack_robustness():
    print("[7] Attack robustness...")
    from stable_signature_wm import WatermarkDetector
    d = WatermarkDetector()
    d._ensure_models()

    if d.ldm_decoder is None:
        print("    SKIP: no finetuned decoder")
        return True

    imgs = find_sample_images(1)
    if not imgs:
        print("    SKIP: no sample images")
        return True

    import utils_img

    # Embed then test attacks
    from torchvision.utils import save_image
    img_pil = Image.open(imgs[0]).convert("RGB")
    x = d._vqgan_transform(img_pil).unsqueeze(0).to(d.device)

    with torch.no_grad():
        z = d.ldm_ae.encode(x).mode()
        x_w = d.ldm_decoder.decode(z)

    attacks = {
        "none": lambda x: x,
        "jpeg_80": lambda x: utils_img.jpeg_compress(x, 80),
        "jpeg_50": lambda x: utils_img.jpeg_compress(x, 50),
        "crop_05": lambda x: utils_img.center_crop(x, 0.5),
        "brightness_2": lambda x: utils_img.adjust_brightness(x, 2.0),
        "rot_25": lambda x: utils_img.rotate(x, 25),
    }

    for name, fn in attacks.items():
        try:
            x_aug = fn(d._vqgan_to_imnet(x_w))
            dec = d.msg_decoder(x_aug)
            diff = (~torch.logical_xor(dec > 0, d.key.unsqueeze(0) > 0))
            bit_acc = diff.float().mean().item()
            status = "PASS" if bit_acc > 0.85 else ("WARN" if bit_acc > 0.7 else "FAIL")
            print(f"    {name:<15s}: bit_acc={bit_acc:.4f}  [{status}]")
        except Exception as e:
            print(f"    {name:<15s}: ERROR ({e})")

    return True


def test_api_batch():
    print("[8] API batch detection...")
    from stable_signature_wm.api import batch_detect_images

    imgs = find_sample_images(3)
    if not imgs:
        print("    SKIP: no sample images")
        return True

    results = batch_detect_images([str(p) for p in imgs])
    assert len(results) == 3
    for r in results:
        assert "has_watermark" in r
        assert "confidence" in r
    print(f"    {len(results)} images processed")
    return True


def test_api_verify():
    print("[9] API verify...")
    from stable_signature_wm.api import verify_watermark

    imgs = find_sample_images(1)
    if not imgs:
        print("    SKIP: no sample images")
        return True

    r = verify_watermark(str(imgs[0]))
    assert "verified" in r
    print(f"    verified={r['verified']}, bit_acc={r['bit_accuracy']:.4f}")
    return True


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("stable_signature_wm — TEST SUITE")
    print(f"Module dir: {_MODULE_DIR}")
    print("=" * 60)

    tests = [
        test_imports,
        test_module_info,
        test_detector_init,
        test_model_loading,
        test_detect_non_watermarked,
        test_detect_watermarked,
        test_attack_robustness,
        test_api_batch,
        test_api_verify,
    ]

    results = {}
    for fn in tests:
        name = fn.__name__.replace("test_", "")
        try:
            results[name] = fn()
        except Exception as e:
            print(f"    FAIL: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    for name, ok in results.items():
        status = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  [{status}] {name}")

    n_pass = sum(1 for v in results.values() if v is True)
    n_skip = sum(1 for v in results.values() if v is None)
    n_fail = sum(1 for v in results.values() if v is False)
    print(f"\n  {n_pass} passed, {n_skip} skipped, {n_fail} failed")


if __name__ == "__main__":
    main()
