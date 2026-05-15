"""Download SD 2.1 checkpoint from HuggingFace (~5 GB)."""
import os
import sys
from huggingface_hub import hf_hub_download

REPO_ID = "stabilityai/stable-diffusion-2-1-base"
FILENAME = "v2-1_512-ema-pruned.ckpt"
DEST_DIR = os.path.join(os.path.dirname(__file__), "sd", "stable-diffusion-2-1-base")

os.makedirs(DEST_DIR, exist_ok=True)
dest = os.path.join(DEST_DIR, FILENAME)

if os.path.exists(dest):
    print(f"[OK] 已存在: {dest}")
    sys.exit(0)

print(f"[...] 正在下载 {REPO_ID}/{FILENAME} (~5 GB)")
print(f"     目标: {dest}")
hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=DEST_DIR, local_dir_use_symlinks=False)
print("[OK] 下载完成！")
