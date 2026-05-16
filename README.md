# Stable Signature Watermark Detection Module *(detection-only)*

基于 Meta **facebookresearch/stable_signature** 的轻量级 AIGC 水印检测与提取模块。仅需 1.2 MB 模型即可对图片和视频进行 48-bit Stable Signature 水印检测。

> 如需水印嵌入功能，请使用 [`master`](../../tree/master) 分支。

## 版本说明

| 分支 | 功能 | 模型大小 | 用途 |
|------|------|----------|------|
| **`detection-only`** | 检测 / 提取 | ~1.2 MB | 轻量部署，接入其他系统 |
| [`master`](../../tree/master) | 检测 + 嵌入 + 攻击测试 | ~6 GB | 完整功能，需要 GPU |

### 本分支移除的内容

以下嵌入端依赖已被移除，不需要也不可用的功能：

- ❌ 水印嵌入 (`detector.embed()`)
- ❌ 攻击鲁棒性测试 (`attack_results`)
- ❌ SD2.1 VAE checkpoint (~5 GB)
- ❌ Finetuned decoder checkpoint (567 MB)
- ❌ LDM / loss / taming 源码
- ❌ `omegaconf`, `einops`, `transformers`, `open_clip_torch` 等重量级依赖

## 项目结构

```
stable_signature_wm/
├── __init__.py           # 模块入口 (v1.1.0-detection)
├── detector.py           # 核心检测引擎 (仅 msg_decoder)
├── api.py                # Python API (6 个函数)
├── server.py             # Flask Web 服务器 + REST API
├── index.html            # Web 前端界面
├── test_module.py        # 测试套件
├── requirements.txt      # Python 依赖 (精简)
├── utils_img.py          # 图像处理
├── models/
│   ├── dec_48b_whit.torchscript.pt   # 48-bit msg_decoder (1.2 MB)
│   └── keys.txt
├── test_samples/         # 测试素材
│   ├── original/         #   原始图片
│   ├── watermarked/      #   水印版图片
│   └── video/            #   测试视频
├── integration/          # SourceTrust 框架对接文件
│   ├── README.md
│   ├── schemas/
│   ├── services/
│   └── endpoints/
└── output/               # 输出目录 (历史记录等)
```

## 环境要求

- Python >= 3.10
- `torch` + `torchvision` (CPU 即可，GPU 更快)
- 无需 CUDA（纯 CPU 可运行，约 0.5 秒/张）

### 安装

```bash
pip install -r requirements.txt
```

仅 5 个依赖：`torch`, `torchvision`, `numpy`, `Pillow`, `flask`

## Web 可视化界面

```bash
python server.py
```

浏览器打开 **http://127.0.0.1:7860**

| 功能 | 说明 |
|------|------|
| 图片上传 | JPG/PNG，多选批量 |
| 视频上传 | MP4/MOV，自动截帧 |
| 中英文切换 | 右上角按钮 |
| 比特可视化 | 48 个彩色方块，绿=匹配 |
| 历史记录 | 自动保存，时间线列表 |

## Web REST API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/detect` | POST | 单张图片检测 |
| `/api/detect_bulk` | POST | 批量图片检测 |
| `/api/detect_video` | POST | 视频检测 (自动截帧) |
| `/api/history` | GET | 历史记录 |
| `/api/history` | DELETE | 清空历史 |
| `/api/history/<id>` | DELETE | 删除单条 |

### 返回值

```json
{
  "filename": "photo.jpg",
  "has_watermark": true,
  "confidence": 0.98,
  "bit_accuracy": 0.99,
  "bits": "111010110101000001010111010011010100010000100111",
  "dimensions": "512 x 341",
  "size_kb": 158.0
}
```

## Python API

```python
import sys
sys.path.insert(0, 'd:/homework/Project')
from stable_signature_wm import WatermarkDetector

detector = WatermarkDetector()

# 图片检测
result = detector.detect("photo.jpg")
print(result.has_watermark)   # True/False
print(result.bits)            # "111010110101..."
print(result.confidence)      # 0.0 ~ 1.0

# 视频检测
video = detector.detect_video("video.mp4", num_frames=5)
print(video.consensus_bits)
```

```python
from stable_signature_wm.api import (
    detect_image,         # 图片水印检测
    verify_watermark,     # 验证是否匹配密钥
    detect_video,         # 视频检测
    batch_detect_images,  # 批量图片
    batch_detect_videos,  # 批量视频
    get_module_info,      # 模块信息
)

r = detect_image("photo.jpg")
v = verify_watermark("photo.jpg")  # → {"verified": True/False, ...}
results = batch_detect_images(["a.jpg", "b.png"])
```

> **注意**: `generate_watermarked_image` 在本分支不可用，需切换到 `master` 分支。

## 接入 SourceTrust 框架

参见 `integration/README.md`。核心步骤：

1. 将 `integration/` 下文件复制到 `SourceTrust/backend/app/` 对应位置
2. 在 `main.py` 注册路由
3. 启动后端，调用 `/api/detect/comprehensive` 即可

## 性能参考

| 操作 | CPU | GPU (RTX 4070) |
|------|-----|----------------|
| 模型加载 | ~2 秒 | ~1 秒 |
| 单张图片 | ~0.5 秒 | ~0.15 秒 |
| 批量 10 张 | ~5 秒 | ~1.5 秒 |

## 已知限制

- 仅支持 Stable Signature 48-bit 水印，不支持其他水印方案
- 旋转 >25° 等几何攻击显著降低检测率
- 视频检测使用单帧采样
- 无水印嵌入和攻击鲁棒性测试功能

## 参考

- [facebookresearch/stable_signature](https://github.com/facebookresearch/stable_signature)
- Pierre Fernandez et al., "The Stable Signature: Rooting Watermarks in Latent Diffusion Models", ICCV 2023
