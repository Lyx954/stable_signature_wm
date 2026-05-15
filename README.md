# Stable Signature Watermark Detection Module

基于 Meta **facebookresearch/stable_signature** 的 AIGC 水印检测与提取模块。支持图片和视频的 48-bit Stable Signature 水印检测、提取、验证，附带 Web 可视化界面。

## 项目结构

```
stable_signature_wm/
├── __init__.py           # 模块入口
├── detector.py           # 核心检测引擎 (WatermarkDetector 类)
├── api.py                # Python API 接口 (8 个函数)
├── server.py             # Flask Web 服务器 + REST API
├── index.html            # Web 前端界面
├── test_module.py        # 测试套件 (9 项)
├── requirements.txt      # Python 依赖
├── README.md             # 本文档
├── utils.py              # 工具函数
├── utils_img.py          # 图像处理 (PIL 实现)
├── utils_model.py        # 模型加载工具
├── models/               # 模型权重
│   ├── dec_48b_whit.torchscript.pt   # 48-bit 消息解码器 (1.2 MB)
│   ├── checkpoint_000.pth            # 微调 VAE Decoder (567 MB)
│   └── keys.txt                      # 水印密钥
├── sd/                   # SD2.1 配置 (checkpoint 需单独放入)
│   └── stable-diffusion-2-1-base/
├── src/                  # LDM 源码 (ldm / loss / taming)
├── test_samples/         # 测试素材
│   ├── original/         #   原始图片 (4 张 JPG)
│   ├── watermarked/      #   对应水印版 (4 张 PNG)
│   └── video/            #   测试视频 (2 个 MP4)
└── output/               # 输出目录 (历史记录等)
    └── history.json       #   检测历史 (自动保存)
```

## 一键部署

```bash
# 1. 克隆（需安装 Git LFS）
git lfs install
git clone https://github.com/Lyx954/stable_signature_wm.git
cd stable_signature_wm

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载 SD 模型（可选，仅嵌入水印/攻击测试需要）
python download_model.py

# 4. 启动
python server.py
```

## 环境要求

- Python >= 3.10
- CUDA >= 11.8 (推荐 12.4)
- GPU 显存 >= 8 GB (纯检测模式仅需 ~2 GB)
- [Git LFS](https://git-lfs.com/)（用于自动下载模型权重）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载模型

检测/提取功能所需的轻量模型已通过 Git LFS 管理，clone 后即可使用。

如需**嵌入水印**或**攻击鲁棒性测试**，还需要 Stable Diffusion 2.1 完整 checkpoint（~5 GB），运行以下命令一键下载：

```bash
python download_model.py
```

> 需要 huggingface_hub：`pip install huggingface_hub`（已包含在 requirements.txt 中）

## Web 可视化界面

启动 Web 服务器后在浏览器中拖拽上传即可使用：

```bash
cd d:/homework/Project/stable_signature_wm
python server.py
```

浏览器打开 **http://127.0.0.1:7860**

### 界面功能

| 功能 | 说明 |
|------|------|
| **图片上传** | 支持 JPG/PNG，多选批量上传 |
| **视频上传** | 支持 MP4/MOV，自动抽取中间帧检测 |
| **拖拽上传** | 拖拽文件到上传区域即可 |
| **缩略图队列** | 已选文件缩略图展示，可单独删除 |
| **批量检测** | 点击一次检测全部文件，显示进度 |
| **中英文切换** | 右上角按钮一键切换界面语言 |
| **结果卡片** | 每张图独立卡片：缩略图 + 文件名 + 状态标签 |
| **比特可视化** | 48 个彩色方块，绿色=匹配密钥，红色=不匹配 |
| **攻击鲁棒性** | 8 种攻击的比特准确率进度条 |
| **折叠详情** | 未检测到水印时默认隐藏详情，点击按钮展开 |
| **历史记录** | 自动保存检测结果，时间线列表，点击查看详情，支持删除 |

### 界面布局

```
┌─────────────────────────────────────────┐
│  Stable Signature 水印检测    [English] │
├─────────────────────────────────────────┤
│  [ 检测 ]    [ 历史记录 ⑤ ]            │
├───────────────┬─────────────────────────┤
│  📷 上传图片   │  🎬 上传视频           │
│  JPG,PNG·多选 │  MP4,MOV·自动截帧      │
├───────────────┴─────────────────────────┤
│  [缩略图] [缩略图] [缩略图🎬]           │
│  [开始检测] [清空]      已选 3 个文件    │
├─────────────────────────────────────────┤
│  ┌─ 结果卡片 ───────────────────────┐  │
│  │ [图] filename.jpg    ● 检测到水印 │  │
│  │ 置信度 98%  比特准确率 99%  ...   │  │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  │
│  │ (48-bit 水印比特可视化)           │  │
│  │ JPEG80 ████████████ 95%          │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Web REST API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/detect` | POST | 单张图片检测 |
| `/api/detect_bulk` | POST | 多张图片批量检测 |
| `/api/detect_video` | POST | 视频检测 (自动截帧) |
| `/api/history` | GET | 获取全部历史记录 |
| `/api/history` | DELETE | 清空全部历史 |
| `/api/history/<id>` | DELETE | 删除单条历史 |
| `/api/info` | GET | 模块元信息 |

### 请求示例

```bash
# 单图检测
curl -X POST http://127.0.0.1:7860/api/detect -F "image=@photo.jpg"

# 批量检测
curl -X POST http://127.0.0.1:7860/api/detect_bulk \
  -F "images=@img1.jpg" -F "images=@img2.png"

# 视频检测
curl -X POST http://127.0.0.1:7860/api/detect_video -F "video=@clip.mp4"
```

### 返回值

```json
{
  "filename": "photo.jpg",
  "size_kb": 158.0,
  "dimensions": "512 x 341",
  "has_watermark": true,
  "confidence": 0.98,
  "bit_accuracy": 0.99,
  "word_accuracy": 1.0,
  "bits": "111010110101000001010111...",
  "bits_display": "111010110101000001010111 ... 010100010000100111",
  "preview": "data:image/jpeg;base64,...",
  "thumb": "data:image/jpeg;base64,...",
  "attack_results": {
    "none": 1.0, "jpeg_80": 0.92, "crop_05": 0.99
  },
  "video_info": {                        // 仅视频
    "total_frames": 120,
    "duration_s": 10.0,
    "sampled_frame": 60
  }
}
```

## Python API

### 快速开始

```python
import sys
sys.path.insert(0, 'd:/homework/Project')

from stable_signature_wm import WatermarkDetector

detector = WatermarkDetector()
result = detector.detect("path/to/image.jpg")
print(result.has_watermark)   # True/False
print(result.bits)            # "111010110101..."
print(result.confidence)      # 0.0 ~ 1.0
```

### 所有 API 函数

```python
from stable_signature_wm.api import (
    detect_image,        # 图片水印检测
    extract_watermark,   # 提取水印比特 (同 detect_image)
    verify_watermark,    # 验证水印是否匹配指定密钥
    detect_video,        # 视频水印检测 (多帧采样)
    batch_detect_images, # 批量图片检测
    batch_detect_videos, # 批量视频检测
    generate_watermarked_image,  # 生成水印图 (需 SD 完整模型)
    get_module_info,     # 模块元信息
)

# 单张
r = detect_image("photo.jpg")

# 验证
r = verify_watermark("photo.jpg")  # → {"verified": True, ...}

# 批量
results = batch_detect_images(["a.jpg", "b.png"])

# 视频 (多帧采样 + 多数投票)
r = detect_video("video.mp4", num_frames=5)
```

### 返回值说明

**WatermarkResult (图片)**

| 字段 | 类型 | 说明 |
|------|------|------|
| `has_watermark` | bool | 检测到水印 (confidence > 0.7) |
| `bits` | str | 提取的 48-bit 字符串 |
| `bit_accuracy` | float | 与已知密钥的比特匹配率 |
| `word_accuracy` | float | 48-bit 全匹配则为 1.0 |
| `confidence` | float | 置信度 0~1 (偏离 0.5 随机基线的程度) |
| `attack_results` | dict | 各攻击下比特准确率 (需完整 SD 模型) |

**VideoWatermarkResult (视频)**

| 字段 | 说明 |
|------|------|
| `has_watermark` | 综合判定 |
| `frames_checked` | 采样帧数 |
| `consensus_bits` | 多帧多数投票共识比特串 |
| `consensus_accuracy` | 各帧与共识的平均一致率 |
| `frame_results` | 每帧详细结果列表 |

## 检测原理

1. **水印嵌入**：微调 SD2.1 VAE Decoder，在图像潜空间中嵌入 48-bit 固定密钥
2. **水印提取**：预训练 msg_decoder 直接从图像像素提取水印比特
3. **检测判定**：提取比特与密钥比对，若 bit_accuracy 显著偏离 0.5（随机基线）则判定含水印

### 功能所需模型

| 功能 | 需要的模型 | 大小 | 获取方式 |
|------|-----------|------|----------|
| **检测/提取** | msg_decoder + VAE decoder | ~570 MB | Git LFS（自动） |
| **嵌入水印** | + SD2.1 完整 checkpoint | ~6 GB | `python download_model.py` |
| **攻击鲁棒性** | 同上 | ~6 GB | 同上 |

## 运行测试

```bash
cd d:/homework/Project
python stable_signature_wm/test_module.py
```

## 接入 A-ST 系统

```python
from stable_signature_wm.api import detect_image, detect_video

def handle_upload(file_path, file_type):
    if file_type == "image":
        return detect_image(file_path)
    elif file_type == "video":
        return detect_video(file_path, num_frames=5)
```

## 性能参考 (RTX 4070)

| 操作 | 耗时 |
|------|------|
| 首次模型加载 | ~20 秒 |
| 单张图片检测 | ~0.15 秒 |
| 批量 10 张 | ~1.5 秒 |
| 视频检测 (截帧+检测) | ~2 秒 |

## 已知限制

- 旋转 >25° 等几何攻击会显著降低检测率
- 48-bit word-level 全对齐对复杂攻击较脆弱
- 视频检测使用单帧采样，非逐帧全检
- SD2.1 完整 checkpoint 需 5GB 额外存储

## 参考

- [facebookresearch/stable_signature](https://github.com/facebookresearch/stable_signature)
- Pierre Fernandez et al., "The Stable Signature: Rooting Watermarks in Latent Diffusion Models", ICCV 2023
