# SourceTrust 框架对接说明

将 `stable_signature_wm` 水印检测模块接入 SourceTrust 后端。

## 前置条件

- `stable_signature_wm/` 已部署在 `Project/stable_signature_wm/`
- SourceTrust 后端使用 FastAPI，目录结构为 `SourceTrust/backend/app/`
- Python 环境已安装依赖（见 `requirements_extra.txt`）

## 对接步骤

### 1. 复制文件

将 `integration/` 下的文件复制到 SourceTrust 后端对应位置：

```
integration/                          SourceTrust/backend/app/
├── schemas/
│   └── watermark_detection.py  →    app/schemas/watermark_detection.py
├── services/
│   ├── watermark_detection_service.py       → app/services/watermark_detection_service.py
│   ├── ai_watermark_detection_service.py    → app/services/ai_watermark_detection_service.py
│   └── detection_coordinator.py             → app/services/detection_coordinator.py
└── endpoints/
    └── watermark_detection.py  →    app/api/endpoints/watermark_detection.py
```

### 2. 注册路由

在 `backend/main.py` 中添加：

```python
from app.api.endpoints import watermark_detection

# 在 app.include_router 区域添加：
app.include_router(watermark_detection.router, tags=["ai_watermark_detection"])
```

### 3. 安装依赖

```bash
pip install -r integration/requirements_extra.txt
```

主要额外依赖：`torch`, `torchvision`, `omegaconf`, `imageio`, `imageio-ffmpeg`

### 4. 验证

```bash
# 启动后端
cd backend && uvicorn main:app --port 8000

# 测试水印检测
curl -X POST http://127.0.0.1:8000/api/watermark/detect \
  -H "Content-Type: application/json" \
  -d '{"image_path": "path/to/image.jpg"}'

# 查看 API 文档
open http://127.0.0.1:8000/docs
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/watermark/detect` | POST | 单文件水印检测 |
| `/api/watermark/batch` | POST | 批量检测 |
| `/api/watermark/health` | GET | 健康检查 |

## 注意事项

- `watermark_detection_service.py` 通过相对路径自动定位 `stable_signature_wm/` 模块
- 首次请求会触发模型加载（约 20 秒），后续请求即时响应
- 如需水印嵌入功能，需将 SD2.1 完整 checkpoint 放入 `stable_signature_wm/sd/stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt`
- 如需视频检测，需安装 `imageio-ffmpeg`
