"""Flask API server for Stable Signature watermark detection."""
import sys, os, io, base64, time, tempfile, json, uuid
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODULE_DIR.parent))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

from stable_signature_wm import WatermarkDetector

app = Flask(__name__, static_folder=str(_MODULE_DIR), static_url_path="")
CORS(app)

detector: WatermarkDetector = None
HISTORY_FILE = _MODULE_DIR / "output" / "history.json"


def get_detector():
    global detector
    if detector is None:
        print("[Server] Loading models (~20s on first request)...")
        detector = WatermarkDetector()
        detector._ensure_models()
        print("[Server] Ready.")
    return detector


def _load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return []


def _save_history(entries: list):
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def _make_preview(img: Image.Image, max_size=512) -> str:
    img = img.copy()
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _make_thumb(img: Image.Image, max_size=160) -> str:
    img = img.copy()
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _do_detect(img: Image.Image, fname: str, size_bytes: int, file_type: str = "image") -> dict:
    tmp_path = _MODULE_DIR / "output" / f"_upload_{int(time.time()*1e6)}.jpg"
    tmp_path.parent.mkdir(exist_ok=True)
    img.save(str(tmp_path), "JPEG", quality=95)

    d = get_detector()
    result = d.detect(str(tmp_path))
    tmp_path.unlink(missing_ok=True)

    attacks = {k: round(v, 4) for k, v in result.attack_results.items() if v is not None}

    entry = {
        "filename": fname,
        "type": file_type,
        "size_kb": round(size_bytes / 1024, 1),
        "dimensions": f"{img.width} x {img.height}",
        "has_watermark": result.has_watermark,
        "confidence": round(result.confidence, 4),
        "bit_accuracy": round(result.bit_accuracy, 4),
        "word_accuracy": round(result.word_accuracy, 4),
        "bits": result.bits,
        "bits_display": f"{result.bits[:24]} ... {result.bits[24:]}",
        "attack_results": attacks,
        "preview": _make_preview(img),
        "thumb": _make_thumb(img),
    }
    # Save to history
    _save_to_history(entry)
    return entry


def _save_to_history(entry: dict):
    h = _load_history()
    entry["id"] = uuid.uuid4().hex[:10]
    entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    h.insert(0, entry)
    if len(h) > 100:
        h = h[:100]
    _save_history(h)


# ---------------------------------------------------------------------------
# Routes: API + static
# ---------------------------------------------------------------------------

@app.route("/api/detect", methods=["POST"])
def api_detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return jsonify(_do_detect(img, file.filename, len(img_bytes)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect_bulk", methods=["POST"])
def api_detect_bulk():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No image files provided"}), 400
    results = []
    for file in files:
        if not file.filename:
            continue
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            results.append(_do_detect(img, file.filename, len(img_bytes)))
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    return jsonify({"count": len(results), "results": results})


@app.route("/api/detect_video", methods=["POST"])
def api_detect_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    video_tmp = ""
    try:
        suffix = Path(file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.read())
            video_tmp = tmp.name

        import imageio
        reader = imageio.get_reader(video_tmp)
        meta = reader.get_meta_data()
        nframes = reader.count_frames() if hasattr(reader, 'count_frames') else 0
        duration = meta.get('duration', 0)
        frame_idx = nframes // 2 if nframes > 1 else 0
        frame = reader.get_data(frame_idx)
        reader.close()

        img = Image.fromarray(frame).convert("RGB")
        size_kb = Path(video_tmp).stat().st_size / 1024
        os.unlink(video_tmp)

        result = _do_detect(img, file.filename, int(size_kb * 1024), file_type="video")
        result["video_info"] = {
            "total_frames": nframes,
            "duration_s": round(duration, 1),
            "sampled_frame": frame_idx,
        }
        return jsonify(result)
    except Exception as e:
        if video_tmp and os.path.exists(video_tmp):
            os.unlink(video_tmp)
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500


@app.route("/api/history", methods=["GET", "DELETE"])
def api_history():
    if request.method == "DELETE":
        _save_history([])
        return jsonify({"ok": True})
    return jsonify(_load_history())


@app.route("/api/history/<item_id>", methods=["DELETE"])
def api_history_item(item_id):
    h = _load_history()
    h = [e for e in h if e.get("id") != item_id]
    _save_history(h)
    return jsonify({"ok": True})


@app.route("/api/info")
def api_info():
    from stable_signature_wm.api import get_module_info
    return jsonify(get_module_info())


@app.route("/")
def index():
    return send_from_directory(str(_MODULE_DIR), "index.html")


if __name__ == "__main__":
    print(f"[Server] http://127.0.0.1:7860")
    app.run(host="127.0.0.1", port=7860, debug=False)
