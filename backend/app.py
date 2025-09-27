"""Flask backend for the ESP32-CAM retail advertising MVP."""
from __future__ import annotations

import base64
import copy
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request, redirect, url_for
from PIL import Image

from .advertising import build_ad_context
from .ai import GeminiService, GeminiUnavailableError
from .database import Database
from .face_pipeline import AdvancedFacePipeline
from .recognizer import FaceRecognizer

_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=_LOG_LEVELS.get(_LOG_LEVEL_NAME, logging.INFO))
if _LOG_LEVEL_NAME not in _LOG_LEVELS:
    logging.warning("LOG_LEVEL=%r 非法，改用 INFO 等級", _LOG_LEVEL_NAME)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "mvp.sqlite3"
RECOGNITION_PATH = DATA_DIR / "last_recognition.json"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["JSON_AS_ASCII"] = False

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logging.warning("環境變數 %s=%r 無法轉成浮點數，改用預設值 %.3f", name, raw, default)
        return default


gemini = GeminiService()
pipeline = AdvancedFacePipeline()
if pipeline.is_available:
    logging.info("Advanced face pipeline 啟用，InsightFace 特徵可使用")
elif pipeline.last_error:
    logging.info("Advanced face pipeline 停用：%s", pipeline.last_error)
else:
    logging.info("Advanced face pipeline 未啟用 (未提供錯誤訊息)")

tolerance = _env_float("RECOGNIZER_TOLERANCE", 0.38)
arcface_tolerance = _env_float("RECOGNIZER_ARCFACE_TOLERANCE", 0.40)
logging.info(
    "FaceRecognizer 容差設定：tolerance=%.3f, arcface_tolerance=%.3f",
    tolerance,
    arcface_tolerance,
)
recognizer = FaceRecognizer(
    gemini,
    tolerance=tolerance,
    arcface_tolerance=arcface_tolerance,
    pipeline=pipeline,
)
database = Database(DB_PATH)
database.ensure_demo_data()


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/upload_face")
def upload_face():
    """Receive an image from the ESP32-CAM and return the member identifier."""

    start_total = perf_counter()
    try:
        image_bytes, mime_type = _extract_image_payload(request)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    upload_duration = perf_counter() - start_total

    recognition_start = perf_counter()
    try:
        encoding = recognizer.encode(image_bytes, mime_type=mime_type)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 422

    member_id, distance = database.find_member_by_encoding(encoding, recognizer)
    new_member = False
    if member_id is None:
        member_id = database.create_member(encoding, recognizer.derive_member_id(encoding))
        _create_welcome_purchase(member_id)
        new_member = True
    recognition_duration = perf_counter() - recognition_start
    total_duration = perf_counter() - start_total

    detection_result = recognizer.consume_last_pipeline_result()
    _create_recognition_event(
        member_id=member_id,
        distance=distance,
        new_member=new_member,
        detection=detection_result,
        image_bytes=image_bytes,
        mime_type=mime_type,
        durations={
            "upload": upload_duration,
            "recognition": recognition_duration,
            "overall": total_duration,
        },
    )

    payload = {
        "status": "ok",
        "member_id": member_id,
        "new_member": new_member,
        "ad_url": url_for("render_ad", member_id=member_id, _external=True),
    }
    if distance is not None:
        payload["distance"] = distance
    logging.info(
        "辨識完成 member_id=%s new_member=%s distance=%s upload=%.3fs recognition=%.3fs total=%.3fs",
        member_id,
        new_member,
        f"{distance:.4f}" if distance is not None else "n/a",
        upload_duration,
        recognition_duration,
        total_duration,
    )
    return jsonify(payload), 201 if new_member else 200


@app.get("/ad/<member_id>")
def render_ad(member_id: str):
    purchases = database.get_purchase_history(member_id)
    creative = None
    ad_generation_duration: float | None = None
    if gemini.can_generate_ads:
        try:
            ad_start = perf_counter()
            creative = gemini.generate_ad_copy(
                member_id,
                [
                    {
                        "item": purchase.item,
                        "last_purchase": purchase.last_purchase,
                        "discount": purchase.discount,
                        "recommendation": purchase.recommendation,
                    }
                    for purchase in purchases
                ],
            )
            ad_generation_duration = perf_counter() - ad_start
        except GeminiUnavailableError as exc:
            logging.warning("Gemini ad generation unavailable: %s", exc)
    context = build_ad_context(member_id, purchases, creative=creative)

    recognition = None
    if ad_generation_duration is not None:
        recognition = _update_last_recognition(
            {"durations": {"ad": ad_generation_duration}},
            member_id=member_id,
        )
    if recognition is None:
        recognition = _load_last_recognition()
    if recognition and recognition.get("member_id") != member_id:
        recognition = None
    recognition_display = _prepare_recognition_for_template(recognition)

    return render_template("ad.html", context=context, recognition=recognition_display)


@app.get("/members")
def manage_members():
    members = database.list_members()
    status = request.args.get("status")
    error = request.args.get("error")
    return render_template("members.html", members=members, status=status, error=error)


@app.post("/members/merge")
def merge_members():
    source = (request.form.get("source_member") or "").strip()
    target = (request.form.get("target_member") or "").strip()
    weight_raw = request.form.get("blend_weight")
    try:
        weight = float(weight_raw) if weight_raw not in (None, "") else 0.6
    except ValueError:
        weight = 0.6
    weight = float(min(max(weight, 0.0), 1.0))

    if not source or not target:
        return redirect(url_for("manage_members", error="請選擇來源會員與目標會員"))
    if source == target:
        return redirect(url_for("manage_members", error="來源與目標會員不可相同"))

    try:
        database.merge_members(source, target, blend_weight=weight)
    except ValueError as exc:
        return redirect(url_for("manage_members", error=str(exc)))

    message = f"已將 {source} 合併至 {target} (權重 {weight:.2f})"
    return redirect(url_for("manage_members", status=message))


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_image_payload(req) -> Tuple[bytes, str]:
    if req.files:
        for key in ("image", "file", "photo"):
            if key in req.files:
                uploaded = req.files[key]
                data = uploaded.read()
                if data:
                    return data, uploaded.mimetype or req.mimetype or "image/jpeg"
    data = req.get_data()
    if not data:
        raise ValueError("No image data found in request")
    return data, req.mimetype or "image/jpeg"


def _create_welcome_purchase(member_id: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d")
    database.add_purchase(
        member_id,
        "歡迎禮盒",
        now,
        0.2,
        "AI 精選：咖啡豆 x 手工甜點組，今天下單享 8 折！",
    )


def _create_recognition_event(
    *,
    member_id: str,
    distance: float | None,
    new_member: bool,
    detection: AdvancedFacePipeline.PipelineResult | None,
    image_bytes: bytes,
    mime_type: str,
    durations: dict[str, float],
) -> None:
    try:
        record: dict[str, Any] = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            "member_id": member_id,
            "new_member": bool(new_member),
            "distance": float(distance) if distance is not None else None,
            "mime_type": mime_type,
            "durations": {k: float(v) for k, v in durations.items() if v is not None},
        }
        faces: list[dict[str, Any]] = []
        annotated = None
        if detection is not None:
            faces = _serialise_faces(detection.faces)
            annotated = _encode_image_array(detection.annotated_frame) or _encode_image_array(
                detection.frame
            )
        else:
            faces, annotated = _detect_faces_fallback(image_bytes)
        record["faces"] = faces
        record["annotated_image"] = annotated or _encode_image_bytes(image_bytes)
        _normalise_durations(record)
        _save_recognition_record(record)
    except Exception as exc:  # pragma: no cover - 防禦性紀錄
        logging.warning("Failed to record recognition event: %s", exc)


def _update_last_recognition(update: dict[str, Any], *, member_id: str) -> dict[str, Any] | None:
    current = _load_last_recognition()
    if not current:
        return None
    if current.get("member_id") != member_id:
        return current
    merged = copy.deepcopy(current)
    durations_update = update.get("durations", {})
    if durations_update:
        target = merged.setdefault("durations", {})
        for key, value in durations_update.items():
            if value is None:
                continue
            try:
                target[key] = float(value)
            except (TypeError, ValueError):
                continue
    _normalise_durations(merged)
    _save_recognition_record(merged)
    return merged


def _prepare_recognition_for_template(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    prepared = copy.deepcopy(record)
    prepared["timestamp_display"] = _format_timestamp(record.get("timestamp"))
    prepared["distance_display"] = _format_distance(record.get("distance"))
    prepared["metrics"] = _build_metrics(record.get("durations", {}))
    prepared["faces"] = _prepare_faces(record.get("faces", []))
    return prepared


def _serialise_faces(faces: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, face in enumerate(faces):
        bbox = getattr(face, "bbox", None)
        if bbox is not None:
            bbox = [int(v) for v in bbox]
        quality = getattr(face, "quality", None)
        try:
            quality_value = float(quality) if quality is not None else None
        except (TypeError, ValueError):
            quality_value = None
        preview = _encode_image_array(getattr(face, "image", None))
        results.append(
            {
                "index": idx,
                "bbox": bbox,
                "quality": quality_value,
                "preview": preview,
            }
        )
    return results


def _detect_faces_fallback(image_bytes: bytes) -> tuple[list[dict[str, Any]], str | None]:
    try:
        import cv2

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return [], None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        detections = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
        faces: list[dict[str, Any]] = []
        annotated = frame_rgb.copy()
        for idx, (x, y, w, h) in enumerate(detections):
            bbox = (int(x), int(y), int(x + w), int(y + h))
            crop = _crop_with_margin(frame_rgb, bbox)
            faces.append(
                {
                    "index": idx,
                    "bbox": bbox,
                    "quality": None,
                    "preview": _encode_image_array(crop),
                }
            )
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (239, 68, 68), 4)
        if not faces:
            return [], _encode_image_array(frame_rgb)
        return faces, _encode_image_array(annotated)
    except ImportError:
        return [], None
    except Exception as exc:  # pragma: no cover - 防禦性紀錄
        logging.warning("Fallback face detection failed: %s", exc)
        return [], None


def _crop_with_margin(frame: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.15) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    expand_x = int(width * margin)
    expand_y = int(height * margin)
    x1 = max(0, x1 - expand_x)
    y1 = max(0, y1 - expand_y)
    x2 = min(w, x2 + expand_x)
    y2 = min(h, y2 + expand_y)
    return frame[y1:y2, x1:x2]


def _prepare_faces(faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for idx, face in enumerate(faces):
        prepared.append(
            {
                "index": idx,
                "preview": face.get("preview"),
                "quality_display": _format_quality(face.get("quality")),
            }
        )
    return prepared


def _build_metrics(durations: dict[str, Any]) -> list[dict[str, str]]:
    upload = _format_seconds(durations.get("upload"))
    recognition = _format_seconds(durations.get("recognition"))
    ad = _format_seconds(durations.get("ad"))
    overall_value = _calculate_overall_duration(durations)
    overall = _format_seconds(overall_value)
    metrics: list[dict[str, str]] = [
        {"label": "上傳", "value": upload},
        {"label": "辨識", "value": recognition},
        {"label": "廣告生成", "value": ad},
        {"label": "總計", "value": overall},
    ]
    return metrics


def _normalise_durations(record: dict[str, Any]) -> None:
    durations = record.setdefault("durations", {})
    for key, value in list(durations.items()):
        try:
            durations[key] = float(value)
        except (TypeError, ValueError):
            durations.pop(key, None)
    durations["overall"] = _calculate_overall_duration(durations)


def _calculate_overall_duration(durations: dict[str, Any]) -> float | None:
    components = [durations.get("upload"), durations.get("recognition"), durations.get("ad")]
    values: list[float] = []
    for value in components:
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return durations.get("overall") if isinstance(durations.get("overall"), (int, float)) else None
    return sum(values)


def _format_seconds(value: Any) -> str:
    try:
        if value is None:
            return "—"
        return f"{float(value):.1f} 秒"
    except (TypeError, ValueError):
        return "—"


def _format_quality(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return None


def _format_distance(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return None


def _format_timestamp(value: Any) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _encode_image_array(array: Any) -> str | None:
    if array is None:
        return None
    try:
        if isinstance(array, Image.Image):
            image = array
        else:
            data = np.asarray(array)
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            if data.dtype != np.uint8:
                data = np.clip(data, 0, 255).astype(np.uint8)
            image = Image.fromarray(data)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # pragma: no cover - 防禦性紀錄
        logging.warning("Failed to encode image array: %s", exc)
        return None


def _encode_image_bytes(image_bytes: bytes) -> str | None:
    try:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as exc:  # pragma: no cover - 防禦性紀錄
        logging.warning("Failed to encode image bytes: %s", exc)
        return None


def _save_recognition_record(record: dict[str, Any]) -> None:
    try:
        RECOGNITION_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, ensure_ascii=False)
        temp_path = RECOGNITION_PATH.with_suffix(".tmp")
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(RECOGNITION_PATH)
    except Exception as exc:  # pragma: no cover - 防禦性紀錄
        logging.warning("Failed to persist recognition record: %s", exc)


def _load_last_recognition() -> dict[str, Any] | None:
    if not RECOGNITION_PATH.exists():
        return None
    try:
        return json.loads(RECOGNITION_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - 防禦性紀錄
        logging.warning("Failed to read recognition record: %s", exc)
        return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
