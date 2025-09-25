"""Flask backend for the ESP32-CAM retail advertising MVP."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

from flask import Flask, jsonify, render_template, request, url_for

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
arcface_tolerance = _env_float("RECOGNIZER_ARCFACE_TOLERANCE", 1.2)
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

    try:
        image_bytes, mime_type = _extract_image_payload(request)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

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

    payload = {
        "status": "ok",
        "member_id": member_id,
        "new_member": new_member,
        "ad_url": url_for("render_ad", member_id=member_id, _external=True),
    }
    if distance is not None:
        payload["distance"] = distance
    logging.info(
        "辨識完成 member_id=%s new_member=%s distance=%s",
        member_id,
        new_member,
        f"{distance:.4f}" if distance is not None else "n/a",
    )
    return jsonify(payload), 201 if new_member else 200


@app.get("/ad/<member_id>")
def render_ad(member_id: str):
    purchases = database.get_purchase_history(member_id)
    creative = None
    if gemini.can_generate_ads:
        try:
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
        except GeminiUnavailableError as exc:
            logging.warning("Gemini ad generation unavailable: %s", exc)
    context = build_ad_context(member_id, purchases, creative=creative)
    return render_template("ad.html", context=context)


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
