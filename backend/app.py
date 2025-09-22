"""Flask backend for the ESP32-CAM retail advertising MVP."""
from __future__ import annotations

import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Tuple
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, url_for

from .advertising import build_ad_context
from .ai import GeminiService, GeminiUnavailableError
from .database import Database
from .recognizer import FaceRecognizer
from .vision import VisionService

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "mvp.sqlite3"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["JSON_AS_ASCII"] = False

gemini = GeminiService()
vision = VisionService()
recognizer = FaceRecognizer(vision=vision)
database = Database(DB_PATH)
database.ensure_demo_data()

if not vision.enabled:
    logging.warning("Google Vision 未設定，將退回雜湊比對模式")


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
        "encoding_source": encoding.source,
    }
    if distance is not None:
        payload["distance"] = distance
    if encoding.metadata and "vision" in encoding.metadata:
        payload["vision"] = encoding.metadata["vision"]

    saved_path = _persist_upload(image_bytes, mime_type, member_id)
    if saved_path:
        payload["stored_image"] = saved_path.name
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


def _persist_upload(image_bytes: bytes, mime_type: str, member_id: str) -> Path | None:
    if not image_bytes:
        return None
    extension = mimetypes.guess_extension(mime_type or "image/jpeg") or ".jpg"
    if not extension.startswith("."):
        extension = f".{extension}"
    if extension == ".jpe":
        extension = ".jpg"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{member_id}_{uuid4().hex[:8]}{extension}"
    path = UPLOAD_DIR / filename
    try:
        with open(path, "wb") as handle:
            handle.write(image_bytes)
    except OSError as exc:
        logging.warning("Failed to persist upload: %s", exc)
        return None
    return path


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
    app.run(host="0.0.0.0", port=5000, debug=True)

