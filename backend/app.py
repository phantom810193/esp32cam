"""Flask backend for the ESP32-CAM retail advertising MVP."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, render_template, request, url_for

from .advertising import build_ad_context
from .database import Database
from .recognizer import FaceRecognizer

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "mvp.sqlite3"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["JSON_AS_ASCII"] = False

recognizer = FaceRecognizer()
database = Database(DB_PATH)
database.ensure_demo_data()


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/upload_face")
def upload_face():
    """Receive an image from the ESP32-CAM and return the member identifier."""

    try:
        image_bytes = _extract_image_bytes(request)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    try:
        encoding = recognizer.encode(image_bytes)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 422

    member_id, distance = database.find_member_by_encoding(encoding, recognizer)
    new_member = False
    if member_id is None:
        member_id = database.create_member(encoding)
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
    return jsonify(payload), 201 if new_member else 200


@app.get("/ad/<member_id>")
def render_ad(member_id: str):
    purchases = database.get_purchase_history(member_id)
    context = build_ad_context(member_id, purchases)
    return render_template("ad.html", context=context)


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_image_bytes(req) -> bytes:
    if req.files:
        for key in ("image", "file", "photo"):
            if key in req.files:
                data = req.files[key].read()
                if data:
                    return data
    data = req.get_data()
    if not data:
        raise ValueError("No image data found in request")
    return data


def _create_welcome_purchase(member_id: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d")
    database.add_purchase(
        member_id,
        "歡迎禮盒",
        now,
        0.2,
        "新朋友限定：即刻購買咖啡豆 + 牛奶組合享 8 折！",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

