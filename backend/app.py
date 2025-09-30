"""Flask backend for the ESP32-CAM retail advertising MVP."""
from __future__ import annotations

import logging
from datetime import datetime

import json
import mimetypes
from io import BytesIO

from pathlib import Path
from time import perf_counter
from typing import Iterable, Tuple
from uuid import uuid4

from queue import Queue
from threading import Lock

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
    url_for,
)

from PIL import Image, ImageOps, UnidentifiedImageError

from .advertising import AdContext, build_ad_context
from .ai import GeminiService, GeminiUnavailableError
from .aws import RekognitionService
from .database import Database
from .recognizer import FaceRecognizer

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "mvp.sqlite3"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["JSON_AS_ASCII"] = False

gemini = GeminiService()
rekognition = RekognitionService()
if rekognition.can_describe_faces:
    if rekognition.reset_collection():
        logging.info("Amazon Rekognition collection reset for a clean start")
    else:
        logging.warning(
            "Amazon Rekognition collection reset failed; continuing with existing entries"
        )
recognizer = FaceRecognizer(rekognition)
database = Database(DB_PATH)
database.ensure_demo_data()


class _LatestAdHub:
    """Broadcast the most recent ad context to connected clients."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._subscribers: set[Queue] = set()
        self._context: dict[str, object] | None = None

    def snapshot(self) -> dict[str, object] | None:
        with self._lock:
            if self._context is None:
                return None
            return json.loads(json.dumps(self._context))

    def subscribe(self) -> Queue:
        queue: Queue = Queue()
        with self._lock:
            self._subscribers.add(queue)
            context = self._context
        if context is not None:
            queue.put(context)
        return queue

    def unsubscribe(self, queue: Queue) -> None:
        with self._lock:
            self._subscribers.discard(queue)

    def publish(self, context: dict[str, object]) -> None:
        with self._lock:
            self._context = context
            subscribers = list(self._subscribers)
        for queue in subscribers:
            queue.put(context)


_latest_ad_hub = _LatestAdHub()


def _serialize_ad_context(context: AdContext) -> dict[str, object]:
    return {
        "member_id": context.member_id,
        "member_code": context.member_code,
        "headline": context.headline,
        "subheading": context.subheading,
        "highlight": context.highlight,
        "purchases": [
            {
                "item": purchase.item,
                "purchased_at": purchase.purchased_at,
                "unit_price": purchase.unit_price,
                "quantity": purchase.quantity,
                "total_price": purchase.total_price,
            }
            for purchase in context.purchases
        ],
    }

_existing_event = database.get_latest_upload_event()
if _existing_event is not None:
    _existing_purchases = database.get_purchase_history(_existing_event.member_id)
    _existing_context = build_ad_context(
        _existing_event.member_id, _existing_purchases
    )
    _latest_ad_hub.publish(_serialize_ad_context(_existing_context))

    del _existing_context, _existing_event, _existing_purchases


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/upload_face")
def upload_face():
    """Receive an image from the ESP32-CAM and return the member identifier."""

    overall_start = perf_counter()

    try:
        image_bytes, mime_type = _extract_image_payload(request)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    upload_duration = perf_counter() - overall_start

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

    if new_member:
        indexed_encoding = recognizer.register_face(image_bytes, member_id)
        if indexed_encoding is not None:
            database.update_member_encoding(member_id, indexed_encoding)

    recognition_duration = perf_counter() - recognition_start

    ad_generation_start = perf_counter()
    purchases = database.get_purchase_history(member_id)
    creative = None
    if gemini.can_generate_ads:
        try:
            creative = gemini.generate_ad_copy(
                member_id,
                [
                    {
                        "member_code": purchase.member_code,
                        "item": purchase.item,
                        "purchased_at": purchase.purchased_at,
                        "unit_price": purchase.unit_price,
                        "quantity": purchase.quantity,
                        "total_price": purchase.total_price,
                    }
                    for purchase in purchases
                ],
            )
        except GeminiUnavailableError as exc:
            logging.warning("Gemini ad generation unavailable: %s", exc)
    context = build_ad_context(member_id, purchases, creative=creative)
    _latest_ad_hub.publish(_serialize_ad_context(context))
    ad_generation_duration = perf_counter() - ad_generation_start

    total_duration = perf_counter() - overall_start

    image_filename = _persist_upload_image(member_id, image_bytes, mime_type)
    database.record_upload_event(
        member_id=member_id,
        image_filename=image_filename,
        upload_duration=upload_duration,
        recognition_duration=recognition_duration,
        ad_duration=ad_generation_duration,
        total_duration=total_duration,
    )
    stale_images = database.cleanup_upload_events(keep_latest=1)
    _purge_upload_images(stale_images)

    payload = {
        "status": "ok",
        "member_id": member_id,
        "member_code": database.get_member_code(member_id),
        "new_member": new_member,
        "ad_url": url_for("render_ad", member_id=member_id, _external=True),
    }
    if distance is not None:
        payload["distance"] = distance
    return jsonify(payload), 201 if new_member else 200


@app.post("/members/merge")
def merge_members():
    """Merge two member identifiers when the same face was duplicated."""

    payload = request.get_json(silent=True) or {}
    source_id = str(payload.get("source") or "").strip()
    target_id = str(payload.get("target") or "").strip()
    prefer_source = bool(payload.get("prefer_source_encoding", False))

    if not source_id or not target_id:
        return (
            jsonify({"status": "error", "message": "source 與 target 參數必須提供"}),
            400,
        )

    try:
        source_encoding, target_encoding = database.merge_members(source_id, target_id)
    except ValueError as exc:
        message = str(exc)
        status = 400 if "不可相同" in message else 404
        return jsonify({"status": "error", "message": message}), status

    deleted_faces = recognizer.remove_member_faces(source_id)

    encoding_updated = False
    if (
        prefer_source
        or (not target_encoding.signature and source_encoding.signature)
        or (
            source_encoding.signature
            and source_encoding.source.startswith("rekognition")
            and not target_encoding.source.startswith("rekognition")
        )
    ):
        database.update_member_encoding(target_id, source_encoding)
        encoding_updated = True

    return (
        jsonify(
            {
                "status": "ok",
                "merged_member": source_id,
                "into": target_id,
                "deleted_cloud_faces": deleted_faces,
                "encoding_updated": encoding_updated,
            }
        ),
        200,
    )


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
                            "member_code": purchase.member_code,
                            "item": purchase.item,
                            "purchased_at": purchase.purchased_at,
                            "unit_price": purchase.unit_price,
                            "quantity": purchase.quantity,
                            "total_price": purchase.total_price,
                        }
                        for purchase in purchases
                    ],
                )
        except GeminiUnavailableError as exc:
            logging.warning("Gemini ad generation unavailable: %s", exc)
    context = build_ad_context(member_id, purchases, creative=creative)
    return render_template("ad.html", context=context)



@app.get("/ad/latest")
def render_latest_ad():
    context = _latest_ad_hub.snapshot()
    return render_template(
        "ad_latest.html",
        context=context,
        stream_url=url_for("latest_ad_stream"),
    )


@app.get("/ad/latest/stream")
def latest_ad_stream():
    def event_stream():
        queue = _latest_ad_hub.subscribe()
        try:
            while True:
                context = queue.get()
                payload = json.dumps(context, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        finally:
            _latest_ad_hub.unsubscribe(queue)

    response = Response(
        stream_with_context(event_stream()), mimetype="text/event-stream"
    )
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/latest_upload")
def latest_upload_dashboard():
    event = database.get_latest_upload_event()
    if event is None:
        return render_template(
            "latest_upload.html",
            event=None,
            members_url=url_for("member_directory"),
        )

    image_url = None
    if event.image_filename:
        image_url = url_for("serve_upload_image", filename=event.image_filename)

    return render_template(
        "latest_upload.html",
        event=event,
        image_url=image_url,
        ad_url=url_for("render_ad", member_id=event.member_id, _external=True),
        members_url=url_for("member_directory"),
    )


@app.get("/members")
def member_directory():
    profiles = database.list_member_profiles()
    directory: list[dict[str, object]] = []

    for profile in profiles:
        purchases = []
        if profile.member_id:
            purchases = database.get_purchase_history(profile.member_id)

        directory.append({"profile": profile, "purchases": purchases})

    return render_template("members.html", members=directory)


@app.get("/uploads/<path:filename>")
def serve_upload_image(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


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
    now = datetime.now().replace(second=0, microsecond=0)
    database.add_purchase(
        member_id,
        item="歡迎禮盒",
        purchased_at=now.strftime("%Y-%m-%d %H:%M"),
        unit_price=880.0,
        quantity=1,
        total_price=880.0,
    )


def _persist_upload_image(member_id: str, image_bytes: bytes, mime_type: str) -> str | None:
    extension = mimetypes.guess_extension(mime_type or "") or ".jpg"
    if extension == ".jpe":
        extension = ".jpg"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_suffix = uuid4().hex
    filename = f"{timestamp}_{unique_suffix}_{member_id}{extension}"
    path = UPLOAD_DIR / filename
    try:

        with Image.open(BytesIO(image_bytes)) as image:
            normalized = ImageOps.exif_transpose(image)
            save_kwargs: dict[str, object] = {}
            image_format = (image.format or normalized.format or "").upper()
            if image_format == "JPEG" or extension.lower() in {".jpg", ".jpeg"}:
                save_kwargs.update(quality=95, optimize=True)
            normalized.save(path, **save_kwargs)
    except (UnidentifiedImageError, OSError) as exc:
        logging.warning(
            "Failed to normalise uploaded image %s via Pillow: %s", filename, exc
        )
        try:
            path.write_bytes(image_bytes)
        except OSError as write_exc:
            logging.warning("Failed to persist uploaded image %s: %s", path, write_exc)
            return None

    return filename


def _purge_upload_images(filenames: Iterable[str]) -> None:
    for filename in set(filter(None, filenames)):
        path = UPLOAD_DIR / filename
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            logging.warning("Failed to delete old upload image %s: %s", path, exc)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

