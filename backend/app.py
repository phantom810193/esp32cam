
# backend/app.py
"""Flask backend for the ESP32-CAM retail advertising MVP."""
from __future__ import annotations

import json
import logging
import math
import mimetypes
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from queue import Queue, Empty
from threading import Lock
from time import perf_counter
from typing import Any, Iterable, Mapping, Tuple, Literal
from uuid import uuid4

from flask import (
    Flask,
    Response,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
    url_for,
)

from PIL import Image, ImageOps, UnidentifiedImageError

from werkzeug.utils import safe_join

from .advertising import (
    AD_IMAGE_BY_SCENARIO,
    AdContext,
    TEMPLATE_IMAGE_BY_ID,
    analyse_purchase_intent,
    build_ad_context,
    derive_scenario_key,
)
from .ai import GeminiService, GeminiUnavailableError
from .aws import RekognitionService
from .database import Database, MemberProfile, Purchase, SEED_MEMBER_IDS

from .prediction import predict_next_purchases
from .recognizer import FaceRecognizer
from .routes import adgen_blueprint

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Paths & Config
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# 允許用環境變數覆蓋 DB 位置（預設為 repo 內 data/mvp.sqlite3）
DB_PATH = Path(os.environ.get("DB_PATH", str(DATA_DIR / "mvp.sqlite3")))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(DATA_DIR / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ADS_DIR = Path(os.environ.get("ADS_DIR", str(DATA_DIR / "ads")))
ADS_DIR.mkdir(parents=True, exist_ok=True)

REKOG_RESET = os.environ.get("REKOG_RESET", "").strip().lower() in {"1", "true", "yes"}

PERSONA_LABELS = {
    "dessert-lover": "甜點收藏家",
    "family-groceries": "幼兒園家長",
    "fitness-enthusiast": "健身族",
    "home-manager": "家庭主婦",
    "wellness-gourmet": "健康食品愛好者",
}

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["JSON_AS_ASCII"] = False
app.config["ADS_DIR"] = str(ADS_DIR)  # 儲存為字串路徑
app.register_blueprint(adgen_blueprint)

# -----------------------------------------------------------------------------
# Services (Gemini Text / AWS Rekognition / DB)
# -----------------------------------------------------------------------------

gemini = GeminiService()

# ---- 重要：相容性 shim（避免其他模組仍 import `Gemini` 時失敗）----
try:
    from . import ai as _ai_mod  # 相對匯入 backend.ai
    if not hasattr(_ai_mod, "Gemini") and hasattr(_ai_mod, "GeminiService"):
        setattr(_ai_mod, "Gemini", _ai_mod.GeminiService)
        logger.info("Back-compat shim enabled: backend.ai.Gemini -> GeminiService")
except Exception as _shim_exc:  # 不影響主流程
    logger.warning("Back-compat shim failed to set backend.ai.Gemini: %s", _shim_exc)

rekognition = RekognitionService()


def _maybe_prepare_rekognition() -> None:
    """預設不重置；只有 REKOG_RESET=1 時才清空重建。否則僅確保存在。"""
    if not getattr(rekognition, "can_describe_faces", False):
        logging.info("Rekognition not available; face features disabled")
        return
    try:
        if REKOG_RESET:
            if rekognition.reset_collection():
                logging.warning("Amazon Rekognition collection reset for a clean start (REKOG_RESET=1)")
            else:
                logging.warning("Amazon Rekognition collection reset requested but failed; continuing")
        else:
            ensure_fn = getattr(rekognition, "ensure_collection", None)
            if callable(ensure_fn):
                ensure_fn()
                logging.info("Amazon Rekognition collection ensured (no reset)")
    except Exception as exc:  # 安全防護，避免啟動因雲端初始化失敗而崩潰
        logging.warning("Rekognition prepare step failed: %s", exc)


_maybe_prepare_rekognition()

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
_warmup_once_lock = Lock()
_warmup_ran = False




def _seed_latest_ad_hub() -> None:
    """Lazy warm-up: run on first incoming request, not at import time."""
    try:
        event = database.get_latest_upload_event()
        if not event:
            return

        purchases = database.get_purchase_history(event.member_id)
        profile = database.get_member_profile(event.member_id)

        prediction_items: list[Any] = []
        try:
            pr = predict_next_purchases(purchases, profile=profile)
            if getattr(pr, "items", None):
                prediction_items = list(pr.items)
        except Exception as exc:
            logging.warning(
                "Prediction pipeline unavailable for %s during warmup: %s",
                event.member_id,
                exc,
            )
            prediction_items = []

        timings = {
            "upload": getattr(event, "upload_duration", None),
            "recognition": getattr(event, "recognition_duration", None),
            "generation": getattr(event, "ad_duration", None),
            "total": getattr(event, "total_duration", None),
        }

        ctx = build_ad_context(
            event.member_id,
            purchases,
            profile=profile,
            profile_snapshot=None,
            prediction_items=prediction_items,
            audience=_determine_audience(
                new_member=False, profile=profile, purchases=purchases
            ),
            timings=timings,
            detected_at=getattr(event, "created_at", None),
        )

        _latest_ad_hub.publish(_serialize_ad_context(ctx))

    except Exception as exc:
        logging.warning("Warmup seed failed (lazy): %s", exc)

def _persona_label_display(profile_label: str | None) -> str | None:
    if not profile_label:
        return None
    return PERSONA_LABELS.get(
        profile_label,
        profile_label.replace("-", " ").title(),
    )



def _serialize_ad_context(context: AdContext) -> dict[str, object]:
    try:
        ad_url = url_for("render_ad", member_id=context.member_id, _external=True)
    except RuntimeError:
        ad_url = f"/ad/{context.member_id}"

    try:
        offer_url = url_for("render_ad_offer", member_id=context.member_id, _external=True)
    except RuntimeError:
        offer_url = f"/ad/{context.member_id}/offer"

    audience = context.audience
    cta_href = context.cta_href or ""
    if not cta_href:
        cta_href = offer_url or ad_url
    elif cta_href.startswith("#") and audience == "member":
        cta_href = offer_url or ad_url

    payload: dict[str, object] = {
        "member_id": context.member_id,
        "member_code": context.member_code,
        "headline": context.headline,
        "subheading": context.subheading,
        "highlight": context.highlight,
        "template_id": context.template_id,
        "audience": context.audience,
        "scenario_key": context.scenario_key,
        "cta_text": context.cta_text,
        "cta_href": cta_href,
        "purchases": [
            {
                "item": purchase.item,
                "purchased_at": purchase.purchased_at,
                "product_category": purchase.product_category,
                "internal_item_code": purchase.internal_item_code,
                "unit_price": purchase.unit_price,
                "quantity": purchase.quantity,
                "total_price": purchase.total_price,
            }
            for purchase in context.purchases
        ],
        "predicted_candidates": [dict(candidate) for candidate in context.predicted_candidates],
        "timings": dict(context.timings or {}),
    }
    if context.profile:
        payload["profile"] = dict(context.profile)
    if context.predicted:
        payload["predicted"] = dict(context.predicted)
    if context.detected_at:
        payload["detected_at"] = context.detected_at

    payload["hero_image_url"] = _resolve_template_image(context.template_id)
    payload["status"] = "ok"
    payload["ad_url"] = ad_url
    payload["offer_url"] = offer_url
    latest_event = database.get_latest_upload_event()
    if latest_event is not None:
        payload["event_id"] = latest_event.id
    return payload


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/demo/upload-ad")
def simple_upload_demo() -> str:
    """Serve a minimal uploader that drives the face recognition flow."""
    return render_template("simple_upload.html")


@app.get("/dashboard")
def dashboard() -> str:
    """Render the customer dashboard demo page."""

    requested_member_id = request.args.get("member_id")
    member_id = requested_member_id or None

    try:
        requested_page = int(request.args.get("page", "1"))
    except ValueError:
        requested_page = 1
    requested_page = max(1, requested_page)

    profile = None
    if member_id:
        profile = database.get_member_profile(member_id)

    if profile is None:
        latest_event = database.get_latest_upload_event()
        if latest_event:
            member_id = latest_event.member_id
            profile = database.get_member_profile(member_id)

    if profile is None:
        for seed_member_id in SEED_MEMBER_IDS:
            seeded_profile = database.get_member_profile(seed_member_id)
            if seeded_profile is not None:
                member_id = seed_member_id
                profile = seeded_profile
                break

    purchases: list[Purchase] = []
    page = 1
    page_count = 1
    has_prev = False
    has_next = False
    total_purchases = 0
    predicted_items = []
    prediction_window_label: str | None = None
    if profile and member_id:
        full_history = database.get_purchase_history(member_id)
        prediction_result = predict_next_purchases(full_history, profile=profile)
        predicted_items = prediction_result.items
        prediction_window_label = prediction_result.window_label

        limit = 7
        total_purchases = len(full_history)
        if total_purchases:
            page_count = max(1, math.ceil(total_purchases / limit))
            page = min(max(1, requested_page), page_count)
            offset = (page - 1) * limit
            purchases = full_history[offset : offset + limit]
        else:
            page = 1
            page_count = 1
            purchases = []
        has_prev = page > 1
        has_next = page < page_count
    else:
        page = 1
        page_count = 1
        has_prev = False
        has_next = False

    points_balance_display: str | None = None
    if profile and profile.points_balance is not None:
        points_balance_display = f"{profile.points_balance:,.0f}"

    joined_at_display: str | None = None
    if profile and profile.joined_at:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                joined_dt = datetime.strptime(profile.joined_at, fmt)
            except ValueError:
                continue
            else:
                joined_at_display = joined_dt.strftime("%Y-%m-%d")
                break
        else:
            joined_at_display = profile.joined_at

    persona_label: str | None = None
    display_name: str | None = None
    if profile:
        persona_label = _persona_label_display(profile.profile_label)
        display_name = profile.name or persona_label
    if display_name is None:
        display_name = "尚未命名會員"

    profile_image_url: str | None = None
    if profile and profile.first_image_filename:
        static_upload_path = Path(app.static_folder or "") / "uploads" / profile.first_image_filename
        if static_upload_path.exists():
            profile_image_url = url_for(
                "static", filename=f"uploads/{profile.first_image_filename}"
            )
        else:
            upload_file = UPLOAD_DIR / profile.first_image_filename
            if upload_file.exists():
                profile_image_url = url_for(
                    "serve_upload_image", filename=profile.first_image_filename
                )

    return render_template(
        "dashboard.html",
        profile=profile,
        purchases=purchases,
        persona_label=persona_label,
        display_name=display_name,
        points_balance_display=points_balance_display,
        joined_at_display=joined_at_display,
        profile_image_url=profile_image_url,
        requested_member_id=requested_member_id,
        resolved_member_id=member_id,
        page=page,
        page_count=page_count,
        has_prev=has_prev,
        has_next=has_next,
        total_purchases=total_purchases,
        predicted_items=predicted_items,
        prediction_window_label=prediction_window_label,
    )


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

    resolved = database.resolve_member_id(encoding, recognizer)
    member_id = resolved.member_id
    distance = resolved.distance
    new_member = resolved.new_member

    if new_member:
        _create_welcome_purchase(member_id)
        indexed_encoding = recognizer.register_face(image_bytes, member_id)
        if indexed_encoding is not None:
            database.update_member_encoding(member_id, indexed_encoding)
    else:
        if resolved.auto_merged_source:
            removed = recognizer.remove_member_faces(resolved.auto_merged_source)
            if removed:
                logger.info("Removed %d faces for provisional member %s after seed merge", removed, resolved.auto_merged_source)
            indexed_encoding = recognizer.register_face(image_bytes, member_id)
            if indexed_encoding is not None:
                database.update_member_encoding(member_id, indexed_encoding)
        if resolved.encoding_updated:
            logger.info("Refreshed stored encoding for member %s", member_id)

    recognition_duration = perf_counter() - recognition_start

    ad_generation_start = perf_counter()
    purchases = database.get_purchase_history(member_id)
    insights = analyse_purchase_intent(purchases, new_member=new_member)
    profile = database.get_member_profile(member_id)


    audience = _determine_audience(new_member=new_member, profile=profile, purchases=purchases)
    prediction_items: list[Any] = []
    predicted_dict: dict[str, Any] | None = None
    try:
        prediction_result = predict_next_purchases(purchases, profile=profile, insights=insights)
        if getattr(prediction_result, "items", None):
            prediction_items = list(prediction_result.items)
            predicted_dict = _prediction_to_dict(prediction_items[0])
    except Exception as exc:
        logging.warning('Prediction pipeline unavailable for %s: %s', member_id, exc)
        prediction_items = []
        predicted_dict = None
    
    creative = None
    if gemini.can_generate_ads and audience != "new":
        try:
            creative = gemini.generate_ad_copy(
                member_id,
                [
                    {
                        "member_code": purchase.member_code,
                        "product_category": purchase.product_category,
                        "internal_item_code": purchase.internal_item_code,
                        "item": purchase.item,
                        "purchased_at": purchase.purchased_at,
                        "unit_price": purchase.unit_price,
                        "quantity": purchase.quantity,
                        "total_price": purchase.total_price,
                    }
                    for purchase in purchases
                ],
                insights=insights,
                predicted=predicted_dict or {},
                audience=audience,
            )
        except GeminiUnavailableError as exc:
            logging.warning("Gemini ad generation unavailable: %s", exc)
    
    ad_generation_duration = perf_counter() - ad_generation_start
    
    image_filename = _persist_upload_image(member_id, image_bytes, mime_type)
    timings_snapshot = {
        "upload": round(upload_duration, 3),
        "recognition": round(recognition_duration, 3),
        "generation": round(ad_generation_duration, 3),
    }
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    profile_snapshot = _profile_snapshot(
        profile,
        member_id=member_id,
        image_filename=image_filename,
    )
    
    context = build_ad_context(
        member_id,
        purchases,
        insights=insights,
        profile=profile,
        profile_snapshot=profile_snapshot,
        creative=creative,
        predicted_item=predicted_dict,
        prediction_items=prediction_items,
        audience=audience,
        timings=timings_snapshot,
        detected_at=detected_at,
    )
    
    total_duration = perf_counter() - overall_start
    timings_snapshot["total"] = round(total_duration, 3)
    context.timings = timings_snapshot
    
    _latest_ad_hub.publish(_serialize_ad_context(context))
    
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

    hero_image_url = _resolve_template_image(context.template_id)
    payload = {
        "status": "ok",
        "member_id": member_id,
        "member_code": database.get_member_code(member_id),
        "new_member": new_member,
        "ad_url": url_for("render_ad", member_id=member_id, _external=True),
        "offer_url": url_for("render_ad_offer", member_id=member_id, _external=True),
        "template_id": context.template_id,
        "audience": audience,
        "scenario_key": context.scenario_key,
        "hero_image_url": hero_image_url,
        "headline": context.headline,
        "subheading": context.subheading,
        "highlight": context.highlight,
        "detected_at": detected_at,
    }
    if predicted_dict:
        payload["predicted"] = predicted_dict
    if context.cta_text:
        payload["cta_text"] = context.cta_text
    cta_href = context.cta_href or ""
    if not cta_href:
        cta_href = payload["offer_url"]
    elif cta_href.startswith("#") and audience == "member":
        cta_href = payload["offer_url"]
    payload["cta_href"] = cta_href

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
        return jsonify({"status": "error", "message": "source 與 target 參數必須提供"}), 400

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

    return jsonify(
        {
            "status": "ok",
            "merged_member": source_id,
            "into": target_id,
            "deleted_cloud_faces": deleted_faces,
            "encoding_updated": encoding_updated,
        }
    ), 200



def _prepare_member_ad_context(member_id: str) -> tuple[dict[str, object], MemberProfile | None]:
    purchases = database.get_purchase_history(member_id)
    insights = analyse_purchase_intent(purchases)
    profile = database.get_member_profile(member_id)

    audience = _determine_audience(new_member=not purchases, profile=profile, purchases=purchases)
    predicted_dict = None
    try:
        prediction_result = predict_next_purchases(purchases, profile=profile, insights=insights)
        predicted_item = prediction_result.items[0] if getattr(prediction_result, "items", None) else None
        predicted_dict = _prediction_to_dict(predicted_item)
    except Exception as exc:
        logging.warning("Prediction pipeline unavailable for %s: %s", member_id, exc)
        predicted_dict = None

    creative = None
    if gemini.can_generate_ads and audience != "new":
        try:
            creative = gemini.generate_ad_copy(
                member_id,
                [
                    {
                        "member_code": purchase.member_code,
                        "product_category": purchase.product_category,
                        "internal_item_code": purchase.internal_item_code,
                        "item": purchase.item,
                        "purchased_at": purchase.purchased_at,
                        "unit_price": purchase.unit_price,
                        "quantity": purchase.quantity,
                        "total_price": purchase.total_price,
                    }
                    for purchase in purchases
                ],
                insights=insights,
                predicted=predicted_dict or {},
                audience=audience,
            )
        except GeminiUnavailableError as exc:
            logging.warning("Gemini ad generation unavailable: %s", exc)

    context = build_ad_context(
        member_id,
        purchases,
        insights=insights,
        profile=profile,
        creative=creative,
        predicted_item=predicted_dict,
        audience=audience,
    )
    context_dict = _serialize_ad_context(context)
    if "profile" not in context_dict:
        context_dict["profile"] = context.profile or _profile_snapshot(profile, member_id=member_id)
    if "predicted_candidates" not in context_dict or context_dict["predicted_candidates"] is None:
        context_dict["predicted_candidates"] = [dict(candidate) for candidate in context.predicted_candidates]
    if not context_dict.get("timings"):
        context_dict["timings"] = dict(context.timings or {})
    if context.detected_at:
        context_dict["detected_at"] = context.detected_at

    return context_dict, profile


@app.get("/ad/<member_id>")
def render_ad(member_id: str):
    context_dict, profile = _prepare_member_ad_context(member_id)
    resolved_profile = context_dict.get("profile") or profile
    return render_template(
        "ad_latest.html",
        context=context_dict,
        profile=resolved_profile,
        stream_url=url_for("latest_ad_stream"),
    )


@app.get("/ad/<member_id>/offer")
def render_ad_offer(member_id: str):
    context_dict, _ = _prepare_member_ad_context(member_id)
    return render_template(
        "ad_offer.html",
        context=context_dict,
    )

@app.get("/ad/latest")
def render_latest_ad():
    context = _latest_ad_hub.snapshot()
    return render_template(
        "ad_latest.html",
        context=context or {},
        profile=(context or {}).get("profile") if context else None,
        stream_url=url_for("latest_ad_stream"),
    )


@app.get("/ad/latest/stream")

def latest_ad_stream():
    interval_param = request.args.get("interval")
    try:
        interval = 15.0 if not interval_param else float(interval_param)
        if interval <= 0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Invalid interval"}), 400

    once_flag = request.args.get("once", "").strip().lower()
    send_once = once_flag in {"1", "true", "yes"}

    def event_stream():
        queue = _latest_ad_hub.subscribe()
        try:
            while True:
                try:
                    context = queue.get(timeout=interval)
                except Empty:
                    yield "event: ping\n\n"
                    continue
                payload = json.dumps(context, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                if send_once:
                    break
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
            display_name=None,
            persona_label=None,
        )

    image_url = None
    if event.image_filename:
        image_url = url_for("serve_upload_image", filename=event.image_filename)

    profile = database.get_member_profile(event.member_id)
    persona_label: str | None = None
    display_name: str | None = None
    if profile:
        persona_label = _persona_label_display(profile.profile_label)
        display_name = profile.name or persona_label
    if display_name is None:
        display_name = "尚未命名會員"

    return render_template(
        "latest_upload.html",
        event=event,
        image_url=image_url,
        ad_url=url_for("render_ad", member_id=event.member_id, _external=True),
        offer_url=url_for("render_ad_offer", member_id=event.member_id, _external=True),
        members_url=url_for("member_directory"),
        display_name=display_name,
        persona_label=persona_label,
    )


@app.get("/members")
def member_directory():
    profiles = database.list_member_profiles()
    directory: list[dict[str, object]] = []

    for profile in profiles:
        purchases = []
        if profile.member_id:
            purchases = database.get_purchase_history(profile.member_id)

        persona_label = _persona_label_display(profile.profile_label)
        display_name = profile.name or persona_label or "尚未命名會員"

        directory.append(
            {
                "profile": profile,
                "purchases": purchases,
                "persona_label": persona_label,
                "display_name": display_name,
            }
        )

    return render_template("members.html", members=directory)


@app.get("/manager")
def manager_dashboard_view():
    member_id = (request.args.get("member_id") or "").strip()
    profiles = database.list_member_profiles()
    selectable_members = [profile for profile in profiles if profile.member_id]

    selected_id = member_id or (selectable_members[0].member_id if selectable_members else "")
    context: dict[str, object] | None = None
    error: str | None = None

    if selected_id:
        purchases = database.get_purchase_history(selected_id)
        profile = database.get_member_profile(selected_id)
        insights = analyse_purchase_intent(purchases)
        scenario_key = derive_scenario_key(insights, profile=profile)
        hero_image_url = _manager_hero_image(profile, scenario_key)
        prediction = predict_next_purchases(
            purchases,
            profile=profile,
            insights=insights,
            limit=7,
        )

        context = {
            "member_id": selected_id,
            "member": profile,
            "analysis": insights,
            "scenario_key": scenario_key,
            "hero_image_url": hero_image_url,
            "prediction": prediction,
            "ad_url": url_for("render_ad", member_id=selected_id, v2=1, _external=False),
            "offer_url": url_for("render_ad_offer", member_id=selected_id, _external=False),
        }
    else:
        error = "尚未建立任何會員資料"

    return render_template(
        "manager.html",
        context=context,
        member_id=member_id,
        members=selectable_members,
        error=error,
    )


@app.get("/uploads/<path:filename>")
def serve_upload_image(filename: str):
    return send_from_directory(UPLOAD_DIR, filename, conditional=True)


# === VM 圖庫對外供圖（/ad-assets/<filename>） ===
@app.get("/ad-assets/<path:filename>")
def serve_ad_asset(filename: str):
    ads_dir = current_app.config.get("ADS_DIR") or ""
    if not ads_dir:
        abort(404)

    # 安全拼接與邊界檢查
    safe_path = safe_join(ads_dir, filename)
    if not safe_path:
        abort(404)

    base = os.path.realpath(ads_dir)
    full = os.path.realpath(safe_path)
    if (not full.startswith(base)) or (not os.path.isfile(full)):
        abort(404)

    return send_from_directory(ads_dir, os.path.basename(full), conditional=True)


# === 新樣式預覽（不影響 /ad/<member_id>）===
@app.get("/ad-preview/<path:filename>")
def ad_preview(filename: str):
    hero_image_url = url_for("serve_ad_asset", filename=filename)
    return render_template(
        "ad.html",
        hero_image_url=hero_image_url,
        scenario_key=request.args.get("scenario_key", "brand_new"),
    )



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




def _manager_hero_image(profile, scenario_key: str) -> str | None:
    if profile and profile.first_image_filename:
        return url_for("serve_upload_image", filename=profile.first_image_filename)
    return _resolve_hero_image_url(scenario_key)


def _resolve_template_image(template_id: str) -> str | None:
    """Resolve a background image based on the ME000x template identifier."""
    filename = TEMPLATE_IMAGE_BY_ID.get(template_id) or TEMPLATE_IMAGE_BY_ID.get("ME0000")
    if not filename:
        return None

    ads_dir = current_app.config.get("ADS_DIR") or ""
    if ads_dir:
        candidate = Path(ads_dir) / filename
        if candidate.is_file():
            return url_for("serve_ad_asset", filename=filename)

    return url_for("static", filename=f"images/ads/{filename}")


def _resolve_hero_image_url(scenario_key: str) -> str | None:
    """Backward compatible wrapper that maps legacy scenario keys to templates."""
    if scenario_key in TEMPLATE_IMAGE_BY_ID:
        return _resolve_template_image(scenario_key)
    if scenario_key in AD_IMAGE_BY_SCENARIO:
        filename = AD_IMAGE_BY_SCENARIO[scenario_key]
        ads_dir = current_app.config.get("ADS_DIR") or ""
        if ads_dir:
            candidate = Path(ads_dir) / filename
            if candidate.is_file():
                return url_for("serve_ad_asset", filename=filename)
        return url_for("static", filename=f"images/ads/{filename}")
    if ":" in scenario_key:
        candidate = scenario_key.split(":")[-1]
        if candidate in TEMPLATE_IMAGE_BY_ID:
            return _resolve_template_image(candidate)
    return _resolve_template_image("ME0000")


def _extract_image_payload(req) -> Tuple[bytes, str]:
    """
    支援：
      - multipart/form-data: field 名稱優先找 'image'，也支援 'file', 'photo'
      - 直接傳 raw image/* 本體
      - （可選）JSON 內含 base64
    """
    # 1) multipart/form-data
    if req.files:
        for key in ("image", "file", "photo"):
            if key in req.files:
                uploaded = req.files[key]
                data = uploaded.read()
                if data:
                    return data, uploaded.mimetype or req.mimetype or "image/jpeg"

    # 2) raw image/*
    ct = (req.headers.get("Content-Type") or "").lower()
    if ct.startswith("image/"):
        data = req.get_data()
        if data:
            return data, req.mimetype or "image/jpeg"

    # 3) base64 in JSON（若你用得到）
    if req.is_json:
        j = req.get_json(silent=True) or {}
        b64 = j.get("image_base64")
        if b64:
            import base64
            return base64.b64decode(b64), j.get("mime_type", "image/jpeg")

    raise ValueError("No image data found in request")


def _create_welcome_purchase(member_id: str) -> None:
    now = datetime.now().replace(second=0, microsecond=0)
    database.add_purchase(
        member_id,
        item="歡迎禮盒",
        product_category="迎新禮遇",
        internal_item_code="WELCOME-001",
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


def _determine_audience(*, new_member: bool, profile: MemberProfile | None, purchases: Iterable[Purchase]) -> Literal["new", "guest", "member"]:
    purchases_list = list(purchases) if not isinstance(purchases, list) else purchases
    if new_member or not purchases_list:
        return "new"
    if profile and getattr(profile, "mall_member_id", ""):
        return "member"
    return "guest"


def _prediction_to_dict(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None
    if isinstance(item, Mapping):
        base = dict(item)
    else:
        base = {
            "product_code": getattr(item, "product_code", None),
            "product_name": getattr(item, "product_name", None),
            "category": getattr(item, "category", None),
            "category_label": getattr(item, "category_label", None),
            "price": getattr(item, "price", None),
            "view_rate_percent": getattr(item, "view_rate_percent", None),
            "probability": getattr(item, "probability", None),
            "probability_percent": getattr(item, "probability_percent", None),
        }
    cleaned = {key: value for key, value in base.items() if value is not None}
    return cleaned or None


def _profile_snapshot(
    profile: MemberProfile | None,
    *,
    member_id: str,
    image_filename: str | None = None,
) -> dict[str, Any]:
    photo_filename = image_filename
    if not photo_filename and profile and profile.first_image_filename:
        photo_filename = profile.first_image_filename

    photo_url: str
    static_candidate: str | None = None
    try:
        if photo_filename:
            uploads_path = UPLOAD_DIR / photo_filename
            if uploads_path.exists():
                photo_url = url_for("serve_upload_image", filename=photo_filename)
            else:
                static_candidate = photo_filename if photo_filename.startswith("images/") else f"images/{photo_filename}"
                photo_url = url_for("static", filename=static_candidate)
        else:
            photo_url = url_for("static", filename="images/face.jpg")
    except RuntimeError:
        if static_candidate:
            photo_url = f"/static/{static_candidate}"
        else:
            photo_url = f"/uploads/{photo_filename}" if photo_filename else "/static/images/face.jpg"

    return {
        "member_id": member_id,
        "member_code": getattr(profile, "mall_member_id", None) if profile else None,
        "name": getattr(profile, "name", None) if profile else None,
        "member_status": getattr(profile, "member_status", None) if profile else None,
        "joined_at": getattr(profile, "joined_at", None) if profile else None,
        "points_balance": getattr(profile, "points_balance", None) if profile else None,
        "gender": getattr(profile, "gender", None) if profile else None,
        "birth_date": getattr(profile, "birth_date", None) if profile else None,
        "phone": getattr(profile, "phone", None) if profile else None,
        "email": getattr(profile, "email", None) if profile else None,
        "address": getattr(profile, "address", None) if profile else None,
        "occupation": getattr(profile, "occupation", None) if profile else None,
        "profile_label": getattr(profile, "profile_label", None) if profile else None,
        "photo_url": photo_url,
    }


if hasattr(app, "before_first_request"):

    @app.before_first_request
    def _warmup_on_first_request():
        """Seed the latest ad hub only when the first request arrives."""
        _seed_latest_ad_hub()

else:

    @app.before_request
    def _warmup_on_first_request():  # pragma: no cover - Flask 3 fallback
        """Fallback for Flask versions without before_first_request."""
        global _warmup_ran
        if _warmup_ran:
            return
        with _warmup_once_lock:
            if _warmup_ran:
                return
            _seed_latest_ad_hub()
            _warmup_ran = True


if __name__ == "__main__":
    # 開發模式直接啟動；部署請用 gunicorn / systemd 並確保帶入 ADS_DIR / DB_PATH 等
    app.run(host="0.0.0.0", port=8000, debug=True)
