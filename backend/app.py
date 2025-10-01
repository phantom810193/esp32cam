# backend/app.py
"""Flask backend for the ESP32-CAM retail advertising MVP."""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import time
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable, Tuple
from uuid import uuid4

from flask import (
    Flask,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    Response,
    send_from_directory,
    stream_with_context,
    url_for,
)
from google.cloud import storage
from werkzeug.utils import safe_join

from .advertising import (
    AD_IMAGE_BY_SCENARIO,
    analyse_purchase_intent,
    build_ad_context,
    derive_scenario_key,
)
from .ai import GeminiService, GeminiUnavailableError
from .aws import RekognitionService
from .database import Database
from .prediction import predict_next_purchases
from .recognizer import FaceRecognizer
from .routes import adgen_blueprint, generate_vertex_ad

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# -----------------------------------------------------------------------------
# Paths & Config
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# 允許用環境變數覆蓋 DB 位置（預設為 repo 內 data/mvp.sqlite3）
DB_PATH = Path(os.environ.get("DB_PATH", str(DATA_DIR / "mvp.sqlite3")))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(DATA_DIR / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# VM 外部圖庫（可透過環境變數覆蓋）
ADS_DIR = os.environ.get("ADS_DIR", "/srv/esp32-ads")
Path(ADS_DIR).mkdir(parents=True, exist_ok=True)

# Rekognition 開關（預設不 reset；只有顯式設 1 才做）
REKOG_RESET = os.environ.get("REKOG_RESET", "0") == "1"

# -----------------------------------------------------------------------------
# Flask App
# -----------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static",
)
app.config["JSON_AS_ASCII"] = False
app.config["ADS_DIR"] = ADS_DIR  # 儲存為字串路徑
app.register_blueprint(adgen_blueprint)

# -----------------------------------------------------------------------------
# Services (Vertex AI / AWS Rekognition / DB)
# -----------------------------------------------------------------------------
gemini = GeminiService()

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
            # 輕量動作：確保 collection 存在即可（若你的 service 沒有 ensure_*，可安全略過）
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

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
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
    insights = analyse_purchase_intent(purchases, new_member=new_member)
    profile = database.get_member_profile(member_id)

    # 只有非 brand_new 才呼叫 LLM 產生文案
    creative = None
    if gemini.can_generate_ads and insights.scenario != "brand_new":
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
                insights=insights,
            )
        except GeminiUnavailableError as exc:
            logging.warning("Gemini ad generation unavailable: %s", exc)
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

    scenario_key = derive_scenario_key(insights, profile=profile)
    hero_image_url = _resolve_hero_image_url(scenario_key)

    payload = {
        "status": "ok",
        "member_id": member_id,
        "member_code": database.get_member_code(member_id),
        "new_member": new_member,
        "ad_url": url_for("render_ad", member_id=member_id, _external=True),
        "scenario_key": scenario_key,
        "hero_image_url": hero_image_url,
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


@app.get("/ad/<member_id>")
def render_ad(member_id: str):
    purchases = database.get_purchase_history(member_id)
    insights = analyse_purchase_intent(purchases)
    profile = database.get_member_profile(member_id)

    creative = None
    if gemini.can_generate_ads and insights.scenario != "brand_new":
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
                insights=insights,
            )
        except GeminiUnavailableError as exc:
            logging.warning("Gemini ad generation unavailable: %s", exc)

    context = build_ad_context(
        member_id,
        purchases,
        insights=insights,
        profile=profile,
        creative=creative,
    )
    hero_image_url = _resolve_hero_image_url(context.scenario_key)

    return render_template(
        "ad.html",
        context=context,
        hero_image_url=hero_image_url,
        scenario_key=context.scenario_key,
    )


@app.get("/ad/latest/stream")
def latest_ad_stream() -> Response:
    """Server-Sent Events feed with the most recent personalised ad metadata."""

    interval_arg = request.args.get("interval", "2.0")
    try:
        poll_interval = float(interval_arg)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid interval"}), 400

    poll_interval = max(0.5, min(10.0, poll_interval))
    send_once = request.args.get("once", "0") == "1"
    last_event_id = request.headers.get("Last-Event-ID") or request.args.get("lastEventId")

    def _event_payload(event) -> tuple[dict[str, object], str]:
        if event is None:
            return {
                "status": "idle",
                "message": "No uploads have been recorded yet.",
            }, "0"

        ad_url = url_for("render_ad", member_id=event.member_id, _external=True)
        image_url = (
            url_for("serve_upload_image", filename=event.image_filename, _external=True)
            if event.image_filename
            else None
        )

        payload = {
            "status": "ok",
            "event_id": event.id,
            "created_at": event.created_at,
            "member_id": event.member_id,
            "member_code": event.member_code,
            "ad_url": ad_url,
        }
        if image_url:
            payload["image_url"] = image_url
        return payload, str(event.id)

    def _format_sse(event_id: str, payload: dict[str, object]) -> str:
        data = json.dumps(payload, ensure_ascii=False)
        return f"id: {event_id}\nevent: latest-ad\ndata: {data}\n\n"

    @stream_with_context
    def _generate():
        last_sent_id = last_event_id or ""

        while True:
            event = database.get_latest_upload_event()
            payload, event_id = _event_payload(event)

            if not send_once and last_sent_id == event_id and event is not None:
                time.sleep(poll_interval)
                continue

            chunk = _format_sse(event_id, payload)
            yield chunk
            last_sent_id = event_id

            if send_once:
                break

            time.sleep(poll_interval)

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no",
    }
    return Response(_generate(), headers=headers)


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

        vertex_ad: dict[str, object] | None = None
        vertex_ad_error: str | None = None
        top_prediction = prediction.items[0] if prediction.items else None

        if top_prediction:
            sku = top_prediction.product_code or top_prediction.product_name
            member_payload = _summarise_member_profile(profile)
            missing_env = [
                env
                for env in ("GCP_PROJECT_ID", "ASSET_BUCKET", "GOOGLE_APPLICATION_CREDENTIALS")
                if not os.environ.get(env)
            ]
            if missing_env:
                vertex_ad_error = (
                    "Vertex AI 服務尚未設定環境變數：" + ", ".join(sorted(missing_env))
                )
            else:
                try:
                    vertex_ad = generate_vertex_ad(sku, member_payload)
                except Exception as exc:  # pylint: disable=broad-except
                    logging.warning("Vertex AI creative unavailable for %s: %s", sku, exc)
                    vertex_ad_error = str(exc)

        context = {
            "member_id": selected_id,
            "member": profile,
            "analysis": insights,
            "scenario_key": scenario_key,
            "hero_image_url": hero_image_url,
            "prediction": prediction,
            "ad_url": url_for("render_ad", member_id=selected_id, v2=1, _external=False),
            "top_prediction": top_prediction,
            "vertex_ad": vertex_ad,
            "vertex_ad_error": vertex_ad_error,
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


@app.get("/health")
def health_check():
    # 強化健康檢查，方便遠端排錯
    ads_dir = current_app.config.get("ADS_DIR") or ""
    ads_path = Path(ads_dir)
    exists = ads_path.is_dir()
    sample: list[str] = []
    if exists:
        try:
            sample = [entry.name for entry in sorted(ads_path.iterdir())[:10]]
        except OSError:
            sample = []
    return jsonify(
        {
            "status": "ok",
            "ads_dir": ads_dir,
            "ads_dir_exists": exists,
            "ads_dir_sample": sample,
        }
    )


@app.get("/healthz")
def extended_health_check():
    ads_dir = current_app.config.get("ADS_DIR") or ""
    ads_path = Path(ads_dir)
    exists = ads_path.is_dir()
    writable = False
    write_error: str | None = None
    if exists:
        test_path = ads_path / f".healthz-{uuid4().hex}.tmp"
        try:
            with open(test_path, "w", encoding="utf-8") as handle:
                handle.write("ok")
            writable = True
        except OSError as exc:
            write_error = str(exc)
            logging.warning("Ads directory write test failed for %s: %s", ads_path, exc)
        finally:
            try:
                test_path.unlink(missing_ok=True)
            except OSError:
                pass
    sample: list[str] = []
    if exists:
        try:
            sample = [entry.name for entry in sorted(ads_path.iterdir())[:10]]
        except OSError as exc:
            logging.warning("Failed to inspect ads directory %s: %s", ads_path, exc)

    bucket_name = os.environ.get("ASSET_BUCKET", "")
    gcs_status: dict[str, object] = {"bucket": bucket_name, "reachable": False}
    if bucket_name:
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            gcs_status["reachable"] = bucket.exists()
        except Exception as exc:  # pylint: disable=broad-except
            gcs_status["error"] = str(exc)
    else:
        gcs_status["error"] = "ASSET_BUCKET not configured"

    # 併入 Vertex AI 健康資訊
    ai_status = gemini.health_probe()

    return jsonify(
        {
            "status": "ok" if ai_status.get("vertexai") == "initialized" else "degraded",
            "ads_dir": str(ads_path),
            "ads_dir_exists": exists,
            "ads_dir_writable": writable,
            "ads_dir_sample": sample,
            "ads_dir_write_error": write_error,
            "gcs": gcs_status,
            "vertex": ai_status,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manager_hero_image(profile, scenario_key: str) -> str | None:
    if profile and profile.first_image_filename:
        return url_for("serve_upload_image", filename=profile.first_image_filename)
    return _resolve_hero_image_url(scenario_key)


def _summarise_member_profile(profile) -> dict[str, object]:
    if not profile:
        return {}

    return {
        "member_id": profile.member_id,
        "mall_member_id": profile.mall_member_id,
        "member_status": profile.member_status,
        "joined_at": profile.joined_at,
        "points_balance": profile.points_balance,
        "gender": profile.gender,
        "birth_date": profile.birth_date,
        "phone": profile.phone,
        "email": profile.email,
        "address": profile.address,
        "occupation": profile.occupation,
        "profile_label": profile.profile_label,
    }


def _resolve_hero_image_url(scenario_key: str) -> str | None:
    """優先使用 VM 圖庫（/ad-assets），缺檔時回退到 repo 內的 /static/images/ads。"""
    filename = AD_IMAGE_BY_SCENARIO.get(scenario_key) or AD_IMAGE_BY_SCENARIO.get("brand_new")
    if not filename:
        return None

    ads_dir = current_app.config.get("ADS_DIR") or ""
    if ads_dir:
        candidate = Path(ads_dir) / filename
        if candidate.is_file():
            return url_for("serve_ad_asset", filename=filename)

    # fallback：讓畫面至少有圖（走 Flask static）
    return url_for("static", filename=f"images/ads/{filename}")


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
        path.write_bytes(image_bytes)
    except OSError as exc:
        logging.warning("Failed to persist uploaded image %s: %s", path, exc)
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
    # 開發模式直接啟動；部署請用 gunicorn / systemd 並確保帶入 ADS_DIR / DB_PATH 等
    app.run(host="0.0.0.0", port=8000, debug=True)
