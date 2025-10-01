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

# --- Blueprints ---
# NOTE: 只註冊 blueprint；不要在這裡引用不存在的舊類別或函式
from .routes.identify import identify_bp         # 提供 /upload_face 與辨識流程
from .routes.predict import predict_bp           # 提供 /members/<id>/predictions 等 API
from .routes import adgen_blueprint              # 提供 /adgen（Vertex 圖文生成）

# 嘗試性匯入可選的 helper（新版可能已移除）
try:
    from .routes.predict import predict_next_purchases as _predict_next_purchases  # type: ignore
except Exception:  # pragma: no cover
    _predict_next_purchases = None

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("backend.app")

# -----------------------------------------------------------------------------
# Paths & Config
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
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
app.config["ADS_DIR"] = ADS_DIR

# 註冊 blueprints
app.register_blueprint(identify_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(adgen_blueprint)

# -----------------------------------------------------------------------------
# Services (Vertex AI / AWS Rekognition / DB)
# -----------------------------------------------------------------------------
gemini = GeminiService()
rekognition = RekognitionService()
database = Database(DB_PATH)
database.ensure_demo_data()


def _maybe_prepare_rekognition() -> None:
    """預設不重置；只有 REKOG_RESET=1 時才清空重建。否則僅確保存在。"""
    if not getattr(rekognition, "can_describe_faces", False):
        log.info("Rekognition not available; face features disabled")
        return
    try:
        if REKOG_RESET:
            if rekognition.reset_collection():
                log.warning("Amazon Rekognition collection reset for a clean start (REKOG_RESET=1)")
            else:
                log.warning("Amazon Rekognition collection reset requested but failed; continuing")
        else:
            ensure_fn = getattr(rekognition, "ensure_collection", None)
            if callable(ensure_fn):
                ensure_fn()
                log.info("Amazon Rekognition collection ensured (no reset)")
    except Exception as exc:  # pragma: no cover
        log.warning("Rekognition prepare step failed: %s", exc)


_maybe_prepare_rekognition()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def index() -> str:
    # 直接導到新版後台
    return render_template("manager.html", context={"prediction": {"window_label": "本月"}})


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
                        "member_code": p.member_code,
                        "item": p.item,
                        "purchased_at": p.purchased_at,
                        "unit_price": p.unit_price,
                        "quantity": p.quantity,
                        "total_price": p.total_price,
                    }
                    for p in purchases
                ],
                insights=insights,
            )
        except GeminiUnavailableError as exc:
            log.warning("Gemini ad generation unavailable: %s", exc)

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
            return {"status": "idle", "message": "No uploads have been recorded yet."}, "0"

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
        purchases = database.get_purchase_history(profile.member_id) if profile.member_id else []
        directory.append({"profile": profile, "purchases": purchases})
    return render_template("members.html", members=directory)


@app.get("/manager")
def manager_dashboard_view():
    member_id = (request.args.get("member_id") or "").strip()
    profiles = database.list_member_profiles()
    selectable_members = [p for p in profiles if p.member_id]

    selected_id = member_id or (selectable_members[0].member_id if selectable_members else "")
    context: dict[str, object] | None = None
    error: str | None = None

    if selected_id:
        purchases = database.get_purchase_history(selected_id)
        profile = database.get_member_profile(selected_id)
        insights = analyse_purchase_intent(purchases)
        scenario_key = derive_scenario_key(insights, profile=profile)
        hero_image_url = _manager_hero_image(profile, scenario_key)

        # 預測：優先使用新版 routes.predict 提供的 helper；沒有就使用安全 fallback
        if callable(_predict_next_purchases):
            prediction = _predict_next_purchases(
                purchases,
                profile=profile,
                insights=insights,
                limit=7,
            )
        else:
            prediction = {
                "window_label": "本月",
                "items": [],
                "explain": "fallback: 無 predict_next_purchases，僅顯示基本資料",
            }

        context = {
            "member_id": selected_id,
            "member": profile,
            "analysis": insights,
            "scenario_key": scenario_key,
            "hero_image_url": hero_image_url,
            "prediction": prediction,
            "ad_url": url_for("render_ad", member_id=selected_id, v2=1, _external=False),
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
    ads_dir = current_app.config.get("ADS_DIR") or ""
    ads_path = Path(ads_dir)
    exists = ads_path.is_dir()
    sample: list[str] = []
    if exists:
        try:
            sample = [e.name for e in sorted(ads_path.iterdir())[:10]]
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
            log.warning("Ads directory write test failed for %s: %s", ads_path, exc)
        finally:
            try:
                test_path.unlink(missing_ok=True)
            except OSError:
                pass
    sample: list[str] = []
    if exists:
        try:
            sample = [e.name for e in sorted(ads_path.iterdir())[:10]]
        except OSError as exc:
            log.warning("Failed to inspect ads directory %s: %s", ads_path, exc)

    bucket_name = os.environ.get("ASSET_BUCKET", "")
    gcs_status: dict[str, object] = {"bucket": bucket_name, "reachable": False}
    if bucket_name:
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            gcs_status["reachable"] = bucket.exists()
        except Exception as exc:  # pragma: no cover
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
    if profile and getattr(profile, "first_image_filename", None):
        return url_for("serve_upload_image", filename=profile.first_image_filename)
    return _resolve_hero_image_url(scenario_key)


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

    return url_for("static", filename=f"images/ads/{filename}")


# 下面兩個 helper 目前僅供 /upload 記錄圖用的舊流程（identify_bp 已接手 /upload_face）
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
    except OSError as exc:  # pragma: no cover
        log.warning("Failed to persist uploaded image %s: %s", path, exc)
        return None
    return filename


def _purge_upload_images(filenames: Iterable[str]) -> None:
    for filename in set(filter(None, filenames)):
        path = UPLOAD_DIR / filename
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:  # pragma: no cover
            log.warning("Failed to delete old upload image %s: %s", path, exc)


if __name__ == "__main__":
    # 本地除錯使用；正式以 systemd/gunicorn 啟動
    app.run(host="0.0.0.0", port=8000, debug=True)
