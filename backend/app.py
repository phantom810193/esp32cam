# backend/app.py
"""Flask backend for the ESP32-CAM retail advertising MVP."""
from __future__ import annotations

import io
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
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
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

logging.basicConfig(level=logging.INFO)

# --- Optional: Pillow for drawing copy onto image ---
try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception as exc:  # Pillow not installed yet
    logging.warning("Pillow not available (image compose fallback to plain image): %s", exc)
    PIL_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "mvp.sqlite3"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# === VM 外部圖庫設定（可透過環境變數覆蓋） ===
ADS_DIR = os.environ.get("ADS_DIR", "/srv/esp32-ads")
Path(ADS_DIR).mkdir(parents=True, exist_ok=True)

# === 確保靜態測試檔存在，供健康檢查 HEAD 驗證用 ===
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
HELLO_TXT = STATIC_DIR / "hello.txt"
if not HELLO_TXT.exists():
    try:
        HELLO_TXT.write_text("ok\n", encoding="utf-8")
    except OSError:
        pass  # 若 filesystem 受限，不影響主要流程

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
app.config["JSON_AS_ASCII"] = False
app.config["ADS_DIR"] = ADS_DIR  # 儲存為字串路徑

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


@app.get("/")
def index() -> str:
    return render_template("index.html")


# ========== Core: Upload & recognize ==========
def _recognize_or_create_member(image_bytes: bytes, mime_type: str) -> tuple[str, bool]:
    """回傳 (member_id, new_member)"""
    encoding = recognizer.encode(image_bytes, mime_type=mime_type)
    member_id, _ = database.find_member_by_encoding(encoding, recognizer)
    new_member = False
    if member_id is None:
        member_id = database.create_member(encoding, recognizer.derive_member_id(encoding))
        _create_welcome_purchase(member_id)
        new_member = True

    if new_member:
        indexed_encoding = recognizer.register_face(image_bytes, member_id)
        if indexed_encoding is not None:
            database.update_member_encoding(member_id, indexed_encoding)
    return member_id, new_member


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
        member_id, new_member = _recognize_or_create_member(image_bytes, mime_type)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 422
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

    payload = {
        "status": "ok",
        "member_id": member_id,
        "member_code": database.get_member_code(member_id),
        "new_member": new_member,
        "ad_url": url_for("render_ad", member_id=member_id, _external=True),
        "scenario_key": scenario_key,
        "hero_image_url": url_for("ad_composed_image", member_id=member_id, _external=True, ts=int(time.time())),
    }
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


# ========== 顧客廣告（自動改用「已畫字的合成圖」） ==========
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
            logging.warning("Gemini ad generation unavailable: %s", exc)

    context = build_ad_context(
        member_id,
        purchases,
        insights=insights,
        profile=profile,
        creative=creative,
    )
    # 重要：讓 ad.html 使用「動態合成圖」
    hero_image_url = url_for("ad_composed_image", member_id=member_id, ts=int(time.time()))

    return render_template(
        "ad.html",
        context=context,
        hero_image_url=hero_image_url,
        scenario_key=context.scenario_key,
    )


# ========== 動態合成圖（把文案畫到圖片上） ==========
@app.get("/ad-image/<member_id>")
def ad_composed_image(member_id: str):
    purchases = database.get_purchase_history(member_id)
    insights = analyse_purchase_intent(purchases)
    profile = database.get_member_profile(member_id)
    scenario_key = derive_scenario_key(insights, profile=profile)

    # 20 字內的活潑中文文案（沒有 Gemini 也會有）
    copy_text = _generate_creative_text(insights=insights, profile=profile, purchases=purchases)

    base_path = _ad_base_image_path(scenario_key)
    if not base_path or not base_path.exists():
        abort(404)

    if not PIL_AVAILABLE:
        # Pillow 不存在就回傳純圖（不畫字）
        return send_from_directory(base_path.parent, base_path.name, conditional=True)

    try:
        image = _compose_ad_bitmap(str(base_path), copy_text)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90, optimize=True)
        buf.seek(0)
        return current_app.response_class(buf.getvalue(), mimetype="image/jpeg")
    except Exception as exc:
        logging.warning("compose ad image failed: %s", exc)
        # 失敗回退純圖
        return send_from_directory(base_path.parent, base_path.name, conditional=True)


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


# ========== 經理後台：固定網址 /manager（支援 GET/POST 上傳辨識） ==========
@app.route("/manager", methods=["GET", "POST"])
def manager_dashboard_view():
    # 若 POST 圖片，就先辨識→redirect 到該 member_id
    if request.method == "POST":
        try:
            image_bytes, mime_type = _extract_image_payload(request)
            member_id, _ = _recognize_or_create_member(image_bytes, mime_type)
            return redirect(url_for("manager_dashboard_view", member_id=member_id))
        except Exception as exc:
            logging.exception("manager upload failed: %s", exc)
            # 帶錯誤訊息回頁面
            return render_template("manager.html", context=None, member_id="", members=[], error=str(exc))

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


# ========== 靜態與資產 ==========
@app.get("/uploads/<path:filename>")
def serve_upload_image(filename: str):
    return send_from_directory(UPLOAD_DIR, filename, conditional=True)


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


@app.get("/ad-preview/<path:filename>")
def ad_preview(filename: str):
    hero_image_url = url_for("serve_ad_asset", filename=filename)
    return render_template(
        "ad.html",
        hero_image_url=hero_image_url,
        scenario_key=request.args.get("scenario_key", "brand_new"),
    )


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------
def _health_payload() -> dict:
    ads_dir = current_app.config.get("ADS_DIR") or ""
    ads_path = Path(ads_dir)
    exists = ads_path.is_dir()
    sample: list[str] = []
    if exists:
        try:
            sample = [entry.name for entry in sorted(ads_path.iterdir())[:10]]
        except OSError:
            sample = []
    return {
        "status": "ok",
        "ads_dir": ads_dir,
        "ads_dir_exists": exists,
        "ads_dir_sample": sample,
        "static_example": "/static/hello.txt",
    }


@app.get("/health")
def health_check():
    return jsonify(_health_payload())


@app.get("/healthz")
def healthz_check():
    return jsonify(_health_payload())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _manager_hero_image(profile, scenario_key: str) -> str | None:
    if profile and profile.first_image_filename:
        return url_for("serve_upload_image", filename=profile.first_image_filename)
    return _resolve_hero_image_url(scenario_key)


def _resolve_hero_image_url(scenario_key: str) -> str | None:
    filename = AD_IMAGE_BY_SCENARIO.get(scenario_key) or AD_IMAGE_BY_SCENARIO.get("brand_new")
    if not filename:
        return None
    ads_dir = current_app.config.get("ADS_DIR") or ""
    if ads_dir:
        candidate = Path(ads_dir) / filename
        if candidate.is_file():
            return url_for("serve_ad_asset", filename=filename)
    return url_for("static", filename=f"images/ads/{filename}")


def _ad_base_image_path(scenario_key: str) -> Path | None:
    """找出實體檔案路徑（優先 ADS_DIR，否則走 repo 靜態）"""
    filename = AD_IMAGE_BY_SCENARIO.get(scenario_key) or AD_IMAGE_BY_SCENARIO.get("brand_new")
    if not filename:
        return None
    ads_dir = Path(current_app.config.get("ADS_DIR") or "")
    if ads_dir and (ads_dir / filename).is_file():
        return ads_dir / filename
    fallback = STATIC_DIR / "images" / "ads" / filename
    return fallback if fallback.is_file() else None


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


# --- 文案產生（無 Gemini 也會有 20 字內中文 copy） ---
def _generate_creative_text(insights, profile, purchases) -> str:
    """
    產生短文案（<=20字）。策略：
    1) 以 insights.recommended_item 或預測第一名為主
    2) 針對常見類型給一句活潑標語
    """
    item = getattr(insights, "recommended_item", None) or ""
    if not item:
        # 嘗試從歷史推一個熱門
        top = None
        try:
            preds = predict_next_purchases(purchases, profile=profile, insights=insights, limit=1)
            if preds and isinstance(preds, list):
                top = getattr(preds[0], "product_name", None) or getattr(preds[0], "item", None)
        except Exception:
            top = None
        item = top or "本月精選"

    # 超短句模板（20 字內）
    templates = [
        f"{item}熱銷回歸，限時搶！",
        f"{item}精選推薦，錯過可惜！",
        f"{item}人氣王，現在下單！",
        f"今天就來一份 {item}！",
        f"{item}新客加碼，快收藏！",
        f"{item}專屬優惠，立即享有！",
    ]
    # 用 item 長度決定挑哪句，以免超過 20 字
    for s in templates:
        if len(s) <= 20:
            return s
    # 保底截斷
    return (templates[0])[:20]


def _load_font(pt: int):
    """盡力載入支援中文的字型，最後退回 PIL 內建。"""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansTC-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, pt)
            except Exception:
                continue
    return ImageFont.load_default()


def _compose_ad_bitmap(base_image_path: str, text: str):
    """把文案畫到圖片下方（半透明黑帶 + 白字）。"""
    image = Image.open(base_image_path).convert("RGB")
    W, H = image.size

    # 半透明黑帶
    band_h = int(H * 0.18)
    overlay = Image.new("RGBA", (W, band_h), (0, 0, 0, 150))
    image.paste(overlay, (0, H - band_h), overlay)

    # 文字
    font = _load_font(max(int(H * 0.06), 28))
    draw = ImageDraw.Draw(image)
    padding = int(0.04 * W)
    draw.text((padding, H - band_h + int(0.2 * band_h)), text, fill=(255, 255, 255), font=font)

    return image


if __name__ == "__main__":
    # 開發模式直接啟動；部署請用 gunicorn / systemd 並確保帶入 ADS_DIR
    app.run(host="0.0.0.0", port=8000, debug=True)
