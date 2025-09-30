# backend/app.py
"""Flask backend for the ESP32-CAM retail advertising MVP."""
from __future__ import annotations

import io
import logging
import os
import mimetypes
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
    send_file,
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

# --- PIL for drawing ad copy ---
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = ImageDraw = ImageFont = None  # Pillow 未安裝時避免匯入期就爆

logging.basicConfig(level=logging.INFO)

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
        logging.warning("Amazon Rekognition collection reset failed; continuing with existing entries")
recognizer = FaceRecognizer(rekognition)
database = Database(DB_PATH)
database.ensure_demo_data()

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> str:
    return render_template("index.html")

# 固定網址管理後台：GET 顯示、POST 上傳照片辨識後導回
@app.route("/manager", methods=["GET", "POST"])
def manager_dashboard_view():
    if request.method == "POST":
        # 允許直接在 /manager 上傳圖片 → 找出 member_id → 導回 GET
        try:
            image_bytes, mime_type = _extract_image_payload(request)
            encoding = recognizer.encode(image_bytes, mime_type=mime_type)
        except Exception as exc:
            return render_template(
                "manager.html",
                context=None,
                members=_selectable_members(),
                error=f"上傳/辨識失敗：{exc}",
            ), 400

        member_id, _ = database.find_member_by_encoding(encoding, recognizer)
        if member_id is None:
            # 新客：建立、給歡迎禮、索引 Rekognition
            member_id = database.create_member(encoding, recognizer.derive_member_id(encoding))
            _create_welcome_purchase(member_id)
            indexed = recognizer.register_face(image_bytes, member_id)
            if indexed is not None:
                database.update_member_encoding(member_id, indexed)

            # 存上傳圖（讓 hero 能顯示）
            _persist_upload_image(member_id, image_bytes, mime_type)

        return redirect(url_for("manager_dashboard_view", member_id=member_id))

    # --- GET：載入畫面 ---
    member_id = (request.args.get("member_id") or "").strip()
    selectable_members = _selectable_members()

    # 沒給 member_id → 用最近一次上傳事件的 member，或第一筆
    if not member_id:
        last = database.get_latest_upload_event()
        if last:
            member_id = last.member_id
        elif selectable_members:
            member_id = selectable_members[0].member_id

    context: dict[str, object] | None = None
    error: str | None = None

    if member_id:
        purchases = database.get_purchase_history(member_id)
        profile = database.get_member_profile(member_id)
        insights = analyse_purchase_intent(purchases)
        scenario_key = derive_scenario_key(insights, profile=profile)
        hero_image_url = _manager_hero_image(profile, scenario_key)
        prediction = predict_next_purchases(
            purchases, profile=profile, insights=insights, limit=7
        )

        context = {
            "member_id": member_id,
            "member": profile,
            "analysis": insights,
            "scenario_key": scenario_key,
            "hero_image_url": hero_image_url,
            "prediction": prediction,
            "ad_url": url_for("render_ad", member_id=member_id, v2=1, _external=False),
        }
    else:
        error = "尚未建立任何會員資料"

    return render_template("manager.html", context=context, members=selectable_members, error=error)

def _selectable_members():
    profiles = database.list_member_profiles()
    return [p for p in profiles if p.member_id]

# ---------------------------------------------------------------------------
# 顧客端：上傳臉部 → 產生廣告 URL
# ---------------------------------------------------------------------------

@app.post("/upload_face")
def upload_face():
    """Receive an image and return the member identifier + ad url."""
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

    creative = None
    if gemini.can_generate_ads and insights.scenario != "brand_new":
        try:
            creative = gemini.generate_ad_copy(
                member_id,
                [
                    {
                        "member_code": x.member_code,
                        "item": x.item,
                        "purchased_at": x.purchased_at,
                        "unit_price": x.unit_price,
                        "quantity": x.quantity,
                        "total_price": x.total_price,
                    }
                    for x in purchases
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

# ---------------------------------------------------------------------------
# 廣告頁（HTML）與動態合成廣告圖（JPG）
# ---------------------------------------------------------------------------

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
                        "member_code": x.member_code,
                        "item": x.item,
                        "purchased_at": x.purchased_at,
                        "unit_price": x.unit_price,
                        "quantity": x.quantity,
                        "total_price": x.total_price,
                    }
                    for x in purchases
                ],
                insights=insights,
            )
        except GeminiUnavailableError as exc:
            logging.warning("Gemini ad generation unavailable: %s", exc)

    context = build_ad_context(member_id, purchases, insights=insights, profile=profile, creative=creative)
    hero_image_url = _resolve_hero_image_url(context.scenario_key)

    return render_template("ad.html", context=context, hero_image_url=hero_image_url, scenario_key=context.scenario_key)

@app.get("/ad-image/<member_id>")
def ad_image(member_id: str):
    """回傳一張已將 20 字內標題文案畫到 hero 圖上的 JPG（底部半透明黑條 + 白字）。"""
    purchases = database.get_purchase_history(member_id)
    insights = analyse_purchase_intent(purchases)
    profile = database.get_member_profile(member_id)
    scenario_key = derive_scenario_key(insights, profile=profile)

    # 取得底圖「實際路徑」
    image_path = _resolve_hero_image_path(scenario_key)
    if image_path is None or not image_path.is_file() or Image is None:
        abort(404)

    base = Image.open(image_path).convert("RGB")
    W, H = base.size

    # 生成 20 字內標題文案（優先 Gemini，否則模板）
    copy_text = None
    if gemini.can_generate_ads and insights.scenario != "brand_new":
        try:
            copy_text = gemini.generate_short_headline(member_id, insights=insights, max_chars=20)
        except Exception:
            copy_text = None
    if not copy_text:
        item = getattr(insights, "recommended_item", None) or "精選商品"
        templates = [
            f"今天就來 {item}",
            f"{item} 限時優惠！",
            f"{item} 熱銷補貨到店",
            f"回饋加碼：{item}",
            f"會員私享：{item}",
        ]
        copy_text = templates[hash(member_id) % len(templates)]
    # 安全截斷到 20 個字
    if len(copy_text) > 20:
        copy_text = copy_text[:20]

    # 合成：半透明黑條 + 置中文字
    rgba = base.convert("RGBA")
    bar_h = int(H * 0.16)           # 黑條高度
    overlay = Image.new("RGBA", (W, bar_h), (0, 0, 0, 170))
    rgba.alpha_composite(overlay, dest=(0, H - bar_h))

    # 字型（優先 Noto CJK）
    font = None
    for fp in [
        "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Medium.otf",
        "/usr/share/fonts/truetype/noto/NotoSansTC-Medium.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if os.path.isfile(fp):
            try:
                font = ImageFont.truetype(fp, size=int(bar_h * 0.45))
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(rgba)
    bbox = draw.textbbox((0, 0), copy_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((W - tw) // 2, H - bar_h + (bar_h - th) // 2), copy_text, fill=(255, 255, 255, 255), font=font)

    buf = io.BytesIO()
    rgba.convert("RGB").save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg", download_name=f"{member_id}.jpg")

# ---------------------------------------------------------------------------
# 成員清單、最新上傳、靜態資源
# ---------------------------------------------------------------------------

@app.get("/latest_upload")
def latest_upload_dashboard():
    event = database.get_latest_upload_event()
    if event is None:
        return render_template("latest_upload.html", event=None, members_url=url_for("member_directory"))

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

@app.get("/uploads/<path:filename>")
def serve_upload_image(filename: str):
    return send_from_directory(UPLOAD_DIR, filename, conditional=True)

# VM 圖庫對外供圖（/ad-assets/<filename>）
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

# 新樣式預覽（不影響 /ad/<member_id>）
@app.get("/ad-preview/<path:filename>")
def ad_preview(filename: str):
    hero_image_url = url_for("serve_ad_asset", filename=filename)
    return render_template("ad.html", hero_image_url=hero_image_url, scenario_key=request.args.get("scenario_key", "brand_new"))

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
    """回傳 URL。優先 VM ADS_DIR，缺檔時回退 /static/images/ads。"""
    filename = AD_IMAGE_BY_SCENARIO.get(scenario_key) or AD_IMAGE_BY_SCENARIO.get("brand_new")
    if not filename:
        return None
    ads_dir = current_app.config.get("ADS_DIR") or ""
    if ads_dir and (Path(ads_dir) / filename).is_file():
        return url_for("serve_ad_asset", filename=filename)
    return url_for("static", filename=f"images/ads/{filename}")

def _resolve_hero_image_path(scenario_key: str) -> Path | None:
    """回傳實際檔案 Path，給 Pillow 開檔用。"""
    filename = AD_IMAGE_BY_SCENARIO.get(scenario_key) or AD_IMAGE_BY_SCENARIO.get("brand_new")
    if not filename:
        return None
    ads_dir = current_app.config.get("ADS_DIR") or ""
    cand = Path(ads_dir) / filename
    if ads_dir and cand.is_file():
        return cand
    # fallback 到 repo 內的靜態圖
    cand = STATIC_DIR / "images" / "ads" / filename
    return cand if cand.is_file() else None

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
    app.run(host="0.0.0.0", port=8000, debug=True)
