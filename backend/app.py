"""Flask backend for the ESP32-CAM retail advertising MVP."""
from __future__ import annotations

import io
import logging
import mimetypes
import os
import textwrap
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable, Tuple
from backend.reco import recommend_for_member
from backend.ai_gemini import ad_copy_unregistered, ad_copy_registered
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

# Pillow for drawing copy onto images
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "mvp.sqlite3"
UPLOAD_DIR = DATA_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# === VM 外部圖庫設定（可透過環境變數覆蓋） ===
ADS_DIR = os.environ.get("ADS_DIR", "/srv/esp32-ads")
Path(ADS_DIR).mkdir(parents=True, exist_ok=True)

# === 確保靜態測試檔存在，供健康檢查 HEAD 驗證用 ===
HELLO_TXT = STATIC_DIR / "hello.txt"
if not HELLO_TXT.exists():
    try:
        HELLO_TXT.write_text("ok\n", encoding="utf-8")
    except OSError:
        pass

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
app.config["JSON_AS_ASCII"] = False
app.config["ADS_DIR"] = ADS_DIR  # 儲存字串路徑

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


def _manager_hero_image(profile, scenario_key: str) -> str | None:
    if profile and profile.first_image_filename:
        return url_for("serve_upload_image", filename=profile.first_image_filename)
    return _resolve_hero_image_url(scenario_key)


def _find_cjk_font() -> str | None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _generate_short_copy(insights, profile) -> str:
    """產生 20 字內活潑文案（不靠 LLM 也能跑）"""
    item = getattr(insights, "recommended_item", None) or "人氣商品"
    scenario = getattr(insights, "scenario", "brand_new")
    if scenario == "brand_new":
        return "新朋友限定｜熱門好物嚐鮮！"
    if profile and getattr(profile, "profile_label", "").startswith("dessert"):
        return f"{item} 甜蜜開賣，限時加碼！"
    if profile and "fitness" in (profile.profile_label or ""):
        return f"{item} 限時補貨，燃脂必備！"
    return f"{item} 熱銷中，把握優惠！"


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/upload_face")
def upload_face():
    """ESP32-CAM / 手動上傳：回傳 member_id 與廣告頁 URL。"""
    overall_start = perf_counter()
    try:
        image_bytes, mime_type = _extract_image_payload(request)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    upload_duration = perf_counter() - overall_start

    # encode & match
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

    # for record
    image_filename = _persist_upload_image(member_id, image_bytes, mime_type)
    database.record_upload_event(
        member_id=member_id,
        image_filename=image_filename,
        upload_duration=upload_duration,
        recognition_duration=recognition_duration,
        ad_duration=0.0,
        total_duration=perf_counter() - overall_start,
    )
    _purge_upload_images(database.cleanup_upload_events(keep_latest=1))

    purchases = database.get_purchase_history(member_id)
    insights = analyse_purchase_intent(purchases, new_member=new_member)
    profile = database.get_member_profile(member_id)
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
        {"status": "ok", "merged_member": source_id, "into": target_id, "deleted_cloud_faces": deleted_faces, "encoding_updated": encoding_updated}
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

    context = build_ad_context(member_id, purchases, insights=insights, profile=profile, creative=creative)
    hero_image_url = _resolve_hero_image_url(context.scenario_key)

    return render_template("ad.html", context=context, hero_image_url=hero_image_url, scenario_key=context.scenario_key)


@app.get("/ad-image/<member_id>")
def ad_image(member_id: str):
    """回傳『已把文案疊字』後的 JPEG 圖片。"""
    purchases = database.get_purchase_history(member_id)
    insights = analyse_purchase_intent(purchases)
    profile = database.get_member_profile(member_id)
    scenario_key = derive_scenario_key(insights, profile=profile)

    # 找到底圖實體檔案
    filename = AD_IMAGE_BY_SCENARIO.get(scenario_key) or AD_IMAGE_BY_SCENARIO.get("brand_new")
    if not filename:
        abort(404)

    base_path = None
    ads_dir = current_app.config.get("ADS_DIR") or ""
    if ads_dir and (Path(ads_dir) / filename).is_file():
        base_path = str(Path(ads_dir) / filename)
    else:
        # fallback 到內建靜態圖
        static_candidate = STATIC_DIR / "images" / "ads" / filename
        if static_candidate.is_file():
            base_path = str(static_candidate)

    if not base_path or not os.path.isfile(base_path):
        abort(404)

    # 產生 20 字內文案（若 Gemini 可用也能改為使用 LLM）
    copy = _generate_short_copy(insights, profile)

    # 開啟圖片並疊字
    img = Image.open(base_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    font_path = _find_cjk_font()
    # 字級與邊距（依寬度比例）
    fs = max(28, int(W * 0.035))
    font = ImageFont.truetype(font_path, fs) if font_path else ImageFont.load_default()

    # 文案換行（最多兩行）
    max_width = int(W * 0.78)
    lines = []
    for line in textwrap.wrap(copy, width=12):  # 先粗略切
        if draw.textlength(line, font=font) <= max_width:
            lines.append(line)
        else:
            # 若仍過長再二分
            half = len(line) // 2
            lines.extend([line[:half], line[half:]])

    lines = lines[:2]
    text_h = sum(draw.textbbox((0, 0), ln, font=font)[3] - draw.textbbox((0, 0), ln, font=font)[1] for ln in lines) + fs // 2
    pad = int(fs * 0.7)
    box_w = max(draw.textlength(ln, font=font) for ln in lines) + pad * 2
    box_h = text_h + pad * 2

    # 位置：靠下方 12% 處置中
    x0 = (W - box_w) // 2
    y0 = H - int(H * 0.12) - box_h

    # 半透明底
    draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=int(fs * 0.6), fill=(0, 0, 0, 180), outline=(255, 255, 255), width=2)

    # 白字描邊
    y = y0 + pad
    for ln in lines:
        tw = draw.textlength(ln, font=font)
        x = x0 + (box_w - tw) / 2
        # 描邊
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), ln, font=font, fill=(0, 0, 0))
        draw.text((x, y), ln, font=font, fill=(255, 255, 255))
        y += (draw.textbbox((0, 0), ln, font=font)[3] - draw.textbbox((0, 0), ln, font=font)[1])

    # 回傳 JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg", as_attachment=False, download_name=f"{member_id}.jpg")


@app.get("/latest_upload")
def latest_upload_dashboard():
    event = database.get_latest_upload_event()
    if event is None:
        return render_template("latest_upload.html", event=None, members_url=url_for("member_directory"))
    image_url = url_for("serve_upload_image", filename=event.image_filename) if event.image_filename else None
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


@app.route("/manager", methods=["GET", "POST"])
def manager_dashboard_view():
    """GET：帶 member_id 顯示；POST：直接上傳圖片取得 member_id 後 303 轉跳。"""
    if request.method == "POST":
        try:
            image_bytes, mime_type = _extract_image_payload(request)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

        try:
            encoding = recognizer.encode(image_bytes, mime_type=mime_type)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 422

        member_id, _ = database.find_member_by_encoding(encoding, recognizer)
        if member_id is None:
            member_id = database.create_member(encoding, recognizer.derive_member_id(encoding))
            _create_welcome_purchase(member_id)
        # 保存上傳影像（可於 UI 顯示）
        _persist_upload_image(member_id, image_bytes, mime_type)
        # 303 讓ブラウザ改 GET
        return redirect(url_for("manager_dashboard_view", member_id=member_id), code=303)

    # GET
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
        prediction = predict_next_purchases(purchases, profile=profile, insights=insights, limit=7)
        context = {
            "member_id": selected_id,
            "member": profile,
            "analysis": insights,
            "scenario_key": scenario_key,
            "hero_image_url": hero_image_url,
            "prediction": prediction,
            "ad_url": url_for("render_ad", member_id=selected_id, v2=1, _external=False),
            "ad_image_url": url_for("ad_image", member_id=selected_id),
        }
    else:
        error = "尚未建立任何會員資料"

    return render_template("manager.html", context=context, member_id=member_id, members=selectable_members, error=error)


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


@app.get("/ad-preview/<path:filename>")
def ad_preview(filename: str):
    hero_image_url = url_for("serve_ad_asset", filename=filename)
    return render_template("ad.html", hero_image_url=hero_image_url, scenario_key=request.args.get("scenario_key", "brand_new"))


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return jsonify(_health_payload())

@app.get("/healthz")
def healthz_check():
    return jsonify(_health_payload())
# backend/app.py（新增段落）
from flask import jsonify, render_template, request
from backend.reco import recommend_for_member
from backend.ai_gemini import ad_copy_unregistered, ad_copy_registered

def _pick_image_for(status: str, dom_cat: str | None) -> str:
    """
    依需求選圖：
    - new: 固定 ME0000.jpg
    - unregistered: fitness->AD0000.jpg, homemaker->AD0001.jpg
      homemaker 對應「日用品/食物」偏好
    - registered: fitness->ME0003.jpg, dessert->ME0001.jpg, kindergarten->ME0002.jpg
      kindergarten 對應「幼兒用品」
    其他類別fall back：ME0000.jpg
    """
    if status == "new":
        return "ME0000.jpg"
    if status == "unregistered":
        if dom_cat in ("健身", "保健食品"):
            return "AD0000.jpg"
        if dom_cat in ("日用品", "食物"):
            return "AD0001.jpg"
        return "AD0001.jpg"
    # registered
    if dom_cat in ("健身", "保健食品"):
        return "ME0003.jpg"
    if dom_cat == "甜點":
        return "ME0001.jpg"
    if dom_cat == "幼兒用品":
        return "ME0002.jpg"
    return "ME0000.jpg"

@app.get("/recommendations/<member_id>")
def api_recommendations(member_id: str):
    """
    回傳七筆推估 + 機率百分比；同時附上廣告文案與建議圖檔。
    """
    data = recommend_for_member(member_id)
    status = data["status"]
    dom_cat = data["dominant_category"]

    image = _pick_image_for(status, dom_cat)

    # 文案
    ad = {"headline": "", "subline": "", "image": image}
    if status == "new":
        ad["headline"] = "歡迎光臨！加入會員享驚喜禮"
    elif status == "unregistered":
        top = data["top7"][0] if data["top7"] else None
        if top:
            ad["headline"] = ad_copy_unregistered(top["product_name"], top["category"])
        else:
            ad["headline"] = "加入會員，領限定試用包"
    else:  # registered
        top = data["top7"][0] if data["top7"] else None
        if top:
            h, s = ad_copy_registered(top["product_name"], top["category"])
            ad["headline"], ad["subline"] = h, s
        else:
            ad["headline"] = "會員專屬驚喜優惠"

    return jsonify({
        "member": data["member"],
        "status": status,
        "period": data["period"],
        "dominant_category": dom_cat,
        "top7": data["top7"],
        "ad": ad
    })

@app.get("/manager/reco")
def page_manager_reco():
    """
    後台經理驗證頁（100% 依你給的版型配置）。
    使用方式：
      /manager/reco?member_id=U001
      /manager/reco   -> 預設U001
    """
    member_id = request.args.get("member_id") or "U001"
    data = recommend_for_member(member_id)
    status = data["status"]
    dom_cat = data["dominant_category"]
    image = _pick_image_for(status, dom_cat)

    ad_headline, ad_subline = "", ""
    if status == "new":
        ad_headline = "歡迎光臨！加入會員享驚喜禮"
    elif status == "unregistered":
        top = data["top7"][0] if data["top7"] else None
        ad_headline = ad_copy_unregistered(top["product_name"], top["category"]) if top else "加入會員，領限定試用包"
    else:
        top = data["top7"][0] if data["top7"] else None
        if top:
            ad_headline, ad_subline = ad_copy_registered(top["product_name"], top["category"])
        else:
            ad_headline = "會員專屬驚喜優惠"

    return render_template(
        "manager_reco.html",
        data=data,
        image=image,
        ad_headline=ad_headline,
        ad_subline=ad_subline
    )

if __name__ == "__main__":
    # 開發模式直接啟動；部署請用 gunicorn / systemd 並確保帶入 ADS_DIR
    app.run(host="0.0.0.0", port=8000, debug=True)
