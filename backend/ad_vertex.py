# backend/ad_vertex.py
from __future__ import annotations
import os, json, textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from PIL import Image, ImageDraw, ImageFont

from .database import Database
from .ai import GeminiService

IMG_W, IMG_H = 1920, 1080
SAFE_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansTC-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansTC-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def _font(size:int):
    for p in SAFE_FONTS:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size=size)
            except Exception: pass
    return ImageFont.load_default()

def _wrap(draw: ImageDraw.ImageDraw, font, text: str, max_px: int) -> list[str]:
    # 粗略用字符寬比做換行
    max_chars = max(8, int(max_px / (font.size * 0.9)))
    return textwrap.wrap(text or "", width=max_chars)

def _pick_top_item(db: Database, member_id: str) -> Tuple[str, Optional[float]]:
    row = db.fetch_one("""
        SELECT item, probability
        FROM reco_purchases
        WHERE member_id=? ORDER BY probability DESC, item ASC LIMIT 1
    """, (member_id,))
    if row and row["item"]: return row["item"], row.get("probability")
    row = db.fetch_one("""
        SELECT item, COUNT(*) AS cnt
        FROM purchases WHERE member_id=?
        GROUP BY item ORDER BY cnt DESC, item ASC LIMIT 1
    """, (member_id,))
    return (row["item"], None) if row and row["item"] else ("拿鐵咖啡", None)

def _generate_copy(member_id: str, item: str) -> Dict:
    try:
        svc = GeminiService()
        ad = svc.generate_ad_copy(member_id=member_id, purchases=[{"item": item}])
        return {"headline": ad.headline, "subheading": ad.subheading, "highlight": ad.highlight}
    except Exception:
        return {
            "headline": f"{item} 限時加碼",
            "subheading": "會員專屬｜48 小時快閃",
            "highlight": [f"本週人氣品：{item}", "第二件 7 折", "滿 299 再折 30"]
        }

def _imagen_generate_base(project: str, region: str, prompt: str):
    vertexai.init(project=project, location=region)
    model_name = os.getenv("IMAGE_MODEL", "imagen-3.0-generate-002")  # 可換回 001
    model = ImageGenerationModel.from_pretrained(model_name)
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="16:9",
        safety_filter_level="block_some",
        person_generation="dont_allow",  # 廣告以產品為主，降低觸發風險
        # add_watermark 預設為啟用；若要 reproducible，可改用 seed（需關閉水印）
    )
    # 官方範例支援 .save(...)；這裡直接回傳 bytes
    return images[0]._image_bytes  # 來源於官方 SDK 回傳對象

def _compose(final_bg: bytes, copy: Dict, out_path: Path):
    # 以生成圖為底，疊暗條＋文字
    from io import BytesIO
    base = Image.open(BytesIO(final_bg)).convert("RGB").resize((IMG_W, IMG_H))
    draw = ImageDraw.Draw(base, "RGBA")
    bar_h = int(IMG_H * 0.34)
    draw.rectangle([(0, IMG_H - bar_h), (IMG_W, IMG_H)], fill=(0, 0, 0, 168))

    f_h1, f_h2, f_li = _font(92), _font(52), _font(42)
    x, y = int(IMG_W * 0.06), IMG_H - bar_h + int(IMG_H * 0.04)
    wrap_w = int(IMG_W * 0.88)

    def put(text: str, font, lh_mul=1.25):
        nonlocal y
        for line in _wrap(draw, font, text, wrap_w):
            draw.text((x, y), line, font=font, fill=(255,255,255))
            y += int(font.size * lh_mul)

    put(copy.get("headline",""), f_h1, 1.25)
    put(copy.get("subheading",""), f_h2, 1.3)
    for bullet in (copy.get("highlight") or [])[:3]:
        put("• " + bullet, f_li, 1.35)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(out_path, "JPEG", quality=92, optimize=True, progressive=True)

def generate_ad_with_vertex(member_id: str, ads_dir: Path) -> Dict:
    """主流程：挑商品 → 文字 → Imagen 生底圖 → 疊字 → 存檔 + JSON"""
    db = Database()
    item, prob = _pick_top_item(db, member_id)
    copy = _generate_copy(member_id, item)

    project = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
    region = os.getenv("VERTEX_REGION", "us-central1")
    if not project:
        raise RuntimeError("VERTEX_PROJECT_ID/PROJECT_ID 未設定")

    # 提示詞：聚焦「產品情境圖」，避免生成文字（文字由我們疊）
    prompt = f"""
    A clean, modern e-commerce product hero photo for: {item}.
    Shot for a 16:9 digital signage banner, ample safe space on the bottom third for overlay text,
    high-key lighting, soft shadows, no people, photorealistic, commercial quality.
    """
    img_bytes = _imagen_generate_base(project, region, prompt)

    image_name = f"AD-{member_id}.jpg"
    out_img = ads_dir / image_name
    _compose(img_bytes, copy, out_img)

    meta = {
        "member_id": member_id,
        "item": item,
        "probability": prob,
        **copy
    }
    (ads_dir / f"AD-{member_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"image_name": image_name, **meta}
