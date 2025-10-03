"""Business logic to transform database rows into advertising copy."""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Literal, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .ai import AdCreative, GeminiService, GeminiUnavailableError
from .database import MemberProfile, Purchase

# Persona → 商品/情境「細分鍵」對應
# （把 home-manager 對到 homemaker；wellness-gourmet 併到 fitness，確保有對應圖片）
PROFILE_SEGMENT_BY_LABEL: dict[str, str] = {
    "dessert-lover": "dessert",
    "family-groceries": "kindergarten",
    "fitness-enthusiast": "fitness",
    "home-manager": "homemaker",
    "wellness-gourmet": "fitness",
}

# 廣告主視覺檔名對應表（由 app.py 的 _resolve_hero_image_url 使用）
AD_IMAGE_BY_SCENARIO: dict[str, str] = {
    # 1) 全新顧客（資料庫完全無歷史） → 固定文案圖
    "brand_new": "ME0000.jpg",

    # 2) 已註冊會員（依商品類別）
    "registered:dessert": "ME0001.jpg",
    "registered:kindergarten": "ME0002.jpg",
    "registered:fitness": "ME0003.jpg",

    # （保險）如果流程產生 repeat_purchase:<cat> 也能對到相同圖
    "repeat_purchase:dessert": "ME0001.jpg",
    "repeat_purchase:kindergarten": "ME0002.jpg",
    "repeat_purchase:fitness": "ME0003.jpg",
    "repeat_purchase": "ME0003.jpg",  # 無類別時的備援

    # 3) 未註冊會員但有明顯偏好（引導入會）
    "unregistered:fitness": "AD0000.jpg",
    "unregistered:homemaker": "AD0001.jpg",
}

@dataclass
class AdCopy:
    title: str
    subtitle: str
    bullets: List[str]


FALLBACK_COPY = AdCopy(
    title="限時優惠",
    subtitle="會員專屬禮遇",
    bullets=["今日下單再享驚喜"],
)


_LOGGER = logging.getLogger(__name__)


SCENARIO_ASSETS: dict[str, dict[str, str]] = {
    "ME0001": {"scenario": "dessert", "label": "甜點", "image": "ME0001.jpg"},
    "ME0002": {"scenario": "kindergarten", "label": "幼兒園", "image": "ME0002.jpg"},
    "ME0003": {"scenario": "fitness", "label": "健身房", "image": "ME0003.jpg"},
}


def _resolve_asset_config(member_id: str) -> dict[str, str]:
    key = member_id[:6]
    if key in SCENARIO_ASSETS:
        return SCENARIO_ASSETS[key]
    raise KeyError(f"No scenario asset mapping for member_id={member_id}")


@dataclass
class PurchaseInsights:
    """Summarised shopping intent derived from historical purchases."""
    scenario: Literal["brand_new", "repeat_purchase", "returning_member"]
    recommended_item: str | None
    probability: float
    repeat_count: int
    total_purchases: int

    @property
    def probability_percent(self) -> int:
        """Return a user-friendly percentage for template rendering."""
        if self.probability <= 0:
            return 62
        return max(45, min(96, round(self.probability * 100)))

@dataclass
class AdContext:
    member_id: str
    member_code: str
    headline: str
    subheading: str
    highlight: str
    purchases: list[Purchase]
    insights: PurchaseInsights
    scenario_key: str

def analyse_purchase_intent(
    purchases: Iterable[Purchase], *, new_member: bool = False
) -> PurchaseInsights:
    """Estimate what should be promoted based on the shopper's history."""
    purchase_list = list(purchases)
    total_orders = len(purchase_list)

    # 把歡迎禮視為「尚無真實消費史」，保留新客 onboarding 體驗
    if purchase_list and total_orders == 1:
        only_item = purchase_list[0].item.strip()
        if only_item == "歡迎禮盒":
            new_member = True

    if new_member or not purchase_list:
        return PurchaseInsights(
            scenario="brand_new",
            recommended_item=None,
            probability=0.0,
            repeat_count=0,
            total_purchases=total_orders,
        )

    frequency = Counter()
    weighted_score: dict[str, float] = defaultdict(float)
    total_weight = 0.0

    for index, purchase in enumerate(purchase_list):
        item = purchase.item.strip()
        # 越新的權重略高（簡單遞減）
        weight = 1.0 + max(0, 10 - index) * 0.05
        total_weight += weight
        frequency[item] += 1
        weighted_score[item] += weight

    top_item, top_weight = max(weighted_score.items(), key=lambda entry: entry[1])
    probability = top_weight / total_weight if total_weight else 0.0
    repeat_item, repeat_count = frequency.most_common(1)[0]

    if repeat_count >= 2:
        return PurchaseInsights(
            scenario="repeat_purchase",
            recommended_item=repeat_item,
            probability=probability,
            repeat_count=repeat_count,
            total_purchases=total_orders,
        )

    return PurchaseInsights(
        scenario="returning_member",
        recommended_item=top_item,
        probability=probability,
        repeat_count=repeat_count,
        total_purchases=total_orders,
    )

def derive_scenario_key(
    insights: PurchaseInsights, *, profile: MemberProfile | None = None
) -> str:
    """Convert insights and persona data into a marketing scenario key."""
    if insights.scenario == "brand_new":
        return "brand_new"

    profile_label = getattr(profile, "profile_label", "")
    segment = PROFILE_SEGMENT_BY_LABEL.get(profile_label, "")
    registered = bool(getattr(profile, "mall_member_id", ""))

    if segment:
        prefix = "registered" if registered else "unregistered"
        scenario_key = f"{prefix}:{segment}"
        if scenario_key in AD_IMAGE_BY_SCENARIO:
            return scenario_key

    if insights.scenario.startswith("repeat_purchase"):
        if segment:
            rp_key = f"repeat_purchase:{segment}"
            if rp_key in AD_IMAGE_BY_SCENARIO:
                return rp_key
        if "repeat_purchase" in AD_IMAGE_BY_SCENARIO:
            return "repeat_purchase"

    return "brand_new"

def build_ad_context(
    member_id: str,
    purchases: Iterable[Purchase],
    *,
    insights: PurchaseInsights | None = None,
    profile: MemberProfile | None = None,
    creative: AdCreative | None = None,
) -> AdContext:
    purchase_list = list(purchases)
    insights = insights or analyse_purchase_intent(purchase_list)
    member_code = purchase_list[0].member_code if purchase_list else ""
    greeting = _member_salutation(member_code)
    subheading_code = _subheading_prefix(member_code)
    scenario_key = derive_scenario_key(insights, profile=profile)

    fallback_headline, fallback_subheading, fallback_highlight = _fallback_copy(
        greeting, subheading_code, purchase_list, insights
    )

    if creative:
        headline = creative.headline or fallback_headline
        subheading = creative.subheading or fallback_subheading
        highlight = creative.highlight or fallback_highlight
    else:
        headline = fallback_headline
        subheading = fallback_subheading
        highlight = fallback_highlight

    return AdContext(
        member_id=member_id,
        member_code=member_code,
        headline=headline,
        subheading=subheading,
        highlight=highlight,
        purchases=purchase_list,
        insights=insights,
        scenario_key=scenario_key,
    )

def _format_quantity(quantity: float) -> str:
    if quantity.is_integer():
        return str(int(quantity))
    return f"{quantity:.1f}"

def _fallback_copy(
    greeting: str,
    subheading_code: str,
    purchases: list[Purchase],
    insights: PurchaseInsights,
) -> tuple[str, str, str]:
    latest_summary = _recent_purchase_summary(purchases)

    if insights.scenario == "brand_new":
        headline = f"{greeting}，歡迎加入！"
        subheading = (
            f"{subheading_code}第一次到店，立即加入會員解鎖紅利點數、生日禮與本週專屬折扣"
        )
        highlight = "掃描服務台 QR Code 馬上入會，今日完成註冊送咖啡招待與 120 點開卡禮！"
        return headline, subheading, highlight

    if insights.scenario == "repeat_purchase":
        item = insights.recommended_item or (purchases[0].item if purchases else "人氣商品")
        headline = f"{greeting}，{item} 回購加碼！"
        subheading = f"{subheading_code}最近 {insights.repeat_count} 次都選擇了 {item}"
        if latest_summary:
            subheading += f"｜上次 {latest_summary}"
        highlight = f"{item} 會員限定：第 {insights.repeat_count + 1} 件 82 折，再贈職人限定隨行包！"
        return headline, subheading, highlight

    item = insights.recommended_item or (purchases[0].item if purchases else "人氣商品")
    probability_text = _format_probability(insights.probability)
    headline = f"{greeting}，預留了你的 {item}"
    subheading = f"{subheading_code}系統預測你對 {item} 的購買機率高達 {probability_text}"
    if latest_summary:
        subheading += f"｜上次 {latest_summary}"
    highlight = f"{item} 今日限量再享會員專屬 88 折，結帳輸入 MEMBER95 加贈點數！"
    return headline, subheading, highlight

def _recent_purchase_summary(purchases: list[Purchase]) -> str | None:
    if not purchases:
        return None
    latest = purchases[0]
    return (
        f"{latest.purchased_at}｜{latest.item}｜${latest.total_price:,.0f}"
        f"（{_format_quantity(latest.quantity)} 件）"
    )

def _format_probability(probability: float) -> str:
    if probability <= 0:
        return "62%"
    percentage = round(probability * 100)
    percentage = max(45, min(96, percentage))
    return f"{percentage}%"

def _member_salutation(member_code: str) -> str:
    if member_code:
        return f"會員 {member_code}"
    return "親愛的貴賓"

def _subheading_prefix(member_code: str) -> str:
    if member_code:
        return f"商場會員代號：{member_code}｜"
    return "尚未綁定商場會員，立即至服務台完成綁定享專屬禮遇｜"


def get_member_scenario(member_id: str) -> tuple[str, str]:
    """Return (scenario_key, scenario_label) for the provided member ID."""

    try:
        config = _resolve_asset_config(member_id)
    except KeyError as exc:  # pragma: no cover - defensive guard for future IDs
        raise FileNotFoundError(f"Unable to resolve scenario for {member_id}") from exc
    return config["scenario"], config["label"]


def member_to_base_image(member_id: str) -> str:
    """依 member_id 決定底圖路徑"""

    base_dir = Path(os.environ.get("AD_BASE_DIR", "/srv/esp32-ads"))
    config = _resolve_asset_config(member_id)
    image_path = base_dir / config["image"]
    return str(image_path)


def _default_font_path() -> str:
    return os.environ.get(
        "FONT_PATH", "/usr/share/fonts/truetype/noto/NotoSansTC-Regular.otf"
    )


def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    tried: list[str] = []
    candidates = [font_path, _default_font_path(), "/usr/share/fonts/truetype/noto/NotoSansTC-Bold.otf"]
    fallback = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            tried.append(str(path))
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            tried.append(str(path))
            continue
    try:
        return ImageFont.truetype(fallback, size=size)
    except OSError:
        _LOGGER.warning("Unable to load preferred fonts (%s); falling back to default.", ", ".join(tried))
        return ImageFont.load_default()


def _normalise_bullets(values: Iterable[str]) -> list[str]:
    bullets: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        bullets.append(text)
        if len(bullets) == 3:
            break
    if not bullets:
        return FALLBACK_COPY.bullets
    return bullets


def _build_prompt(
    member_id: str,
    scenario_label: str,
    top_predictions: list[dict[str, object]],
    recent_summary: str,
) -> str:
    if top_predictions:
        lines = []
        for item in top_predictions:
            name = str(item.get("item") or item.get("product_name") or "推薦商品")
            price = item.get("price")
            probability = item.get("probability_percent") or item.get("probability")
            if isinstance(probability, float):
                probability = round(probability * 100, 1)
            if isinstance(price, (int, float)):
                price_text = f"NT${price:,.0f}"
            else:
                price_text = str(price or "-")
            lines.append(f"- {name}｜預估價格 {price_text}｜購買機率 {probability}%")
        top_block = "\n".join(lines)
    else:
        top_block = "- 暫無預測商品，請聚焦會員開卡禮"

    prompt = (
        "你是零售行銷文案專家。請根據「會員輪廓與預估商品」產生 1 組廣告文案：\n"
        f"- 語氣：溫暖、直接，貼合{scenario_label}場景。\n"
        "- 產出 JSON 物件，鍵包含：title, subtitle, bullets（陣列，長度 1~3，單行不超過 20 個全形字）。\n\n"
        "【會員資訊】\n"
        f"- 會員ID：{member_id}\n"
        f"- 情境：{scenario_label}\n"
        f"- 過去購買摘要：{recent_summary}\n\n"
        "【預估推薦 TopN】\n"
        f"{top_block}\n\n"
        "請只輸出 JSON，勿加說明文字。"
    )
    return prompt


def _parse_ad_copy_payload(text: str) -> AdCopy:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise GeminiUnavailableError(f"Gemini 回傳格式錯誤：{exc}") from exc

    if not isinstance(payload, dict):
        raise GeminiUnavailableError("Gemini 回傳內容不是 JSON 物件")

    title = str(payload.get("title") or "").strip()
    subtitle = str(payload.get("subtitle") or "").strip()
    bullets_raw = payload.get("bullets")
    if isinstance(bullets_raw, str):
        bullets = _normalise_bullets([bullets_raw])
    elif isinstance(bullets_raw, Iterable):
        bullets = _normalise_bullets(bullets_raw)
    else:
        bullets = FALLBACK_COPY.bullets

    if not title:
        title = FALLBACK_COPY.title
    if not subtitle:
        subtitle = FALLBACK_COPY.subtitle

    return AdCopy(title=title, subtitle=subtitle, bullets=bullets)


def _get_gemini_service() -> GeminiService:
    text_model = os.environ.get("GEMINI_TEXT_MODEL")
    vision_model = os.environ.get("GEMINI_VISION_MODEL")
    return GeminiService(text_model=text_model, vision_model=vision_model)


def generate_ad_copy_via_gemini(
    member_id: str,
    scenario_label: str,
    top_predictions: list,
    recent_summary: str,
    svc: Optional[GeminiService] = None,
) -> AdCopy:
    """呼叫 Gemini 產生文案，回傳 AdCopy 結構"""

    service = svc or _get_gemini_service()
    if not getattr(service, "can_generate_ads", False):
        _LOGGER.warning("Gemini service unavailable; using fallback copy")
        return FALLBACK_COPY

    prompt = _build_prompt(member_id, scenario_label, top_predictions, recent_summary)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            model = getattr(service, "_text_model", None)
            if model is None:
                raise GeminiUnavailableError("Gemini 文字模型未初始化")
            response = model.generate_content(
                prompt,
                request_options={"timeout": getattr(service, "_timeout", 20.0)},
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise GeminiUnavailableError("Gemini 回傳為空白")
            return _parse_ad_copy_payload(text)
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            _LOGGER.warning("Gemini generate attempt %s failed: %s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(0.6 * (2**attempt))

    _LOGGER.error("Gemini generation failed after retries: %s", last_error)
    return FALLBACK_COPY


def _draw_rounded_rectangle(draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int], radius: int, fill: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(bbox, radius=radius, fill=fill)


def _ensure_output_dir(out_dir: str) -> Path:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output


def compose_final_image(
    base_image_path: str,
    copy: AdCopy,
    out_dir: str = "/srv/esp32-ads/out",
    font_path: Optional[str] = None,
) -> str:
    """用 Pillow 將文案合成到底圖上，回傳輸出檔路徑"""

    base_path = Path(base_image_path)
    if not base_path.exists():
        raise FileNotFoundError(f"底圖不存在：{base_image_path}")

    output_dir = _ensure_output_dir(out_dir)
    resampling_attr = getattr(Image, "Resampling", None)
    if resampling_attr is not None:
        resample_filter = getattr(resampling_attr, "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))
    else:
        resample_filter = getattr(Image, "LANCZOS", Image.BICUBIC)

    with Image.open(base_path) as img:
        canvas = ImageOps.fit(img.convert("RGBA"), (1920, 1080), method=resample_filter)

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    background_alpha = int(255 * 0.4)
    _draw_rounded_rectangle(overlay_draw, (80, 80, 1840, 360), 36, (0, 0, 0, background_alpha))
    _draw_rounded_rectangle(overlay_draw, (80, 300, 1840, 500), 30, (0, 0, 0, background_alpha))
    _draw_rounded_rectangle(overlay_draw, (80, 480, 1840, 820), 30, (0, 0, 0, background_alpha))

    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)

    title_font = _load_font(font_path, 72)
    subtitle_font = _load_font(font_path, 44)
    bullet_font = _load_font(font_path, 40)

    draw.text((100, 120), copy.title, font=title_font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    draw.text((100, 330), copy.subtitle, font=subtitle_font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))

    bullets = _normalise_bullets(copy.bullets)
    for index, bullet in enumerate(bullets):
        y = 520 + index * 64
        draw.text((100, y), f"• {bullet}", font=bullet_font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    member_id = Path(base_image_path).stem.split("_")[0]
    filename = f"{member_id}_{timestamp}.jpg"
    out_path = output_dir / filename
    canvas.convert("RGB").save(out_path, format="JPEG", quality=92)
    return str(out_path)
