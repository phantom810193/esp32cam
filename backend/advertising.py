"""Business logic to transform database rows into advertisement board payloads."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from secrets import token_hex
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

from .ai import AdCreative
from .database import MemberProfile, NEW_GUEST_MEMBER_ID, Purchase

PROFILE_SEGMENT_BY_LABEL: dict[str, str] = {
    "dessert-lover": "dessert",
    "family-groceries": "kindergarten",
    "fitness-enthusiast": "fitness",
    "home-manager": "homemaker",
    "wellness-gourmet": "fitness",
}

TEMPLATE_IMAGE_BY_ID: dict[str, str] = {
    "ME0000": "ME0000.jpg",
    "ME0001": "ME0001.jpg",
    "ME0002": "ME0002.jpg",
    "ME0003": "ME0003.jpg",
}

CATEGORY_TEMPLATE_RULES: dict[str, set[str]] = {
    "ME0001": {"dessert", "sweet", "bakery", "breakfast"},
    "ME0002": {"kindergarten", "kids", "family", "parent"},
    "ME0003": {"fitness", "wellness", "home", "general", "beauty"},
}

DEFAULT_TEMPLATE_ID = "ME0003"
CTA_JOIN_MEMBER = "立即加入會員解鎖專屬禮遇"
CTA_MEMBER_OFFER = "會員限定優惠立即領取"
CTA_DISCOVER = "立即了解活動"


def _random_line(options: Sequence[str], fallback: str) -> str:
    pool = [option for option in options if option]
    if not pool:
        pool = [fallback]
    return random.choice(pool)


def _unique_highlight(options: Sequence[str], fallback: str) -> str:
    base = _random_line(options, fallback)
    code = token_hex(2).upper()
    if "{code}" in base:
        return base.format(code=code)
    return f"{base}｜限時代碼 {code}"


def _parse_purchase_timestamp(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now()


def _build_purchase_months(purchases: Sequence[Purchase]) -> list[dict[str, Any]]:
    if not purchases:
        return []

    buckets: defaultdict[tuple[int, int], list[tuple[datetime, Purchase]]] = defaultdict(list)
    for purchase in purchases:
        timestamp = _parse_purchase_timestamp(purchase.purchased_at)
        buckets[(timestamp.year, timestamp.month)].append((timestamp, purchase))

    ordered = sorted(buckets.items(), key=lambda entry: entry[0], reverse=True)
    months: list[dict[str, Any]] = []
    for (year, month), entries in ordered[:2]:
        entries.sort(key=lambda pair: pair[0], reverse=True)
        months.append(
            {
                "label": f"{year}年{month:02d}月",
                "year": year,
                "month": month,
                "purchases": [purchase for _, purchase in entries],
            }
        )
    return months



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
    """Final payload consumed by the kiosk front-end."""

    member_id: str
    member_code: str
    headline: str
    subheading: str
    highlight: str
    template_id: str
    audience: Literal["new", "guest", "member"]
    purchases: list[Purchase]
    insights: PurchaseInsights | None
    predicted: Mapping[str, Any] | None = None
    predicted_candidates: list[Mapping[str, Any]] = field(default_factory=list)
    cta_text: str | None = None
    cta_href: str | None = None
    scenario_key: str = "brand_new"
    profile: Mapping[str, Any] | None = None
    timings: Mapping[str, Any] | None = None
    detected_at: str | None = None
    purchase_months: list[dict[str, Any]] = field(default_factory=list)


def analyse_purchase_intent(
    purchases: Iterable[Purchase], *, new_member: bool = False
) -> PurchaseInsights:
    """Estimate what should be promoted based on the shopper's history."""
    purchase_list = list(purchases)
    total_orders = len(purchase_list)

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
    insights: PurchaseInsights,
    *,
    profile: MemberProfile | None = None,
    template_id: str | None = None,
    audience: str | None = None,
) -> str:
    """Convert insights and persona data into a marketing scenario key."""
    if template_id:
        prefix = audience or ("member" if getattr(profile, "mall_member_id", "") else "guest")
        return f"{prefix}:{template_id}"

    if insights.scenario == "brand_new":
        return "brand_new"

    profile_label = getattr(profile, "profile_label", "")
    segment = PROFILE_SEGMENT_BY_LABEL.get(profile_label, "")
    registered = bool(getattr(profile, "mall_member_id", ""))

    if segment:
        prefix = "registered" if registered else "unregistered"
        return f"{prefix}:{segment}"

    return "brand_new"


def build_ad_context(
    member_id: str,
    purchases: Iterable[Purchase],
    *,
    insights: PurchaseInsights | None = None,
    profile: MemberProfile | None = None,
    profile_snapshot: Mapping[str, Any] | None = None,
    creative: AdCreative | None = None,
    predicted_item: Mapping[str, Any] | None = None,
    prediction_items: Sequence[Mapping[str, Any] | Any] | None = None,
    audience: Literal["new", "guest", "member"] = "guest",
    timings: Mapping[str, Any] | None = None,
    detected_at: str | None = None,
    cta_override: str | None = None,
) -> AdContext:
    purchase_list = list(purchases)
    insights = insights or analyse_purchase_intent(purchase_list)

    member_code = ""
    if purchase_list:
        member_code = purchase_list[0].member_code
    elif profile and profile.mall_member_id:
        member_code = profile.mall_member_id

    normalized_prediction = _normalise_prediction(predicted_item)
    candidate_list = _normalise_prediction_list(prediction_items)

    template_id = _select_template_id(audience, normalized_prediction)
    scenario_key = derive_scenario_key(
        insights,
        profile=profile,
        template_id=template_id,
        audience=audience,
    )

    fallback_headline, fallback_subheading, fallback_highlight, fallback_cta = _fallback_copy(
        audience,
        normalized_prediction,
    )

    if creative:
        headline = creative.headline or fallback_headline
        subheading = creative.subheading or fallback_subheading
        highlight = creative.highlight or fallback_highlight
        cta_text = creative.cta or fallback_cta
    else:
        headline = fallback_headline
        subheading = fallback_subheading
        highlight = fallback_highlight
        cta_text = fallback_cta

    if cta_override:
        cta_text = cta_override

    if audience == "new":
        register_image = TEMPLATE_IMAGE_BY_ID.get("ME0000", "ME0000.jpg")
        cta_href = f"/static/images/ads/{register_image}"
        if cta_text is None:
            cta_text = CTA_JOIN_MEMBER
    elif audience == "guest":
        if member_id and member_id != NEW_GUEST_MEMBER_ID:
            cta_href = f"/ad/{member_id}"
        else:
            image_key = template_id if template_id in TEMPLATE_IMAGE_BY_ID else None
            if not image_key:
                image_key = "ME0003"
            cta_href = f"/static/images/ads/{TEMPLATE_IMAGE_BY_ID.get(image_key, TEMPLATE_IMAGE_BY_ID['ME0003'])}"
        if cta_text is None:
            cta_text = CTA_JOIN_MEMBER
    else:

        fallback_ad_link = f"/ad/{member_id}/offer" if member_id else "#member-offer"

        cta_href = fallback_ad_link
        if cta_text is None:
            cta_text = CTA_MEMBER_OFFER

    profile_dict = profile_snapshot or _profile_to_dict(profile)

    if member_id == NEW_GUEST_MEMBER_ID:
        cta_href = f"/static/images/ads/{TEMPLATE_IMAGE_BY_ID.get('ME0000', 'ME0000.jpg')}"
        if cta_text is None:
            cta_text = CTA_JOIN_MEMBER

    purchase_months = _build_purchase_months(purchase_list)

    return AdContext(
        member_id=member_id,
        member_code=member_code,
        headline=headline,
        subheading=subheading,
        highlight=highlight,
        template_id=template_id,
        audience=audience,
        purchases=purchase_list,
        insights=insights,
        predicted=normalized_prediction,
        predicted_candidates=candidate_list,
        cta_text=cta_text,
        cta_href=cta_href,
        scenario_key=scenario_key,
        profile=profile_dict,
        timings=timings,
        detected_at=detected_at,
        purchase_months=purchase_months,
    )


def _select_template_id(
    audience: Literal["new", "guest", "member"],
    predicted_item: Mapping[str, Any] | None,
) -> str:
    if audience == "new":
        return "ME0000"

    category_key = _normalise_category(predicted_item.get("category") if predicted_item else None)
    product_name = (predicted_item or {}).get("product_name")
    if not category_key and isinstance(product_name, str):
        category_key = _infer_category_from_name(product_name)

    if category_key:
        for template_id, candidates in CATEGORY_TEMPLATE_RULES.items():
            if category_key in candidates:
                return template_id
    return DEFAULT_TEMPLATE_ID


def _normalise_prediction(
    predicted: Mapping[str, Any] | Any | None,
) -> Mapping[str, Any] | None:
    if predicted is None:
        return None

    if isinstance(predicted, Mapping):
        base = dict(predicted)
    else:
        base = {
            "product_code": getattr(predicted, "product_code", None),
            "product_name": getattr(predicted, "product_name", None),
            "category": getattr(predicted, "category", None),
            "category_label": getattr(predicted, "category_label", None),
            "price": getattr(predicted, "price", None),
            "probability": getattr(predicted, "probability", None),
            "probability_percent": getattr(predicted, "probability_percent", None),
        }

    probability = base.get("probability")
    if probability is not None and base.get("probability_percent") is None:
        try:
            base["probability_percent"] = round(float(probability) * 100, 1)
        except (TypeError, ValueError):
            base.pop("probability_percent", None)

    return {key: value for key, value in base.items() if value is not None}


def _normalise_prediction_list(
    items: Sequence[Mapping[str, Any] | Any] | None,
) -> list[Mapping[str, Any]]:
    if not items:
        return []
    normalized: list[Mapping[str, Any]] = []
    for item in items:
        value = _normalise_prediction(item)
        if value:
            normalized.append(value)
    return normalized


def _normalise_category(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower()


def _infer_category_from_name(name: str | None) -> str:
    if not name:
        return ""
    lowered = name.lower()
    if any(keyword in lowered for keyword in ("蛋糕", "甜", "慕斯", "鬆餅", "bread", "cake")):
        return "dessert"
    if any(keyword in lowered for keyword in ("親子", "幼兒", "園", "兒童")):
        return "kindergarten"
    if any(keyword in lowered for keyword in ("健身", "運動", "能量", "瑜珈", "protein")):
        return "fitness"
    return ""


def _profile_to_dict(profile: MemberProfile | None) -> Mapping[str, Any] | None:
    if profile is None:
        return None

    return {
        "name": profile.name,
        "member_id": profile.member_id,
        "member_code": profile.mall_member_id,
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
        "first_image_filename": profile.first_image_filename,
    }


def _fallback_copy(
    audience: Literal["new", "guest", "member"],
    predicted: Mapping[str, Any] | None,
) -> tuple[str, str, str, Optional[str]]:
    if audience == "new":
        headline = _random_line(
            ["歡迎光臨星悅商場", "第一次來店，立即享受尊榮體驗"],
            "歡迎光臨！",
        )
        subheading = _random_line(
            [
                "第一次到訪，服務人員將協助完成入會並介紹今日亮點",
                "帶您快速了解熱門櫃點與入會流程，現場即可啟用禮遇",
            ],
            "第一次來到本店，服務人員將協助你加入會員並介紹今日亮點。",
        )
        highlight = _unique_highlight(
            [
                "掃描右下角 QR Code 完成入會，即享迎賓咖啡與 120 點開卡禮",
                "入會即可領取甜點兌換券與專屬諮詢席次",
            ],
            "掃描右下角 QR Code 完成入會，即享迎賓咖啡與 120 點開卡禮。",
        )
        return (headline, subheading, highlight, CTA_JOIN_MEMBER)

    product_name = str(predicted.get("product_name")) if predicted and predicted.get("product_name") else None
    price = predicted.get("price") if predicted else None
    category_label = str(predicted.get("category_label", "人氣商品")) if predicted else "人氣商品"

    price_text = ""
    if isinstance(price, (int, float)) and price > 0:
        price_text = f"建議售價 NT${int(round(price))}"

    if audience == "guest":
        if product_name:
            headline = _random_line(
                [
                    f"{product_name} 限時預留",
                    f"{product_name} 今日僅此一檔",
                ],
                f"{product_name} 限時預留",
            )
            subheading = _random_line(
                [
                    "依照你的上月消費偏好，我們已為你預留熱門商品",
                    "為你保留最愛的熱銷品項，現場即可體驗",
                ],
                "依照你的上月消費偏好，我們已為你預留熱門商品",
            )
            if price_text:
                subheading = f"{subheading}，{price_text}"
            highlight = _unique_highlight(
                [
                    f"加入會員即享 {product_name} 首購 85 折，再送迎賓點數",
                    f"完成入會，{product_name} 立享專屬優惠",
                ],
                f"加入會員即享 {product_name} 首購 85 折，再送迎賓點數！",
            )
            return (headline, subheading, highlight, CTA_JOIN_MEMBER)
        headline = _random_line(
            ["加入會員，開啟專屬禮遇", "立即加入會員解鎖專屬優惠"],
            "加入會員，開啟專屬禮遇",
        )
        subheading = _random_line(
            [
                "立即解鎖生日禮、消費回饋與專屬活動席次",
                "入會即可綁定點數回饋，活動席次優先通知",
            ],
            "立即解鎖生日禮、消費回饋與專屬活動席次。",
        )
        highlight = _unique_highlight(
            [
                "現在入會送健康輕飲兌換券，掃描螢幕 QR Code 立刻加入",
                "入會加碼送甜點飲品券，現場立即兌換",
            ],
            "現在入會送健康輕飲兌換券，掃描螢幕 QR Code 立刻加入！",
        )
        return (headline, subheading, highlight, CTA_JOIN_MEMBER)

    # audience == "member"
    if product_name:
        detail = f"{category_label} 主題推薦"
        if price_text:
            detail = f"{detail}｜{price_text}"
        headline = _random_line(
            [f"會員專屬 {product_name}", f"{product_name} 會員限定回饋"],
            f"會員專屬 {product_name}",
        )
        highlight = _unique_highlight(
            [
                f"今日刷會員卡享 {product_name} 88 折，點數雙倍奉上",
                f"刷卡加碼點數，{product_name} 現場限量供應",
            ],
            f"今日刷會員卡享 {product_name} 88 折，點數雙倍奉上！",
        )
        return (headline, detail, highlight, CTA_MEMBER_OFFER)
    headline = _random_line(
        ["會員限定驚喜回饋", "會員尊享限定禮遇"],
        "會員限定驚喜回饋",
    )
    subheading = _random_line(
        [
            "點數可綁定熱門活動與體驗課程，快來補貨",
            "消費即可參加品牌限定活動，敬請把握",
        ],
        "點數可綁定熱門活動與體驗課程，快來補貨！",
    )
    highlight = _unique_highlight(
        [
            "本週消費滿額即送品牌旅行組，櫃點限時加碼中",
            "於專櫃消費滿額享豪華禮遇，再享點數加倍",
        ],
        "本週消費滿額即送品牌旅行組，櫃點限時加碼中。",
    )
    return (headline, subheading, highlight, CTA_MEMBER_OFFER)


def _member_salutation(member_code: str) -> str:
    if member_code:
        return f"會員 {member_code}"
    return "親愛的貴賓"


def _subheading_prefix(member_code: str) -> str:
    if member_code:
        return f"商場會員代號：{member_code}｜"
    return "尚未綁定商場會員，立即至服務台完成綁定享專屬禮遇｜"


AD_IMAGE_BY_SCENARIO: dict[str, str] = {
    "brand_new": TEMPLATE_IMAGE_BY_ID["ME0000"],
    "registered:dessert": TEMPLATE_IMAGE_BY_ID["ME0001"],
    "registered:kindergarten": TEMPLATE_IMAGE_BY_ID["ME0002"],
    "registered:fitness": TEMPLATE_IMAGE_BY_ID["ME0003"],
    "unregistered:dessert": TEMPLATE_IMAGE_BY_ID["ME0001"],
    "unregistered:kindergarten": TEMPLATE_IMAGE_BY_ID["ME0002"],
    "unregistered:fitness": TEMPLATE_IMAGE_BY_ID["ME0003"],
    "member:ME0001": TEMPLATE_IMAGE_BY_ID["ME0001"],
    "member:ME0002": TEMPLATE_IMAGE_BY_ID["ME0002"],
    "member:ME0003": TEMPLATE_IMAGE_BY_ID["ME0003"],
}
