"""Business logic to transform database rows into advertising copy."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

from .ai import AdCreative
from .database import MemberProfile, Purchase


PROFILE_SEGMENT_BY_LABEL: dict[str, str] = {
    "dessert-lover": "dessert",
    "family-groceries": "kindergarten",
    "fitness-enthusiast": "fitness",
    "home-manager": "homemaker",
    "wellness-gourmet": "fitness",
}


AD_IMAGE_BY_SCENARIO: dict[str, str] = {
    "brand_new": "ME0000.jpg",
    "registered:fitness": "ME0003.jpg",
    "registered:dessert": "ME0001.jpg",
    "registered:kindergarten": "ME0002.jpg",
    "unregistered:fitness": "AD0000.jpg",
    "unregistered:homemaker": "AD0001.jpg",
    "repeat_purchase:fitness": "ME0003.jpg",
    "repeat_purchase:dessert": "ME0001.jpg",
    "repeat_purchase:kindergarten": "ME0002.jpg",
    "repeat_purchase": "ME0001.jpg",
}


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

    # Treat the welcome gift as "no real history" so that brand-new members
    # still receive the onboarding style advertisement.
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
        # Recent purchases should have a higher impact – weight them slightly
        # more by decaying with the position in the list.
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
    segment = PROFILE_SEGMENT_BY_LABEL.get(profile_label)
    registered = bool(getattr(profile, "mall_member_id", ""))
    prefix = "registered" if registered else "unregistered"
    candidate_keys: list[str] = []

    if insights.scenario == "repeat_purchase":
        if segment:
            candidate_keys.append(f"repeat_purchase:{segment}")
        candidate_keys.append("repeat_purchase")

    if segment:
        candidate_keys.append(f"{prefix}:{segment}")

    candidate_keys.append("brand_new")

    for key in candidate_keys:
        if key in AD_IMAGE_BY_SCENARIO:
            return key

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
        highlight = (
            "掃描服務台 QR Code 馬上入會，今日完成註冊送咖啡招待與 120 點開卡禮！"
        )
        return headline, subheading, highlight

    if insights.scenario == "repeat_purchase":
        item = insights.recommended_item or (purchases[0].item if purchases else "人氣商品")
        headline = f"{greeting}，{item} 回購加碼！"
        subheading = (
            f"{subheading_code}最近 {insights.repeat_count} 次都選擇了 {item}"
        )
        if latest_summary:
            subheading += f"｜上次 {latest_summary}"
        highlight = (
            f"{item} 會員限定：第 {insights.repeat_count + 1} 件 82 折，再贈職人限定隨行包！"
        )
        return headline, subheading, highlight

    item = insights.recommended_item or (purchases[0].item if purchases else "人氣商品")
    probability_text = _format_probability(insights.probability)
    headline = f"{greeting}，預留了你的 {item}"
    subheading = f"{subheading_code}系統預測你對 {item} 的購買機率高達 {probability_text}"
    if latest_summary:
        subheading += f"｜上次 {latest_summary}"
    highlight = (
        f"{item} 今日限量再享會員專屬 88 折，結帳輸入 MEMBER95 加贈點數！"
    )
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

