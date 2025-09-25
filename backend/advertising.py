"""Business logic to transform database rows into advertising copy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .ai import AdCreative
from .database import Purchase


@dataclass
class AdContext:
    member_id: str
    headline: str
    subheading: str
    highlight: str
    purchases: list[Purchase]


def build_ad_context(
    member_id: str, purchases: Iterable[Purchase], creative: AdCreative | None = None
) -> AdContext:
    purchases = list(purchases)
    if creative:
        headline = creative.headline or f"會員 {member_id}，歡迎回來！"
        subheading = creative.subheading or "專屬優惠馬上開啟"
        highlight = creative.highlight or "今日限定：全店指定品項 85 折"
    elif purchases:
        latest = purchases[0]
        headline = f"會員 {member_id}，歡迎回來！"
        subheading = (
            f"上次消費 {latest.purchased_at}｜{latest.item}｜${latest.total_price:,.0f}（{_format_quantity(latest.quantity)} 件）"
        )
        highlight = _derive_highlight(member_id, purchases)
    else:
        headline = f"歡迎加入，會員 {member_id}!"
        subheading = "首次消費享 95 折，再送咖啡一杯"
        highlight = "快來體驗本週人氣商品：莊園咖啡豆 + 手工可頌組合"
    return AdContext(
        member_id=member_id,
        headline=headline,
        subheading=subheading,
        highlight=highlight,
        purchases=purchases,
    )


def _format_quantity(quantity: float) -> str:
    if quantity.is_integer():
        return str(int(quantity))
    return f"{quantity:.1f}"


def _derive_highlight(member_id: str, purchases: list[Purchase]) -> str:
    dessert_keywords = ("蛋糕", "塔", "布丁", "慕斯", "鬆餅", "捲", "派", "甜", "奶酪", "可麗餅")
    kids_keywords = ("幼兒", "親子", "園", "兒童", "才藝")

    dessert_hits = sum(1 for purchase in purchases if _matches_keywords(purchase.item, dessert_keywords))
    kids_hits = sum(1 for purchase in purchases if _matches_keywords(purchase.item, kids_keywords))

    top_items = [purchase.item for purchase in purchases[:3]]
    if kids_hits >= dessert_hits and kids_hits > 0:
        focus = "、".join(top_items[:2]) if top_items else "近期活動"
        return (
            f"幼兒園異業合作限定：持 {focus} 消費憑證，至合作幼兒園體驗課享 85 折，再送入園準備包！"
        )

    if dessert_hits > 0:
        focus = "、".join(top_items[:2]) if top_items else "人氣甜點"
        return (
            f"甜點控必看：{focus} 今日第二件 6 折，加碼手作迷你甜塔免費送！"
        )

    if purchases:
        focus = purchases[0].item
        return f"本週推薦 {focus}，結帳再享會員加碼 95 折。"

    return "今日加購指定品項，再享會員點數雙倍回饋！"


def _matches_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)

