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
        discount = int(latest.discount * 100)
        subheading = (
            f"上次購買：{latest.item}（{latest.last_purchase}），今日專屬 {discount}% OFF"
            if latest.discount > 0
            else f"上次購買：{latest.item}（{latest.last_purchase}）"
        )
        highlight = latest.recommendation
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

