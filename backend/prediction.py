"""Prediction helpers for estimating next-best offers."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp
from typing import Iterable, Sequence

from .advertising import PurchaseInsights
from .catalogue import (
    CATEGORY_LABELS,
    Product,
    category_label,
    get_catalogue,
    infer_category_from_item,
    purchased_product_codes,
)
from .database import MemberProfile, Purchase


@dataclass
class PredictedItem:
    product_code: str
    product_name: str
    category: str
    category_label: str
    price: float
    view_rate_percent: float
    probability: float
    probability_percent: float


@dataclass
class PredictionResult:
    items: list[PredictedItem]
    history: list[Purchase]
    top_products: list[dict[str, object]]
    activities: list[dict[str, str]]
    window_label: str


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError:
        return datetime.fromisoformat(value)


def get_previous_month_purchases(
    purchases: Iterable[Purchase], *, reference: datetime | None = None
) -> list[Purchase]:
    ordered = sorted(purchases, key=lambda purchase: parse_timestamp(purchase.purchased_at))
    if not ordered:
        return []

    reference_dt = reference or parse_timestamp(ordered[-1].purchased_at)
    reference_dt = reference_dt.replace(day=1)
    previous_month = reference_dt - timedelta(days=1)
    target_year = previous_month.year
    target_month = previous_month.month

    window: list[Purchase] = [
        purchase
        for purchase in ordered
        if (
            (ts := parse_timestamp(purchase.purchased_at)).year == target_year
            and ts.month == target_month
        )
    ]
    if window:
        return window

    buckets: defaultdict[tuple[int, int], list[Purchase]] = defaultdict(list)
    for purchase in ordered:
        ts = parse_timestamp(purchase.purchased_at)
        buckets[(ts.year, ts.month)].append(purchase)

    if not buckets:
        return []

    latest_bucket = max(buckets.keys())
    return buckets[latest_bucket]


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    shift = max(scores)
    exponentials = [exp(score - shift) for score in scores]
    total = sum(exponentials)
    if total == 0:
        return [0.0 for _ in scores]
    return [value / total for value in exponentials]


def predict_next_purchases(
    purchases: Iterable[Purchase],
    *,
    profile: MemberProfile | None = None,
    insights: PurchaseInsights | None = None,
    limit: int = 7,
) -> PredictionResult:
    purchase_list = list(purchases)
    window = get_previous_month_purchases(purchase_list)
    if not window:
        window = purchase_list

    # Build frequency data.
    category_counter: Counter[str] = Counter()
    price_points: list[float] = []
    for purchase in window:
        category = infer_category_from_item(purchase.item)
        category_counter[category] += 1
        price_points.append(float(purchase.unit_price))

    history_set = {purchase.item for purchase in purchase_list}
    purchased_codes = purchased_product_codes(history_set)

    total_category = sum(category_counter.values()) or 1
    category_weight_map = {
        category: count / total_category for category, count in category_counter.items()
    }

    recent_categories = [infer_category_from_item(p.item) for p in purchase_list[-5:]]
    trend_weight: defaultdict[str, float] = defaultdict(float)
    for index, category in enumerate(reversed(recent_categories), start=1):
        trend_weight[category] += 1 / index
    if trend_weight:
        max_trend = max(trend_weight.values())
    else:
        max_trend = 1.0

    average_price = sum(price_points) / len(price_points) if price_points else 1200.0

    novelty_boost = 1.0 if profile and profile.mall_member_id else 0.8

    catalogue = get_catalogue()
    scored: list[tuple[float, Product]] = []
    for product in catalogue:
        if product.code in purchased_codes:
            continue

        cat_weight = category_weight_map.get(product.category, 0.15)
        recency_bonus = trend_weight.get(product.category, 0.0) / max_trend
        price_similarity = 1.0 - min(
            abs(product.price - average_price) / max(product.price, average_price, 1.0),
            1.0,
        )
        novelty = novelty_boost if product.name not in history_set else 0.3

        score = (
            0.45 * cat_weight
            + 0.25 * recency_bonus
            + 0.2 * price_similarity
            + 0.1 * novelty
        )
        scored.append((score, product))

    scored.sort(key=lambda entry: entry[0], reverse=True)
    top_scored = scored[: limit * 2]
    probabilities = _softmax([score for score, _ in top_scored])

    results: list[PredictedItem] = []
    for index, ((score, product), probability) in enumerate(zip(top_scored, probabilities)):
        if len(results) >= limit:
            break
        adjusted_view_rate = product.view_rate * (0.9 + category_weight_map.get(product.category, 0.1))
        adjusted_view_rate = min(1.0, adjusted_view_rate)
        probability_percent = round(probability * 100, 1)
        results.append(
            PredictedItem(
                product_code=product.code,
                product_name=product.name,
                category=product.category,
                category_label=category_label(product.category),
                price=product.price,
                view_rate_percent=round(adjusted_view_rate * 100, 1),
                probability=probability,
                probability_percent=probability_percent,
            )
        )

    # Build supporting data for the manager dashboard.
    window_label = _format_window_label(window)
    history = sorted(window, key=lambda p: parse_timestamp(p.purchased_at), reverse=True)
    top_products = _summarise_top_products(window)
    activities = _build_activity_feed(window, insights)

    return PredictionResult(
        items=results,
        history=history,
        top_products=top_products,
        activities=activities,
        window_label=window_label,
    )


def _format_window_label(window: Sequence[Purchase]) -> str:
    if not window:
        return "近期"
    latest = parse_timestamp(window[-1].purchased_at)
    return f"{latest.year} 年 {latest.month:02d} 月"


def _summarise_top_products(purchases: Sequence[Purchase]) -> list[dict[str, object]]:
    counter: Counter[str] = Counter()
    revenue: defaultdict[str, float] = defaultdict(float)
    for purchase in purchases:
        counter[purchase.item] += float(purchase.quantity)
        revenue[purchase.item] += float(purchase.total_price)
    top_items = counter.most_common(3)
    summary: list[dict[str, object]] = []
    for name, quantity in top_items:
        summary.append(
            {
                "item": name,
                "quantity": quantity,
                "total_price": round(revenue[name], 1),
            }
        )
    return summary


def _build_activity_feed(
    purchases: Sequence[Purchase], insights: PurchaseInsights | None
) -> list[dict[str, str]]:
    if not purchases:
        return []
    latest_timestamp = parse_timestamp(purchases[-1].purchased_at)
    categories = [infer_category_from_item(purchase.item) for purchase in purchases]
    category_counts = Counter(categories)

    templates = {
        "fitness": "參與了會員健身互動課程，完成體能評估",
        "dessert": "參加甜點試吃活動，留下高度滿意回饋",
        "kindergarten": "參與親子共學日並關注幼兒成長課程",
        "homemaker": "參與家居市集體驗新品居家用品",
        "general": "逛遊生活選品快閃店，對永續主題最感興趣",
    }

    feed: list[dict[str, str]] = []
    for category, count in category_counts.most_common(3):
        description = templates.get(category, templates["general"])
        feed.append(
            {
                "title": CATEGORY_LABELS.get(category, "生活選品"),
                "description": description,
                "timestamp": (latest_timestamp - timedelta(days=count)).strftime("%Y-%m-%d"),
            }
        )

    if insights and insights.recommended_item:
        feed.insert(
            0,
            {
                "title": "AI 推薦",
                "description": f"針對 {insights.recommended_item} 提供專屬回訪方案",
                "timestamp": latest_timestamp.strftime("%Y-%m-%d"),
            },
        )

    return feed[:3]
