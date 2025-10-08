"""Prediction helpers for estimating next-best offers."""
from __future__ import annotations

import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import exp
from pathlib import Path
from statistics import mean, pstdev
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
from .database import MemberProfile, NEW_GUEST_MEMBER_ID, Purchase


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


def _zscore(values: Sequence[float], tol: float = 1e-9) -> list[float]:
    data = list(values)
    if len(data) < 2:
        return data

    mu = mean(data)
    sigma = pstdev(data)
    if sigma <= tol:
        return data

    return [(value - mu) / sigma for value in data]


def _softmax(scores: Sequence[float], tau: float = 1.5, eps: float = 1e-6) -> list[float]:
    if not scores:
        return []

    if tau <= 0:
        tau = 1.0

    normalised = _zscore(scores)

    scaled = [score / tau for score in normalised]
    shift = max(scaled)
    exponentials = [exp(score - shift) for score in scaled]
    total = sum(exponentials)
    if total <= eps:
        return [1.0 / len(scores) for _ in scores]
    return [value / total for value in exponentials]


def _blend_with_prior(
    p_softmax: Sequence[float], prior: Sequence[float], n: int, k: int
) -> list[float]:
    if not p_softmax:
        return []

    if len(p_softmax) != len(prior):
        prior = [1.0 for _ in p_softmax]

    k = max(0, int(k))
    n = max(0, int(n))
    if not any(prior):
        prior = [1.0 / len(p_softmax) for _ in p_softmax]
    total_prior = sum(prior)
    if total_prior <= 0:
        normalised_prior = [1.0 / len(prior) for _ in prior]
    else:
        normalised_prior = [value / total_prior for value in prior]

    if k == 0:
        return list(p_softmax)

    lam = k / (n + k)
    blended = [
        (1 - lam) * p + lam * q for p, q in zip(p_softmax, normalised_prior)
    ]
    total_blended = sum(blended)
    if total_blended <= 0:
        return [1.0 / len(blended) for _ in blended]
    return [value / total_blended for value in blended]


def _clip_and_normalize(probabilities: Sequence[float]) -> list[float]:
    if not probabilities:
        return []
    clipped = [min(0.999, max(0.001, float(p))) for p in probabilities]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / len(clipped) for _ in clipped]
    return [value / total for value in clipped]


def _load_calibrator(path: str | None) -> object | None:
    if not path:
        return None
    calibrator_path = Path(path)
    if not calibrator_path.exists():
        return None
    try:
        with calibrator_path.open("rb") as fh:
            return pickle.load(fh)
    except (OSError, pickle.PickleError):
        return None


def _apply_calibration(probabilities: Sequence[float], calibrator: object | None) -> list[float]:
    if not probabilities:
        return []
    if calibrator is None:
        return list(probabilities)
    try:
        if hasattr(calibrator, "transform"):
            transformed = calibrator.transform([probabilities])
            if transformed is not None:
                return list(transformed[0])
        if hasattr(calibrator, "predict_proba"):
            predicted = calibrator.predict_proba([probabilities])
            if predicted is not None:
                first = predicted[0]
                if isinstance(first, Sequence):
                    return list(first)
        if callable(calibrator):
            result = calibrator(probabilities)
            if isinstance(result, Sequence):
                return [float(value) for value in result]
    except Exception:
        return list(probabilities)
    return list(probabilities)


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _flag_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _feature_weights() -> dict[str, float]:
    weights = {
        "category": _get_env_float("PRED_WEIGHT_CATEGORY", 0.45),
        "trend": _get_env_float("PRED_WEIGHT_TREND", 0.25),
        "price": _get_env_float("PRED_WEIGHT_PRICE", 0.2),
        "novelty": _get_env_float("PRED_WEIGHT_NOVELTY", 0.1),
        "view_rate": _get_env_float("PRED_WEIGHT_VIEW_RATE", 0.0),
    }
    if not _flag_from_env("PRED_ENABLE_VIEW_RATE", False):
        weights["view_rate"] = 0.0
    return weights


def _global_category_popularity(catalogue: Sequence[Product]) -> dict[str, float]:
    popularity: defaultdict[str, float] = defaultdict(float)
    for product in catalogue:
        view_rate = getattr(product, "view_rate", 0.0) or 0.0
        base = view_rate if view_rate > 0 else 1.0
        popularity[product.category] += float(base)
    total = sum(popularity.values())
    if total <= 0:
        return {}
    return {category: value / total for category, value in popularity.items()}


def _build_prior_vector(
    candidates: Sequence[tuple[float, Product]],
    member_category_weights: dict[str, float],
    global_popularity: dict[str, float],
) -> list[float]:
    if not candidates:
        return []

    prior_values: list[float] = []
    for _, product in candidates:
        value: float | None = None
        if member_category_weights:
            value = member_category_weights.get(product.category)
        if value is None or value == 0:
            value = global_popularity.get(product.category)
        if value is None or value == 0:
            value = 1.0
        prior_values.append(float(value))

    total = sum(prior_values)
    if total <= 0:
        return [1.0 / len(prior_values) for _ in prior_values]
    return [value / total for value in prior_values]


def _count_recent_orders(purchases: Sequence[Purchase], window_days: int = 30) -> int:
    if not purchases:
        return 0
    timestamps = [parse_timestamp(purchase.purchased_at) for purchase in purchases]
    if not timestamps:
        return 0
    reference = max(timestamps)
    cutoff = reference - timedelta(days=window_days)
    return sum(1 for ts in timestamps if ts >= cutoff)


def predict_next_purchases(
    purchases: Iterable[Purchase],
    *,
    profile: MemberProfile | None = None,
    insights: PurchaseInsights | None = None,
    limit: int = 7,
) -> PredictionResult:
    purchase_list = list(purchases)
    profile_label = (profile.profile_label or "") if profile else ""
    is_new_guest = bool(
        profile
        and (
            profile.member_id == NEW_GUEST_MEMBER_ID
            or profile_label.strip().lower() == "brand-new-guest"
        )
    )

    if is_new_guest:
        activities = _build_activity_feed([], insights)
        return PredictionResult(
            items=[],
            history=[],
            top_products=[],
            activities=activities,
            window_label=_format_window_label([]),
        )
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
    weights = _feature_weights()
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
        raw_view_rate = getattr(product, "view_rate", 0.0) or 0.0
        view_rate_feature = float(raw_view_rate)

        score = (
            weights["category"] * cat_weight
            + weights["trend"] * recency_bonus
            + weights["price"] * price_similarity
            + weights["novelty"] * novelty
            + weights["view_rate"] * view_rate_feature
        )
        scored.append((score, product))

    scored.sort(key=lambda entry: entry[0], reverse=True)
    top_scored = scored[: limit * 2]
    tau = _get_env_float("PRED_TAU", 1.5)
    raw_scores = [score for score, _ in top_scored]
    probabilities = _softmax(raw_scores, tau=tau)

    global_popularity = _global_category_popularity(catalogue)
    prior_vector = _build_prior_vector(top_scored, category_weight_map, global_popularity)

    k = _get_env_int("PRED_K", 10)
    n_recent_orders = _count_recent_orders(purchase_list, window_days=30)
    probabilities = _blend_with_prior(probabilities, prior_vector, n_recent_orders, k)

    calibrator_path = os.getenv("PRED_CALIB_PATH", "backend/data/calibration.pkl")
    calibrator = _load_calibrator(calibrator_path)
    probabilities = _apply_calibration(probabilities, calibrator)
    probabilities = probabilities[:limit]
    output_probabilities = _clip_and_normalize(probabilities)

    results: list[PredictedItem] = []
    for (score, product), probability in zip(top_scored[:limit], output_probabilities):
        raw_view_rate = getattr(product, "view_rate", 0.0) or 0.0
        adjusted_view_rate = raw_view_rate * (
            0.9 + category_weight_map.get(product.category, 0.1)
        )
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
