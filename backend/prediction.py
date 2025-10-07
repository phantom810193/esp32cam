"""Prediction helpers for estimating next-best offers."""
from __future__ import annotations

import math
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Sequence

try:  # Optional dependency for calibration models.
    import numpy as _np
except Exception:  # pragma: no cover - numpy may be unavailable.
    _np = None

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


TAU_DEFAULT = 1.5
K_DEFAULT = 10
CALIB_DEFAULT = "backend/data/calibration.pkl"
_PROB_EPS = 1e-6


def _softmax(
    scores: Sequence[float], tau: float | None = None, eps: float = 1e-6
) -> list[float]:
    if not scores:
        return []
    tau_value = tau
    tau_env = os.getenv("PRED_TAU") if tau_value is None else None
    if tau_env:
        try:
            tau_value = float(tau_env)
        except ValueError:
            tau_value = None
    if tau_value is None:
        tau_value = TAU_DEFAULT
    if tau_value <= 0:
        tau_value = 1.0
    shift = max(scores)
    exponentials = [math.exp((score - shift) / tau_value) for score in scores]
    total = sum(exponentials)
    if total <= 0:
        return [1.0 / len(scores) for _ in scores]
    probs = [(value + eps) / (total + eps * len(exponentials)) for value in exponentials]
    normaliser = sum(probs) or 1.0
    return [prob / normaliser for prob in probs]


def _zscore(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= 1e-12:
        return list(values)
    std_dev = math.sqrt(variance)
    return [(value - mean) / std_dev for value in values]


def _load_calibrator(path: str | None):
    if not path:
        return None
    try:
        if os.path.exists(path):
            with open(path, "rb") as handle:
                return pickle.load(handle)
    except Exception:
        return None
    return None


def _apply_calibration(calibrator, probs: Sequence[float]) -> list[float]:
    if calibrator is None or not probs:
        return list(probs)
    if _np is None:
        return list(probs)
    try:
        array = _np.array(probs, dtype=float).reshape(-1, 1)
        if hasattr(calibrator, "predict_proba"):
            calibrated = calibrator.predict_proba(array)
            if calibrated.ndim == 2 and calibrated.shape[1] > 1:
                calibrated = calibrated[:, -1]
        else:
            calibrated = calibrator.predict(array)
        calibrated_list = [float(value) for value in _np.ravel(calibrated)]
        if len(calibrated_list) != len(probs):
            return list(probs)
        clipped = [max(0.001, min(0.999, value)) for value in calibrated_list]
        total = sum(clipped) or 1.0
        return [value / total for value in clipped]
    except Exception:
        return list(probs)


def _blend_with_prior(
    p_softmax: Sequence[float], prior: Sequence[float] | None, n: int | None, k: int
) -> list[float]:
    if not p_softmax:
        return []
    length = len(p_softmax)
    if not prior or len(prior) != length:
        prior = [1.0 / length] * length
    try:
        n_value = int(n) if n is not None else 0
    except (TypeError, ValueError):
        n_value = 0
    n_value = max(0, n_value)
    k_value = max(0, k)
    lambda_value = (
        float(k_value) / float(n_value + k_value)
        if (k_value > 0 and (n_value + k_value) > 0)
        else 0.0
    )
    blended = [
        (1 - lambda_value) * float(ps) + lambda_value * float(q)
        for ps, q in zip(p_softmax, prior)
    ]
    total = sum(blended) or 1.0
    return [value / total for value in blended]


def _normalise_probabilities(values: Sequence[float], eps: float = _PROB_EPS) -> list[float]:
    if not values:
        return []
    adjusted = [float(value) for value in values]
    total = sum(adjusted) + eps * len(adjusted)
    if total <= 0:
        return [1.0 / len(adjusted) for _ in adjusted]
    normalised = [(value + eps) / total for value in adjusted]
    norm_total = sum(normalised) or 1.0
    return [value / norm_total for value in normalised]


def _normalise_mapping(values: Mapping[str, float]) -> dict[str, float]:
    positive = {key: float(val) for key, val in values.items() if float(val) > 0}
    total = sum(positive.values())
    if total <= 0:
        return {}
    return {key: val / total for key, val in positive.items()}


def _recent_category_counts(
    purchases: Sequence[Purchase], days: int = 30
) -> Counter[str]:
    threshold = datetime.utcnow() - timedelta(days=days)
    counts: Counter[str] = Counter()
    for purchase in purchases:
        try:
            timestamp = parse_timestamp(purchase.purchased_at)
        except Exception:
            continue
        if timestamp >= threshold:
            category = infer_category_from_item(purchase.item)
            counts[category] += 1
    return counts


def _global_category_popularity(products: Sequence[Product]) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for product in products:
        weight = getattr(product, "view_rate", 0.0)
        if weight and weight > 0:
            totals[product.category] += float(weight)
        else:
            totals[product.category] += 1.0
    return _normalise_mapping(totals)


def _build_prior_vector(
    products: Sequence[Product],
    member_weights: Mapping[str, float] | None,
    global_weights: Mapping[str, float] | None,
) -> list[float]:
    if not products:
        return []
    weights: list[float] = []
    length = len(products)
    uniform_weight = 1.0 / length if length else 1.0
    for product in products:
        weight = None
        if member_weights:
            weight = member_weights.get(product.category)
        if (weight is None or weight <= 0) and global_weights:
            weight = global_weights.get(product.category)
        if weight is None or weight <= 0:
            weight = getattr(product, "view_rate", 0.0)
        if weight is None or weight <= 0:
            weight = uniform_weight
        weights.append(float(weight))
    return _normalise_probabilities(weights)


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
    global_category_weights = _global_category_popularity(catalogue)
    recent_category_counts = _recent_category_counts(purchase_list, days=30)
    member_category_weights = _normalise_mapping(recent_category_counts)
    n_recent_orders = int(sum(recent_category_counts.values()))
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
    scores = [score for score, _ in top_scored]
    standardised_scores = _zscore(scores)
    p_softmax = _softmax(standardised_scores)

    prior_vector = _build_prior_vector(
        [product for _, product in top_scored],
        member_category_weights,
        global_category_weights,
    )
    k_env = os.getenv("PRED_K")
    try:
        k_value = int(k_env) if k_env is not None else K_DEFAULT
    except ValueError:
        k_value = K_DEFAULT
    p_blend = _blend_with_prior(p_softmax, prior_vector, n_recent_orders, k_value)

    cal_path = os.getenv("PRED_CALIB_PATH", CALIB_DEFAULT)
    calibrator = _load_calibrator(cal_path)
    probabilities = _apply_calibration(calibrator, p_blend)
    clipped = [max(0.001, min(0.999, prob)) for prob in probabilities]
    probabilities = _normalise_probabilities(clipped)

    results: list[PredictedItem] = []
    for (_score, product), probability in zip(top_scored, probabilities):
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
