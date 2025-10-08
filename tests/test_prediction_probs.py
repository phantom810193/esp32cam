"""Tests for probability helpers in backend.prediction."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import MemberProfile, NEW_GUEST_MEMBER_ID, Purchase
from backend.prediction import (
    _blend_with_prior,
    _clip_and_normalize,
    _softmax,
    predict_next_purchases,
)


def test_softmax_outputs_valid_distribution():
    scores = [1.2, 3.4, 0.5]
    probabilities = _softmax(scores, tau=1.5)

    assert len(probabilities) == len(scores)
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert abs(sum(probabilities) - 1.0) < 1e-6


def test_softmax_temperature_controls_sharpness():
    scores = [1.0, 2.0, 5.0]
    cold = _softmax(scores, tau=0.5)
    hot = _softmax(scores, tau=5.0)

    assert max(cold) > max(hot)
    assert min(cold) < min(hot)


def test_blend_with_prior_respects_sample_size():
    p_softmax = [0.7, 0.2, 0.1]
    prior = [0.5, 0.3, 0.2]

    blended_small_n = _blend_with_prior(p_softmax, prior, n=0, k=10)
    blended_large_n = _blend_with_prior(p_softmax, prior, n=100, k=10)

    assert blended_small_n == _clip_and_normalize(prior)
    assert abs(blended_large_n[0] - p_softmax[0]) < 0.1


def test_softmax_translation_invariance_after_zscore():
    scores = [0.2, 1.4, -0.5]
    shifted = [score + 42.0 for score in scores]

    base = _softmax(scores)
    shifted_base = _softmax(shifted)

    assert abs(sum(base) - 1.0) < 1e-6
    assert abs(sum(shifted_base) - 1.0) < 1e-6
    for original, translated in zip(base, shifted_base):
        assert abs(original - translated) < 1e-9


def test_probability_percent_has_single_decimal():
    previous_calib = os.environ.pop("PRED_CALIB_PATH", None)

    purchase = Purchase(
        member_id="M001",
        member_code="M001",
        product_category="wellness",
        internal_item_code="SKU401022",
        item="冷壓綜合果汁 350ml",
        purchased_at="2024-05-10 10:00",
        unit_price=95.0,
        quantity=1.0,
        total_price=95.0,
    )

    profile = MemberProfile(
        profile_id=1,
        profile_label="test",
        name="Tester",
        member_id="M001",
        mall_member_id=None,
        member_status=None,
        joined_at=None,
        points_balance=None,
        gender=None,
        birth_date=None,
        phone=None,
        email=None,
        address=None,
        occupation=None,
        first_image_filename=None,
    )

    try:
        result = predict_next_purchases([purchase], profile=profile, limit=3)

        assert result.items
        for item in result.items:
            scaled = round(item.probability * 100, 1)
            assert item.probability_percent == scaled
            assert (
                abs(item.probability_percent * 10 - round(item.probability_percent * 10))
                < 1e-9
            )
    finally:
        if previous_calib is not None:
            os.environ["PRED_CALIB_PATH"] = previous_calib


def test_dessert_preferences_surface_dessert_catalogue_items():
    purchases = [
        Purchase(
            member_id="MEME0383FE3AA",
            member_code="ME0001",
            product_category="甜點",
            internal_item_code="DES-FAKE-001",
            item="草莓千層蛋糕",
            purchased_at="2025-09-02 12:30",
            unit_price=360.0,
            quantity=1.0,
            total_price=360.0,
        ),
        Purchase(
            member_id="MEME0383FE3AA",
            member_code="ME0001",
            product_category="甜點",
            internal_item_code="DES-FAKE-002",
            item="焦糖布丁禮盒",
            purchased_at="2025-09-05 18:40",
            unit_price=320.0,
            quantity=1.0,
            total_price=320.0,
        ),
        Purchase(
            member_id="MEME0383FE3AA",
            member_code="ME0001",
            product_category="咖啡廳",
            internal_item_code="DES-FAKE-003",
            item="可可奶霜鬆餅",
            purchased_at="2025-09-10 17:10",
            unit_price=280.0,
            quantity=1.0,
            total_price=280.0,
        ),
    ]

    profile = MemberProfile(
        profile_id=1,
        profile_label="dessert-lover",
        name="甜點愛好者",
        member_id="MEME0383FE3AA",
        mall_member_id="ME0001",
        member_status="有效",
        joined_at=None,
        points_balance=None,
        gender=None,
        birth_date=None,
        phone=None,
        email=None,
        address=None,
        occupation=None,
        first_image_filename=None,
    )

    result = predict_next_purchases(purchases, profile=profile, limit=4)

    assert result.items
    top_categories = {item.category for item in result.items[:3]}
    assert "dessert" in top_categories


def test_new_guest_member_has_no_predictions():
    profile = MemberProfile(
        profile_id=999,
        profile_label="brand-new-guest",
        name="新客",
        member_id=NEW_GUEST_MEMBER_ID,
        mall_member_id="",
        member_status="未入會",
        joined_at=None,
        points_balance=0.0,
        gender=None,
        birth_date=None,
        phone=None,
        email=None,
        address=None,
        occupation=None,
        first_image_filename=None,
    )

    result = predict_next_purchases([], profile=profile, limit=5)

    assert result.items == []
    assert result.history == []


def test_new_guest_without_member_id_still_returns_empty_predictions():
    profile = MemberProfile(
        profile_id=1000,
        profile_label="brand-new-guest",
        name="新客",
        member_id=None,
        mall_member_id="",
        member_status="未入會",
        joined_at=None,
        points_balance=0.0,
        gender=None,
        birth_date=None,
        phone=None,
        email=None,
        address=None,
        occupation=None,
        first_image_filename=None,
    )

    sample_purchase = Purchase(
        member_id="TEMP001",
        member_code="TEMP001",
        product_category="甜點",
        internal_item_code="TMP-0001",
        item="草莓蛋糕",
        purchased_at="2025-09-10 10:00",
        unit_price=280.0,
        quantity=1.0,
        total_price=280.0,
    )

    result = predict_next_purchases([sample_purchase], profile=profile, limit=5)

    assert result.items == []
    assert result.history == []

