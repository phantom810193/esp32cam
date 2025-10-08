"""Google Gemini powered advertisement generation endpoints."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

import random
from secrets import token_hex

from flask import Blueprint, jsonify, request, url_for

from ..ai import GeminiService, GeminiUnavailableError
from ..advertising import CATEGORY_TEMPLATE_RULES, TEMPLATE_IMAGE_BY_ID

adgen_blueprint = Blueprint("adgen", __name__)

_gemini_service = GeminiService()

_DEFAULT_TEMPLATE_ID = "ME0003"
_KEYWORD_TEMPLATE_HINTS = {
    "dessert": "ME0001",
    "sweet": "ME0001",
    "cake": "ME0001",
    "bakery": "ME0001",
    "幼兒": "ME0002",
    "親子": "ME0002",
    "kids": "ME0002",
    "kindergarten": "ME0002",
    "健身": "ME0003",
    "運動": "ME0003",
    "fitness": "ME0003",
}


def _normalise_template_id(candidate: str | None) -> str | None:
    if not candidate:
        return None
    value = candidate.strip().upper()
    if value.endswith(".JPG"):
        value = value[:-4]
    if value in TEMPLATE_IMAGE_BY_ID:
        return value
    return None


def _collect_hint_tokens(payload: Dict[str, Any], member_profile: Dict[str, Any]) -> Iterable[str]:
    tokens: list[str] = []
    for key in ("category", "theme", "template", "segment"):
        value = payload.get(key)
        if isinstance(value, str):
            tokens.append(value.lower())
    sku = payload.get("sku")
    if isinstance(sku, str):
        tokens.extend(part.lower() for part in sku.split())
    for key in ("segment", "profile_label", "template_id"):
        value = member_profile.get(key)
        if isinstance(value, str):
            tokens.append(value.lower())
    tags = member_profile.get("tags")
    if isinstance(tags, (list, tuple)):
        for tag in tags:
            if isinstance(tag, str):
                tokens.append(tag.lower())
    return tokens


def _resolve_template_id(payload: Dict[str, Any], member_profile: Dict[str, Any]) -> str:
    explicit = _normalise_template_id(payload.get("template_id"))
    if explicit:
        return explicit

    profile_hint = _normalise_template_id(member_profile.get("template_id"))
    if profile_hint:
        return profile_hint

    tokens = list(_collect_hint_tokens(payload, member_profile))
    for token in tokens:
        for template_id, categories in CATEGORY_TEMPLATE_RULES.items():
            if token in categories:
                return template_id

    combined = " ".join(tokens)
    for keyword, template_id in _KEYWORD_TEMPLATE_HINTS.items():
        if keyword in combined:
            return template_id

    return _DEFAULT_TEMPLATE_ID


def _build_background_url(template_id: str) -> str:
    filename = TEMPLATE_IMAGE_BY_ID.get(template_id) or TEMPLATE_IMAGE_BY_ID[_DEFAULT_TEMPLATE_ID]
    return url_for("static", filename=f"images/ads/{filename}", _external=True)


def _random_line(options: Iterable[str], fallback: str) -> str:
    pool = [option for option in options if option]
    if not pool:
        pool = [fallback]
    return random.choice(pool)


def _unique_highlight(options: Iterable[str], fallback: str) -> str:
    base = _random_line(options, fallback)
    code = token_hex(2).upper()
    if "{code}" in base:
        return base.format(code=code)
    return f"{base}｜限時代碼 {code}"


def _fallback_copy(audience: str) -> Dict[str, str]:
    audience_key = (audience or "guest").strip().lower()
    if audience_key == "member":
        return {
            "headline": _random_line(
                ["會員尊享限定禮遇", "會員專屬加碼回饋"],
                "會員尊享限定禮遇",
            ),
            "subheading": _random_line(
                [
                    "立即憑會員編號兌換專屬好禮",
                    "點數同步加碼，尊享多重優惠",
                ],
                "立即憑會員編號兌換專屬好禮",
            ),
            "highlight": _unique_highlight(
                [
                    "感謝長期支持，館內人氣品牌同步加碼折扣，錯過不再",
                    "憑會員卡消費即享加碼禮，人氣品牌同步回饋",
                ],
                "感謝長期支持，館內人氣品牌同步加碼折扣，錯過不再。",
            ),
            "cta": "立即領取會員獎勵",
        }
    if audience_key == "new":
        return {
            "headline": _random_line(
                ["歡迎加入星悅商場", "首次來店，專屬禮遇立即開啟"],
                "歡迎加入星悅商場",
            ),
            "subheading": _random_line(
                [
                    "首次來店享入會好禮與免費體驗",
                    "現場專人協助入會，立即解鎖多重驚喜",
                ],
                "首次來店享入會好禮與免費體驗",
            ),
            "highlight": _unique_highlight(
                [
                    "填寫基本資料即可獲得甜點招待券，還有多項入會驚喜等你探索",
                    "入會即送咖啡兌換券與體驗課程，一次擁有多重禮遇",
                ],
                "填寫基本資料即可獲得甜點招待券，還有多項入會驚喜等你探索。",
            ),
            "cta": "立即啟動迎賓禮",
        }
    return {
        "headline": _random_line(
            ["加入會員 解鎖專屬優惠", "加入會員即享限定回饋"],
            "加入會員 解鎖專屬優惠",
        ),
        "subheading": _random_line(
            [
                "立即完成註冊，暢享館內好康",
                "綁定會員即可獲得生日禮與專屬點數",
            ],
            "立即完成註冊，暢享館內好康",
        ),
        "highlight": _unique_highlight(
            [
                "新朋友限定！加入即可領取甜點兌換券與全館 95 折購物禮遇",
                "入會送迎賓點數與專屬折扣券，立即行動",
            ],
            "新朋友限定！加入即可領取甜點兌換券與全館 95 折購物禮遇。",
        ),
        "cta": "立即加入會員",
    }


@adgen_blueprint.post("/adgen")
def create_generated_ad():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        payload = {}

    member_id = str(payload.get("member_id") or "GUEST").strip() or "GUEST"
    audience = str(payload.get("audience") or "guest")
    purchases = payload.get("purchases") or []
    if not isinstance(purchases, list):
        purchases = []
    purchases_payload: list[dict[str, Any]] = []
    for entry in purchases:
        if isinstance(entry, dict):
            purchases_payload.append(entry)

    predicted = payload.get("predicted")
    if not isinstance(predicted, dict):
        predicted = None

    member_profile = payload.get("member_profile")
    if not isinstance(member_profile, dict):
        member_profile = {}

    try:
        creative = _gemini_service.generate_ad_copy(
            member_id=member_id,
            purchases=purchases_payload,
            predicted=predicted,
            audience=audience,
        )
        copy_payload = {
            "headline": creative.headline,
            "subheading": creative.subheading,
            "highlight": creative.highlight,
            "cta": creative.cta or _fallback_copy(audience)["cta"],
        }
    except GeminiUnavailableError as exc:
        logging.warning("Gemini unavailable during /adgen: %s", exc)
        copy_payload = _fallback_copy(audience)
        status_code = 503
    except Exception as exc:  # pragma: no cover - defensive catch
        logging.exception("Unexpected Gemini failure during /adgen: %s", exc)
        copy_payload = _fallback_copy(audience)
        status_code = 500
    else:
        status_code = 200

    template_id = _resolve_template_id(payload, member_profile)
    image_url = _build_background_url(template_id)

    response_body = {
        **copy_payload,
        "template_id": template_id,
        "image_url": image_url,
    }
    return jsonify(response_body), status_code
