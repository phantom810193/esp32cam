"""Gemini service helpers for generating personalised advertising copy."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

import google.generativeai as genai

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from .advertising import PurchaseInsights

_LOGGER = logging.getLogger("backend.ai")


class GeminiUnavailableError(RuntimeError):
    """Raised when Gemini generative features are requested but unavailable."""


@dataclass
class AdCreative:
    """Structured ad copy returned by the AI generator."""

    headline: str
    subheading: str
    highlight: str
    cta: str | None = None


class GeminiService:
    """Wrapper around Google Gemini text models used by the application."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 20.0,
    ) -> None:
        # Allow both the new GEMINI_* variables and legacy ADGEN_TEXT_MODEL overrides.
        default_model = (
            os.getenv("GEMINI_TEXT_MODEL")
            or os.getenv("ADGEN_TEXT_MODEL")
            or "gemini-2.5-flash-lite"
        )
        self._model_name = model_name or default_model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._timeout = timeout

        self._model: Optional[genai.GenerativeModel] = None
        self._init_error: Optional[str] = None

    # ------------------------------------------------------------------
    def _ensure_model(self) -> genai.GenerativeModel:
        if self._model is not None:
            return self._model

        if not self._api_key:
            self._init_error = "GEMINI_API_KEY is not configured"
            raise GeminiUnavailableError(self._init_error)

        try:
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self._model_name)
            self._init_error = None
            _LOGGER.info("Gemini model initialised: %s", self._model_name)
        except Exception as exc:  # pragma: no cover - network/SDK failures
            self._init_error = str(exc)
            raise GeminiUnavailableError(f"Failed to initialise Gemini model: {exc}") from exc

        return self._model

    # ------------------------------------------------------------------
    @property
    def can_describe_faces(self) -> bool:
        """Face description is not available via the Gemini text-only client."""
        return False

    @property
    def can_generate_ads(self) -> bool:
        if self._model is not None:
            return True
        if self._init_error:
            return False
        return bool(self._api_key)

    # ------------------------------------------------------------------
    def describe_face(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:  # noqa: ARG002 - signature retained for API compatibility
        raise GeminiUnavailableError("Gemini Vision is not available in this deployment")

    # ------------------------------------------------------------------
    def generate_ad_copy(
        self,
        member_id: str,
        purchases: Iterable[dict[str, object]],
        *,
        insights: "PurchaseInsights | None" = None,
        predicted: Mapping[str, Any] | None = None,
        audience: str = "guest",
    ) -> AdCreative:
        """Use Gemini Text to produce fresh advertising copy."""
        model = self._ensure_model()

        purchase_list = list(purchases)
        prompt_payload = json.dumps(purchase_list, ensure_ascii=False)
        insight_summary = self._describe_insights(insights)
        insight_block = f"情境重點：{insight_summary}" if insight_summary else ""

        audience_key = (audience or "guest").strip().lower()
        if audience_key not in {"guest", "member", "new"}:
            audience_key = "guest"

        if audience_key == "member":
            audience_label = "已註冊會員"
            objective_line = "請凸顯會員專屬優惠、點數或加碼折扣，促使用戶立即回購。"
            cta_hint = "cta 欄位需是 8~14 字元的行動呼籲，可包含「立即搶購」或「前往櫃位」。"
        elif audience_key == "guest":
            audience_label = "尚未加入會員的潛在顧客"
            objective_line = "請強調加入會員的好處與立即購買的誘因，語氣友善熱情。"
            cta_hint = "cta 欄位需是 8~14 字元的行動呼籲，必須包含「加入會員」或同義字詞。"
        else:
            audience_label = "首次來店的訪客"
            objective_line = "請以歡迎語氣介紹商場亮點並引導加入會員。"
            cta_hint = "cta 欄位需是 8~14 字元的行動呼籲，帶入「立即了解」或「加入會員」。"

        prediction_line = ""
        if predicted:
            product_name = str(predicted.get("product_name", "即將主推商品"))
            category_label = str(predicted.get("category_label", "人氣品類"))
            probability = predicted.get("probability_percent")
            price = predicted.get("price")
            probability_text = f"預估購買機率約 {probability}% " if probability is not None else ""
            if isinstance(price, (int, float)):
                price_text = f"建議售價 NT${int(round(price))}"
            else:
                price_text = ""
            prediction_line = f"目標推播商品：{product_name}（類別：{category_label}）。 {probability_text}{price_text}".strip()

        prompt = f"""
你是一位零售商場的廣告文案專家，目標是為{audience_label}生成 3 段式的繁體中文廣告內容。{objective_line}
請閱讀以下資料並輸出 JSON 物件，其中 headline 不超過 16 個字、subheading 不超過 32 個字、highlight 為 20~40 字的吸睛促購語。{cta_hint}

背景資訊：
- 會員 ID：{member_id}
- {prediction_line if prediction_line else "目標推播商品：請延伸自歷史資料"}
- {insight_block if insight_block else "無額外情境備註"}

歷史消費紀錄 JSON：{prompt_payload}

請僅輸出以下格式的 JSON，不得加入解說：
{{
  "headline": "...",
  "subheading": "...",
  "highlight": "...",
  "cta": "..."
}}
"""
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.65, "max_output_tokens": 512},
            )
        except Exception as exc:  # pragma: no cover - SDK/network failure paths
            raise GeminiUnavailableError(f"Gemini Text generation failed: {exc}") from exc

        text = self._extract_text(response)
        if not text:
            raise GeminiUnavailableError("Gemini Text returned an empty response")
        return self._parse_ad_response(text)

    # ------------------------------------------------------------------
    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if candidates:
            for candidate in candidates:
                parts = getattr(candidate, "content", None)
                if parts and getattr(parts, "parts", None):
                    joined = "".join(getattr(part, "text", "") for part in parts.parts)
                    if joined.strip():
                        return joined.strip()
                if getattr(candidate, "text", None):
                    cand_text = str(candidate.text).strip()
                    if cand_text:
                        return cand_text
        return ""

    # ------------------------------------------------------------------
    def _describe_insights(self, insights: "PurchaseInsights | None") -> str:
        if not insights:
            return ""

        scenario = getattr(insights, "scenario", "")
        recommended = getattr(insights, "recommended_item", None)
        repeat_count = getattr(insights, "repeat_count", 0)
        probability = getattr(insights, "probability", 0.0)
        probability_percent = getattr(insights, "probability_percent", None)
        total_orders = getattr(insights, "total_purchases", 0)

        if scenario == "brand_new":
            return "第一次來店，請引導加入會員並介紹開卡禮。"

        if scenario == "repeat_purchase" and recommended:
            return (
                f"顧客近期 {repeat_count} 次都購買 {recommended}，"
                "請針對此商品延伸加值組合與升級方案。"
            )

        if scenario == "returning_member" and recommended:
            probability_pct = probability_percent
            if probability_pct is None:
                probability_pct = round(max(0.0, float(probability)) * 100)
            return (
                f"老會員累積 {total_orders} 筆消費，AI 預測對 {recommended} 的購買機率約 "
                f"{probability_pct}% ，請強調會員專屬優惠。"
            )

        return ""

    # ------------------------------------------------------------------
    def _parse_ad_response(self, text: str) -> AdCreative:
        cleaned = self._strip_code_fence(text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GeminiUnavailableError(f"無法解析 Gemini 回傳的 JSON：{exc}") from exc

        if not isinstance(payload, dict):
            raise GeminiUnavailableError("Gemini 廣告文案格式不正確，預期 JSON 物件")

        headline = str(payload.get("headline", "")).strip()
        subheading = str(payload.get("subheading", "")).strip()
        highlight = str(payload.get("highlight", "")).strip()
        cta_value = str(payload.get("cta", "")).strip()
        if not any((headline, subheading, highlight)):
            raise GeminiUnavailableError("Gemini 廣告文案為空")
        return AdCreative(
            headline=headline,
            subheading=subheading,
            highlight=highlight,
            cta=cta_value or None,
        )

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text
