"""Utilities for interacting with Gemini to power the cloud-based MVP."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

try:  # pragma: no cover - optional dependency for offline development
    import google.generativeai as genai  # type: ignore
    from google.api_core import exceptions as google_exceptions  # type: ignore
except Exception:  # pragma: no cover - guard against missing dependencies
    genai = None  # type: ignore
    google_exceptions = None  # type: ignore

_LOGGER = logging.getLogger(__name__)


class GeminiUnavailableError(RuntimeError):
    """Raised when Gemini features are requested but not configured."""


@dataclass
class AdCreative:
    """Structured ad copy returned by the AI generator."""

    headline: str
    subheading: str
    highlight: str


class GeminiService:
    """Thin wrapper around the Gemini Vision + Text APIs used by the MVP."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        vision_model: str = "gemini-1.5-flash",
        text_model: str = "gemini-1.5-flash",
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._vision_model_name = vision_model
        self._text_model_name = text_model
        self._timeout = timeout
        self._vision_model = None
        self._text_model = None

        if self.api_key and genai is not None:
            genai.configure(api_key=self.api_key)
            try:
                self._vision_model = genai.GenerativeModel(self._vision_model_name)
                self._text_model = genai.GenerativeModel(self._text_model_name)
                _LOGGER.info(
                    "Gemini service initialised with models %s (vision) / %s (text)",
                    self._vision_model_name,
                    self._text_model_name,
                )
            except Exception as exc:  # pragma: no cover - configuration failure
                _LOGGER.error("Failed to initialise Gemini models: %s", exc)
                self._vision_model = None
                self._text_model = None
        elif self.api_key and genai is None:
            _LOGGER.warning(
                "google-generativeai is not installed. Install it to enable Gemini features."
            )
        else:
            _LOGGER.info("GEMINI_API_KEY not provided. Gemini-powered features are disabled.")

    # ------------------------------------------------------------------
    @property
    def can_describe_faces(self) -> bool:
        return self._vision_model is not None

    @property
    def can_generate_ads(self) -> bool:
        return self._text_model is not None

    # ------------------------------------------------------------------
    def describe_face(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """Return a stable textual signature for the supplied face image."""

        if not self.can_describe_faces:
            raise GeminiUnavailableError("Gemini Vision model is not configured")

        assert self._vision_model is not None  # for the type checker
        prompt = (
            "你是一個用於匿名會員辨識的生物特徵助手。請用 3~4 個重點描述此照片中人物"\
            "的臉部特徵（例如：髮型、配件、年齡層、明顯特徵），"\
            "並輸出為單行中文短句，避免包含任何個資或主觀評價。"
        )

        content: Sequence[object] = (
            {"mime_type": mime_type or "image/jpeg", "data": image_bytes},
            prompt,
        )
        try:
            response = self._vision_model.generate_content(
                content,
                request_options={"timeout": self._timeout},
            )
        except Exception as exc:  # pragma: no cover - network/runtime errors
            if google_exceptions and isinstance(exc, google_exceptions.GoogleAPICallError):
                raise GeminiUnavailableError(f"Gemini Vision API error: {exc}") from exc
            raise GeminiUnavailableError(f"Gemini Vision failed: {exc}") from exc

        description = (response.text or "").strip()
        if not description:
            raise GeminiUnavailableError("Gemini Vision returned an empty description")
        _LOGGER.debug("Gemini Vision signature: %s", description)
        return description

    # ------------------------------------------------------------------
    def generate_ad_copy(self, member_id: str, purchases: Iterable[dict[str, object]]) -> AdCreative:
        """Use Gemini Text to produce fresh advertising copy."""

        if not self.can_generate_ads:
            raise GeminiUnavailableError("Gemini text model is not configured")

        assert self._text_model is not None  # for the type checker
        purchase_list = list(purchases)
        prompt_payload = json.dumps(purchase_list, ensure_ascii=False)
        prompt = f"""
你是一位零售行銷 AI，目標是根據歷史消費紀錄產生一段動態廣告文案。
請閱讀以下 JSON 陣列描述的購買紀錄：{prompt_payload}
每筆包含 member_code、item、purchased_at、unit_price、quantity、total_price 等欄位，金額為新台幣。
請輸出符合以下格式的 JSON（不要加任何額外說明）：
{{
  "headline": "...主標...",
  "subheading": "...副標...",
  "highlight": "...吸睛促購語..."
}}
文案語氣請友善、以繁體中文呈現，若沒有歷史紀錄，請推廣今日的主打商品。
會員 ID：{member_id}
"""
        try:
            response = self._text_model.generate_content(
                prompt,
                request_options={"timeout": self._timeout},
            )
        except Exception as exc:  # pragma: no cover - network/runtime errors
            if google_exceptions and isinstance(exc, google_exceptions.GoogleAPICallError):
                raise GeminiUnavailableError(f"Gemini Text API error: {exc}") from exc
            raise GeminiUnavailableError(f"Gemini Text generation failed: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise GeminiUnavailableError("Gemini Text returned an empty response")
        return self._parse_ad_response(text)

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
        if not any((headline, subheading, highlight)):
            raise GeminiUnavailableError("Gemini 廣告文案為空")
        return AdCreative(headline=headline, subheading=subheading, highlight=highlight)

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # drop first fence
            lines = lines[1:]
            # drop closing fence if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text

