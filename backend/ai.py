# backend/ai.py
"""Utilities for interacting with Vertex AI (Gemini/Imagen) to power the cloud-based MVP."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from google.cloud import aiplatform

# Vertex AI SDK (Generative AI)
from vertexai.generative_models import GenerativeModel, Part

# 影像生成功能命名空間在不同版位於 preview 或正式版，兩個都試。
try:
    from vertexai.preview.vision_models import ImageGenerationModel  # type: ignore
except Exception:  # pragma: no cover
    from vertexai.vision_models import ImageGenerationModel  # type: ignore

if TYPE_CHECKING:
    from .advertising import PurchaseInsights  # pragma: no cover

# 與你原本的日誌名稱保持一致，方便追蹤
_LOGGER = logging.getLogger("backend.ai")


class GeminiUnavailableError(RuntimeError):
    """Raised when Vertex AI generative features are requested but unavailable."""


@dataclass
class AdCreative:
    """Structured ad copy returned by the AI generator."""

    headline: str
    subheading: str
    highlight: str


class GeminiService:
    """
    Thin wrapper around Vertex AI Gemini (text/vision) + Imagen (image generation).

    差異重點 vs. 舊版：
    - 不再使用 google-generativeai 與 GEMINI_API_KEY。
    - 透過 GCE VM 的服務帳號（Application Default Credentials）存取。
    - 模型、專案、區域皆由環境變數配置：
        GCP_PROJECT_ID, GCP_REGION
        ADGEN_TEXT_MODEL (ex: gemini-2.5-flash 或 gemini-2.5-pro)
        ADGEN_IMAGE_MODEL (ex: imagen-3.0-generate-001 或 imagen-4.0-generate-001)
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        region: str | None = None,
        vision_model: str | None = None,
        text_model: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self._project = project or os.getenv("GCP_PROJECT_ID", "esp32cam-472912")
        self._region = region or os.getenv("GCP_REGION", "us-central1")
        self._vision_model_name = vision_model or os.getenv("ADGEN_TEXT_MODEL", "gemini-2.5-flash")
        # 上面沿用同一套 Gemini 2.5，能吃影像也能產文案；若你想分開，可另外加 ADGEN_VISION_MODEL
        self._text_model_name = text_model or os.getenv("ADGEN_TEXT_MODEL", "gemini-2.5-flash")
        self._image_model_name = os.getenv("ADGEN_IMAGE_MODEL", "imagen-3.0-generate-001")
        self._timeout = timeout

        self._vision_model: GenerativeModel | None = None
        self._text_model: GenerativeModel | None = None
        self._image_model: ImageGenerationModel | None = None

        self._inited: bool = False
        self._init_error: str | None = None

        self._init_vertex()

    # ------------------------------------------------------------------
    def _init_vertex(self) -> None:
        if self._inited:
            return
        try:
            aiplatform.init(project=self._project, location=self._region)
            # Gemini（多模/文字）
            self._vision_model = GenerativeModel(self._vision_model_name)
            self._text_model = GenerativeModel(self._text_model_name)
            # Imagen（生成圖片，可選）
            try:
                self._image_model = ImageGenerationModel.from_pretrained(self._image_model_name)
            except Exception as e:  # 影像模型非必要，失敗就記錄
                self._image_model = None
                _LOGGER.warning("Imagen model init failed (%s): %s", self._image_model_name, e)

            self._inited = True
            _LOGGER.info(
                "Gemini service initialised with models %s (vision) / %s (text)",
                self._vision_model_name,
                self._text_model_name,
            )
        except Exception as exc:
            self._init_error = str(exc)
            self._inited = False
            self._vision_model = None
            self._text_model = None
            self._image_model = None
            _LOGGER.exception("Vertex AI init failed: %s", exc)

    # ------------------------------------------------------------------
    @property
    def can_describe_faces(self) -> bool:
        return self._vision_model is not None and self._inited and not self._init_error

    @property
    def can_generate_ads(self) -> bool:
        return self._text_model is not None and self._inited and not self._init_error

    # ------------------------------------------------------------------
    def describe_face(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """Return a stable textual signature for the supplied face image using Gemini Vision."""
        if not self.can_describe_faces or self._vision_model is None:
            raise GeminiUnavailableError(self._init_error or "Gemini Vision model is not configured")

        prompt = (
            "你是一個用於匿名會員辨識的生物特徵助手。請用 3~4 個重點描述此照片中人物"
            "的臉部特徵（例如：髮型、配件、年齡層、明顯特徵），"
            "並輸出為單行中文短句，避免包含任何個資或主觀評價。"
        )
        try:
            parts = [
                Part.from_data(mime_type=mime_type or "image/jpeg", data=image_bytes),
                prompt,
            ]
            resp = self._vision_model.generate_content(
                parts,
                request_options={"timeout": self._timeout},
            )
        except Exception as exc:
            raise GeminiUnavailableError(f"Gemini Vision failed: {exc}") from exc

        description = (getattr(resp, "text", None) or "").strip()
        if not description:
            raise GeminiUnavailableError("Gemini Vision returned an empty description")
        _LOGGER.debug("Gemini Vision signature: %s", description)
        return description

    # ------------------------------------------------------------------
    def generate_ad_copy(
        self,
        member_id: str,
        purchases: Iterable[dict[str, object]],
        *,
        insights: "PurchaseInsights | None" = None,
    ) -> AdCreative:
        """Use Gemini Text to produce fresh advertising copy."""
        if not self.can_generate_ads or self._text_model is None:
            raise GeminiUnavailableError(self._init_error or "Gemini text model is not configured")

        purchase_list = list(purchases)
        prompt_payload = json.dumps(purchase_list, ensure_ascii=False)
        insight_summary = self._describe_insights(insights)
        insight_block = f"\n情境重點：{insight_summary}" if insight_summary else ""
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
文案語氣請友善、以繁體中文呈現，若沒有歷史紀錄，請推廣今日的主打商品。{insight_block}
會員 ID：{member_id}
"""
        try:
            resp = self._text_model.generate_content(
                prompt,
                request_options={"timeout": self._timeout},
            )
        except Exception as exc:
            raise GeminiUnavailableError(f"Gemini Text generation failed: {exc}") from exc

        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            raise GeminiUnavailableError("Gemini Text returned an empty response")
        return self._parse_ad_response(text)

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
        if not any((headline, subheading, highlight)):
            raise GeminiUnavailableError("Gemini 廣告文案為空")
        return AdCreative(headline=headline, subheading=subheading, highlight=highlight)

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

    # ------------------------------------------------------------------
    def health_probe(self) -> dict:
        """Expose init + model info to /healthz."""
        return {
            "vertexai": "initialized" if self._inited and not self._init_error else "init_failed",
            "project": self._project,
            "region": self._region,
            "text_model": self._text_model_name,
            "vision_model": self._vision_model_name,
            "image_model": self._image_model_name,
            "error": self._init_error,
            "has_image_model": bool(self._image_model),
        }
