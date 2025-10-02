# backend/ai.py
"""Utilities for interacting with Vertex AI (Gemini/Imagen) to power the cloud-based MVP."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Any, Literal

from google.cloud import aiplatform
from google.api_core import exceptions as gapi_exceptions

# Vertex AI SDK (Generative AI)
from vertexai.generative_models import GenerativeModel, Part

# 影像生成功能命名空間在不同版位於 preview 或正式版；兩個都試。
try:
    from vertexai.preview.vision_models import ImageGenerationModel  # type: ignore
except Exception:  # pragma: no cover
    from vertexai.vision_models import ImageGenerationModel  # type: ignore

if TYPE_CHECKING:
    from .advertising import PurchaseInsights  # pragma: no cover
    from .prediction import PredictedItem  # pragma: no cover

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
    Wrapper around Vertex AI Gemini (text/vision) + Imagen (image generation).

    - 使用 GCE VM 的服務帳號（Application Default Credentials）。
    - 模型/專案/區域由環境變數配置（可帶預設）：
        GCP_PROJECT_ID, GCP_REGION
        ADGEN_TEXT_MODEL（如：gemini-1.5-pro 或 gemini-2.5-flash）
        ADGEN_VISION_MODEL（預設同 ADGEN_TEXT_MODEL）
        ADGEN_IMAGE_MODEL（如：imagen-3.0 或 imagen-4.0-generate-001）
    - 內建「區域自動回退」：第一優先用 GCP_REGION；若該區域找不到模型，改用 us-central1。
    """

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        region: Optional[str] = None,
        vision_model: Optional[str] = None,
        text_model: Optional[str] = None,
        timeout: float = 20.0,
    ) -> None:
        # ---- 基本設定（含預設值，避免少設參數時阻斷啟動）----
        self._project_cfg = project or os.getenv("GCP_PROJECT_ID", "esp32cam-472912")
        self._region_cfg = region or os.getenv("GCP_REGION", "asia-east1")

        # 文字/視覺模型（可分開設定，未提供視覺時沿用文字模型）
        default_text = os.getenv("ADGEN_TEXT_MODEL", "gemini-1.5-pro")
        self._text_model_name = text_model or default_text
        self._vision_model_name = vision_model or os.getenv("ADGEN_VISION_MODEL", default_text)

        # Imagen 影像模型（不一定要用到）
        self._image_model_name = os.getenv("ADGEN_IMAGE_MODEL", "imagen-3.0")

        self._timeout = timeout

        # 實際生效的區域（可能因回退而不同於 _region_cfg）
        self._active_region: Optional[str] = None

        # 實際模型實例
        self._vision_model: Optional[GenerativeModel] = None
        self._text_model: Optional[GenerativeModel] = None
        self._image_model: Optional[ImageGenerationModel] = None

        # 狀態
        self._inited: bool = False
        self._init_error: Optional[str] = None

        self._init_vertex_with_region_fallback()

    # ------------------------------------------------------------------
    def _try_init_for_region(self, region: str) -> bool:
        """嘗試在指定區域初始化 Vertex 與模型；若模型不存在，拋出 NotFound 讓上層決定是否回退。"""
        _LOGGER.info("Initializing Vertex AI in region %s for project %s", region, self._project_cfg)
        aiplatform.init(project=self._project_cfg, location=region)

        # 建立 Gemini（多模/文字）
        try:
            vision_model = GenerativeModel(self._vision_model_name)
            text_model = GenerativeModel(self._text_model_name)
        except gapi_exceptions.NotFound as nf:
            # 區域有啟但這個型號不存在 -> 讓上層做回退
            raise
        except Exception as exc:
            # 其他錯誤（權限/網路等）=> 不中斷回退，直接讓上層判斷
            raise

        # 建立 Imagen（可選）
        image_model: Optional[ImageGenerationModel]
        try:
            image_model = ImageGenerationModel.from_pretrained(self._image_model_name)
        except Exception as e:  # 影像模型非關鍵，記錄但不阻擋
            image_model = None
            _LOGGER.warning("Imagen model init failed in %s (%s): %s", region, self._image_model_name, e)

        # 一切 OK，寫回狀態
        self._active_region = region
        self._vision_model = vision_model
        self._text_model = text_model
        self._image_model = image_model
        self._inited = True
        self._init_error = None

        _LOGGER.info(
            "Gemini service initialised in %s with models %s (vision) / %s (text)",
            region,
            self._vision_model_name,
            self._text_model_name,
        )
        return True

    def _init_vertex_with_region_fallback(self) -> None:
        """優先用 GCP_REGION，若該區域模型不存在則回退 us-central1。"""
        # 依序嘗試：指定區域 -> us-central1（除非兩者相同）
        candidates = [self._region_cfg]
        if self._region_cfg != "us-central1":
            candidates.append("us-central1")

        last_error: Optional[Exception] = None
        for region in candidates:
            try:
                if self._try_init_for_region(region):
                    return
            except gapi_exceptions.NotFound as nf:
                last_error = nf
                _LOGGER.warning(
                    "Model not found in region %s (text=%s, vision=%s). Will try next region if any.",
                    region,
                    self._text_model_name,
                    self._vision_model_name,
                )
            except Exception as exc:  # 其他初始化失敗
                last_error = exc
                _LOGGER.exception("Vertex AI init failed in region %s: %s", region, exc)

        # 全部失敗
        self._inited = False
        self._init_error = str(last_error) if last_error else "Unknown init error"
        self._vision_model = None
        self._text_model = None
        self._image_model = None

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
        predicted_item: "PredictedItem | dict[str, Any] | None" = None,
        membership_status: "Literal['anonymous', 'prospect', 'member'] | None" = None,
    ) -> AdCreative:
        """Use Gemini Text to produce fresh advertising copy."""
        if not self.can_generate_ads or self._text_model is None:
            raise GeminiUnavailableError(self._init_error or "Gemini text model is not configured")

        purchase_list = list(purchases)
        prompt_payload = json.dumps(purchase_list, ensure_ascii=False)
        insight_summary = self._describe_insights(insights)
        insight_block = f"\n情境重點：{insight_summary}" if insight_summary else ""
        prediction_block = self._describe_prediction(predicted_item)
        membership_instruction = self._membership_instruction(membership_status)
        prompt = f"""
你是一位零售行銷 AI，目標是根據歷史消費紀錄產生一段動態廣告文案。
請閱讀以下 JSON 陣列描述的購買紀錄：{prompt_payload}
每筆包含 member_code、product_category、internal_item_code、item、purchased_at、unit_price、quantity、total_price 等欄位，金額為新台幣。
請輸出符合以下格式的 JSON（不要加任何額外說明）：
{{
  "headline": "...主標...",
  "subheading": "...副標...",
  "highlight": "...吸睛促購語..."
}}
文案語氣請友善、以繁體中文呈現，若沒有歷史紀錄，請推廣今日的主打商品。{insight_block}{prediction_block}
會員 ID：{member_id}
受眾提示：{membership_instruction}
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
    def _describe_prediction(
        self, predicted_item: "PredictedItem | dict[str, Any] | None"
    ) -> str:
        if not predicted_item:
            return ""

        if isinstance(predicted_item, dict):
            payload = predicted_item
        else:
            payload = {
                "product_code": getattr(predicted_item, "product_code", ""),
                "product_name": getattr(predicted_item, "product_name", ""),
                "category": getattr(predicted_item, "category", ""),
                "category_label": getattr(predicted_item, "category_label", ""),
                "price": getattr(predicted_item, "price", None),
                "probability": getattr(predicted_item, "probability", None),
                "probability_percent": getattr(predicted_item, "probability_percent", None),
            }

        return "\n主要推廣商品：" + json.dumps(payload, ensure_ascii=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _membership_instruction(
        membership_status: "Literal['anonymous', 'prospect', 'member'] | None"
    ) -> str:
        if membership_status == "member":
            return "顧客已是商場會員，請強調專屬折扣、點數回饋與升級禮遇。"
        if membership_status == "prospect":
            return "顧客尚未加入會員，請以加入會員可立即享有的優惠與專屬禮遇為主軸。"
        return "顧客身份未明，請維持熱情邀請並聚焦主打商品亮點。"

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
            "project": self._project_cfg,
            "configured_region": self._region_cfg,
            "active_region": self._active_region or "",
            "text_model": self._text_model_name,
            "vision_model": self._vision_model_name,
            "image_model": self._image_model_name,
            "error": self._init_error,
            "has_image_model": bool(self._image_model),
        }
