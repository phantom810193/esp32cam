# backend/routes/adgen.py
"""Vertex AI powered advertisement generation endpoints."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any, Dict
from uuid import uuid4

from flask import Blueprint, jsonify, request, url_for
from google.cloud import storage

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# 兼容新版與舊版套件的匯入位置
try:
    # 新版
    from vertexai.vision_models import ImageGenerationModel  # type: ignore
except Exception:  # pragma: no cover
    # 舊版（preview 路徑）
    from vertexai.preview.vision_models import ImageGenerationModel  # type: ignore

adgen_blueprint = Blueprint("adgen", __name__)

# ── 可由環境變數覆寫的參數 ───────────────────────────────────────────────
_TEXT_MODEL_NAME = os.environ.get("VERTEX_TEXT_MODEL", "gemini-2.5-flash")
_IMAGE_MODEL_NAME = os.environ.get("VERTEX_IMAGE_MODEL", "imagen-3.0-generate-001")
_IMAGE_SIZE = os.environ.get("VERTEX_IMAGE_SIZE", "1080x1080")  # e.g. "1080x1080"
_DEFAULT_REGION = os.environ.get("GCP_REGION", "asia-east1")

# ── 內部單例 ─────────────────────────────────────────────────────────────
_vertex_initialised = False
_text_model_instance: GenerativeModel | None = None
_image_model_instance: ImageGenerationModel | None = None


def _ensure_vertex_initialised() -> tuple[str, str]:
    """Lazy-init Vertex AI with project/region."""
    global _vertex_initialised
    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        raise RuntimeError("GCP_PROJECT_ID environment variable is required for Vertex AI")

    region = os.environ.get("GCP_REGION", _DEFAULT_REGION)
    if not _vertex_initialised:
        vertexai.init(project=project_id, location=region)
        _vertex_initialised = True
        logging.info("Vertex AI initialized (project=%s, region=%s)", project_id, region)
    return project_id, region


def _get_text_model() -> GenerativeModel:
    """Return an instantiated GenerativeModel (NOT a class/function)."""
    global _text_model_instance
    if _text_model_instance is None:
        _ensure_vertex_initialised()
        _text_model_instance = GenerativeModel(_TEXT_MODEL_NAME)
    return _text_model_instance


def _get_image_model() -> ImageGenerationModel:
    """Return an instantiated ImageGenerationModel."""
    global _image_model_instance
    if _image_model_instance is None:
        _ensure_vertex_initialised()
        _image_model_instance = ImageGenerationModel.from_pretrained(_IMAGE_MODEL_NAME)
    return _image_model_instance


# ── Prompt helpers ───────────────────────────────────────────────────────
def _build_text_prompt(sku: str, member_profile: Dict[str, Any]) -> str:
    persona = json.dumps(member_profile or {}, ensure_ascii=False)
    return (
        "你是一位商場的廣告行銷策劃師，正在為會員製作 1:1 的推播文案。\n"
        "請以 JSON 格式輸出，鍵值必須包含：title、subline、cta。\n"
        "文案限制：繁體中文、適合數位看板、每段約 6~18 字、具行動力。\n"
        f"SKU：{sku}\n"
        f"會員資料（JSON）：{persona}\n"
        "若資訊不足，以吸引新客為目標設計文案，但仍須輸出 JSON。"
    )


def _build_image_prompt(sku: str, copy: Dict[str, str], member_profile: Dict[str, Any]) -> str:
    persona_bits = ", ".join(f"{k}:{v}" for k, v in (member_profile or {}).items() if v)
    headline = (copy or {}).get("title") or sku
    description = (copy or {}).get("subline") or ""
    prompt = (
        "生成一張 1:1 比例、明亮吸睛的商場活動宣傳海報；"
        "具高端百貨視覺風格、畫面乾淨，主視覺聚焦於商品/服務意象，避免浮水印與品牌 logo。"
        f" 主標題靈感：{headline}。副標語靈感：{description}。"
        f" 針對品項 {sku}。"
    )
    if persona_bits:
        prompt += f" 目標客群線索：{persona_bits}。"
    return prompt


# ── Model wrappers ───────────────────────────────────────────────────────
def _extract_text(resp: Any) -> str:
    """Robustly extract text from Vertex responses across SDK versions."""
    if not resp:
        return ""
    # 新版通常有 .text
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    # 或從 candidates 結構中取
    try:
        cands = getattr(resp, "candidates", None) or []
        if cands and getattr(cands[0], "content", None):
            parts = getattr(cands[0].content, "parts", None) or []
            # 找第一段文字
            for p in parts:
                val = getattr(p, "text", None)
                if isinstance(val, str) and val.strip():
                    return val
    except Exception:  # pragma: no cover
        pass
    return ""


def _generate_copy(sku: str, member_profile: Dict[str, Any]) -> Dict[str, str]:
    model = _get_text_model()
    prompt = _build_text_prompt(sku, member_profile)
    resp = model.generate_content(
        prompt,
        generation_config=GenerationConfig(temperature=0.6, max_output_tokens=256),
    )
    text = _extract_text(resp)
    try:
        parsed = json.loads(text)
        return {
            "title": str(parsed.get("title") or f"{sku} 精選活動").strip(),
            "subline": str(parsed.get("subline") or "加入會員即可領取驚喜好禮").strip(),
            "cta": str(parsed.get("cta") or "立即參加").strip(),
        }
    except Exception:
        logging.warning("Gemini response not JSON, fallback copy. raw=%r", text[:200])
        return {
            "title": f"{sku} 限時禮遇",
            "subline": "專屬禮遇即刻開啟，加入會員解鎖驚喜。",
            "cta": "立即了解活動",
        }


def _generate_image_bytes(sku: str, copy: Dict[str, str], member_profile: Dict[str, Any]) -> bytes:
    model = _get_image_model()
    prompt = _build_image_prompt(sku, copy, member_profile)

    # 嘗試使用 image_size（如 1080x1080），若舊版 SDK 不支援則退回 aspect_ratio
    kwargs: Dict[str, Any] = {"prompt": prompt, "number_of_images": 1}
    if _IMAGE_SIZE:
        kwargs["image_size"] = _IMAGE_SIZE
    else:
        kwargs["aspect_ratio"] = "1:1"

    try:
        result = model.generate_images(**kwargs)
    except TypeError:
        kwargs.pop("image_size", None)
        kwargs["aspect_ratio"] = "1:1"
        result = model.generate_images(**kwargs)

    if not getattr(result, "images", None):
        raise RuntimeError("Vertex Images returned no images")

    img = result.images[0]
    # 兼容不同 SDK 欄位
    if hasattr(img, "as_bytes") and callable(img.as_bytes):
        return img.as_bytes()
    if hasattr(img, "_image_bytes"):
        return img._image_bytes  # type: ignore[attr-defined]
    if hasattr(img, "bytes_data"):
        return img.bytes_data  # type: ignore[attr-defined]
    raise RuntimeError("Unsupported image object from Vertex Images")


def _upload_to_gcs(image_bytes: bytes, sku: str) -> str:
    bucket_name = os.environ.get("ASSET_BUCKET")
    if not bucket_name:
        raise RuntimeError("ASSET_BUCKET environment variable is required")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    object_name = f"adgen/{sku}/{datetime.utcnow():%Y%m%d}/{uuid4().hex}.png"
    blob = bucket.blob(object_name)
    blob.upload_from_string(image_bytes, content_type="image/png")

    expires = datetime.utcnow() + timedelta(hours=12)
    return blob.generate_signed_url(version="v4", expiration=expires, method="GET")


def _fallback_payload(sku: str) -> Dict[str, str]:
    image_url = url_for("static", filename="images/ads/ME0000.jpg", _external=True)
    return {
        "title": f"{sku} 推廣活動",
        "subline": "系統暫時使用預設素材，仍歡迎洽詢現場人員。",
        "cta": "現場了解更多",
        "image_url": image_url,
    }


# ── Route ────────────────────────────────────────────────────────────────
@adgen_blueprint.post("/adgen")
def create_generated_ad():
    """Generate an advertisement asset via Vertex AI (Gemini + Imagen)."""
    payload = request.get_json(silent=True) or {}
    sku = str(payload.get("sku") or "").strip()
    if not sku:
        return jsonify({"error": "sku is required"}), 400

    member_profile = payload.get("member_profile") or {}
    if not isinstance(member_profile, dict):
        member_profile = {}

    started_at = perf_counter()
    try:
        _ensure_vertex_initialised()

        t0 = perf_counter()
        copy = _generate_copy(sku, member_profile)
        t_copy = perf_counter() - t0

        t1 = perf_counter()
        image_bytes = _generate_image_bytes(sku, copy, member_profile)
        t_img = perf_counter() - t1

        t2 = perf_counter()
        image_url = _upload_to_gcs(image_bytes, sku)
        t_up = perf_counter() - t2

    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Ad generation pipeline failed for SKU %s: %s", sku, exc)
        return jsonify(_fallback_payload(sku)), 503

    total = perf_counter() - started_at
    logging.info(
        "Ad generation for %s completed in %.2fs (copy=%.2fs, image=%.2fs, upload=%.2fs)",
        sku, total, t_copy, t_img, t_up,
    )
    if total > 10:
        logging.warning("Ad generation for %s exceeded 10s SLA (%.2fs)", sku, total)

    return jsonify(
        {
            "title": copy["title"],
            "subline": copy["subline"],
            "cta": copy["cta"],
            "image_url": image_url,
        }
    )
