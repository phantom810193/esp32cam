"""Vertex AI powered advertisement generation endpoints."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any, Dict
from uuid import uuid4

from flask import (
    Blueprint,
    current_app,
    jsonify,
    request,
    url_for,
)
from flask import has_app_context, has_request_context
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel

adgen_blueprint = Blueprint("adgen", __name__)

_TEXT_MODEL_NAME = os.environ.get("VERTEX_TEXT_MODEL", "gemini-2.5-pro")
_IMAGE_MODEL_NAME = os.environ.get("VERTEX_IMAGE_MODEL", "imagen-3.0-generate-001")
_IMAGE_SIZE = os.environ.get("VERTEX_IMAGE_SIZE", "1080x1080")
_DEFAULT_REGION = os.environ.get("GCP_REGION", "asia-east1")

_vertex_initialised = False
_text_model_instance: GenerativeModel | None = None
_image_model_instance: ImageGenerationModel | None = None


def _ensure_vertex_initialised() -> tuple[str, str]:
    global _vertex_initialised
    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        raise RuntimeError("GCP_PROJECT_ID environment variable is required for Vertex AI")

    region = os.environ.get("GCP_REGION", _DEFAULT_REGION)
    if not _vertex_initialised:
        vertexai.init(project=project_id, location=region)
        _vertex_initialised = True
    return project_id, region


def _get_text_model() -> GenerativeModel:
    global _text_model_instance
    if _text_model_instance is None:
        _ensure_vertex_initialised()
        _text_model_instance = GenerativeModel(_TEXT_MODEL_NAME)
    return _text_model_instance


def _get_image_model() -> ImageGenerationModel:
    global _image_model_instance
    if _image_model_instance is None:
        _ensure_vertex_initialised()
        _image_model_instance = ImageGenerationModel.from_pretrained(_IMAGE_MODEL_NAME)
    return _image_model_instance


def _build_text_prompt(sku: str, member_profile: Dict[str, Any]) -> str:
    persona = json.dumps(member_profile, ensure_ascii=False) if member_profile else "{}"
    return (
        "你是一位商場的廣告行銷策劃師，正在為會員製作 1:1 的推播文案。\n"
        "請針對下列資訊，輸出一段 JSON，鍵值必須包含 title、subline、cta 三項，"
        "內容需使用繁體中文且適合數位看板，保持 6~18 個字且富有行動力。\n"
        "\nSKU: "
        f"{sku}\n"
        "會員資料(JSON)："
        f"{persona}\n"
        "若資訊不足，請以吸引新客為目標設計文案，仍須輸出 JSON。"
    )


def _build_image_prompt(sku: str, copy: Dict[str, str], member_profile: Dict[str, Any]) -> str:
    persona_bits = ", ".join(
        f"{key}:{value}" for key, value in member_profile.items() if value
    )
    headline = copy.get("title") or sku
    description = copy.get("subline") or ""
    prompt = (
        "生成一張 1:1 比例、充滿活力的商場活動宣傳海報。"
        "畫面需加入代表商品或服務的視覺元素，避免文字擁擠。"
        "以高端百貨風格呈現，色彩明亮吸睛，適合 1080x1080 LED 看板。"
        f"主標題靈感：{headline}. 副標語靈感：{description}."
    )
    if persona_bits:
        prompt += f" 目標客群特徵：{persona_bits}."
    prompt += f" 針對品項 {sku} 打造，避免浮水印與品牌 logo。"
    return prompt


def _generate_copy(sku: str, member_profile: Dict[str, Any]) -> Dict[str, str]:
    model = _get_text_model()
    prompt = _build_text_prompt(sku, member_profile)
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(temperature=0.6, max_output_tokens=256),
    )
    text = getattr(response, "text", None) or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logging.warning("Vertex AI Gemini response is not JSON, fallback to template: %s", text)
        return {
            "title": f"{sku} 限時禮遇",
            "subline": "專屬禮遇即刻開啟，加入會員解鎖驚喜。",
            "cta": "立即了解活動"
        }
    result = {
        "title": str(parsed.get("title") or f"{sku} 精選活動").strip(),
        "subline": str(parsed.get("subline") or "加入會員即可領取驚喜好禮").strip(),
        "cta": str(parsed.get("cta") or "立即參加").strip(),
    }
    return result


def _generate_image_bytes(sku: str, copy: Dict[str, str], member_profile: Dict[str, Any]) -> bytes:
    model = _get_image_model()
    prompt = _build_image_prompt(sku, copy, member_profile)
    request_kwargs = {"prompt": prompt, "number_of_images": 1}
    if _IMAGE_SIZE:
        request_kwargs["image_size"] = _IMAGE_SIZE
    else:
        request_kwargs["aspect_ratio"] = "1:1"

    try:
        result = model.generate_images(**request_kwargs)
    except TypeError:
        # 舊版 SDK 不支援 image_size，退回使用 aspect_ratio 參數
        request_kwargs.pop("image_size", None)
        request_kwargs["aspect_ratio"] = "1:1"
        result = model.generate_images(**request_kwargs)
    if not result.images:
        raise RuntimeError("Vertex AI Images did not return any image data")
    image = result.images[0]
    if hasattr(image, "as_bytes"):
        return image.as_bytes()
    if hasattr(image, "bytes_data"):
        return image.bytes_data
    raise RuntimeError("Unsupported image response format from Vertex AI Images")


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
    return blob.generate_signed_url(expiration=expires, method="GET")


def _static_asset_url(filename: str) -> str:
    """Return a static asset URL that works inside or outside a request context."""

    if has_request_context():
        return url_for("static", filename=filename, _external=True)

    # In CLI / background contexts we might not have a request, but the app
    # context can still be available. Build a best-effort absolute path.
    static_url_path = "/static"
    if has_app_context():
        static_url_path = (current_app.static_url_path or "/static").rstrip("/")

    return f"{static_url_path}/{filename.lstrip('/')}"


def _fallback_payload(sku: str) -> Dict[str, str]:
    image_url = _static_asset_url("images/ads/ME0000.jpg")
    return {
        "title": f"{sku} 推廣活動",
        "subline": "系統暫時使用預設素材，仍歡迎洽詢現場人員。",
        "cta": "現場了解更多",
        "image_url": image_url,
    }


def generate_vertex_ad(sku: str, member_profile: Dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the Vertex AI copy + image pipeline and return payload with timings."""

    if not sku:
        raise ValueError("SKU must be provided for Vertex AI ad generation")

    member_profile = member_profile or {}

    started_at = perf_counter()

    _ensure_vertex_initialised()

    copy_started_at = perf_counter()
    copy = _generate_copy(sku, member_profile)
    copy_duration = perf_counter() - copy_started_at

    image_started_at = perf_counter()
    image_bytes = _generate_image_bytes(sku, copy, member_profile)
    image_duration = perf_counter() - image_started_at

    upload_started_at = perf_counter()
    image_url = _upload_to_gcs(image_bytes, sku)
    upload_duration = perf_counter() - upload_started_at

    total_duration = perf_counter() - started_at

    logging.info(
        "Vertex AI ad for %s completed in %.2fs (copy=%.2fs, image=%.2fs, upload=%.2fs)",
        sku,
        total_duration,
        copy_duration,
        image_duration,
        upload_duration,
    )

    if total_duration > 10:
        logging.warning("Ad generation for %s exceeded 10s SLA (%.2fs)", sku, total_duration)

    return {
        "title": copy["title"],
        "subline": copy["subline"],
        "cta": copy["cta"],
        "image_url": image_url,
        "timings": {
            "total": total_duration,
            "copy": copy_duration,
            "image": image_duration,
            "upload": upload_duration,
        },
    }


@adgen_blueprint.post("/adgen")
def create_generated_ad():
    """Generate an advertisement asset via Vertex AI Gemini + Imagen."""

    payload = request.get_json(silent=True) or {}
    sku = str(payload.get("sku") or "").strip()
    if not sku:
        return jsonify({"error": "sku is required"}), 400

    member_profile = payload.get("member_profile") or {}
    if not isinstance(member_profile, dict):
        member_profile = {}

    try:
        result = generate_vertex_ad(sku, member_profile)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Ad generation pipeline failed for SKU %s: %s", sku, exc)
        fallback = _fallback_payload(sku)
        return jsonify(fallback), 503

    return jsonify(result)
