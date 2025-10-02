"""Vertex AI powered advertisement generation endpoints."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict
from uuid import uuid4

from flask import Blueprint, jsonify, request, url_for
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel

adgen_blueprint = Blueprint("adgen", __name__)

_TEXT_MODEL_NAME = os.environ.get("VERTEX_TEXT_MODEL", "gemini-1.5-pro")
_IMAGE_MODEL_NAME = os.environ.get("VERTEX_IMAGE_MODEL", "imagegeneration@002")
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
    result = model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio="1:1")
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


def _fallback_payload(sku: str) -> Dict[str, str]:
    image_url = url_for("static", filename="images/ads/ME0000.jpg", _external=True)
    return {
        "title": f"{sku} 推廣活動",
        "subline": "系統暫時使用預設素材，仍歡迎洽詢現場人員。",
        "cta": "現場了解更多",
        "image_url": image_url,
    }


@adgen_blueprint.post("/adgen")
def create_generated_ad():
    payload = request.get_json(silent=True) or {}
    sku = str(payload.get("sku") or "").strip()
    if not sku:
        return jsonify({"error": "sku is required"}), 400

    member_profile = payload.get("member_profile") or {}
    if not isinstance(member_profile, dict):
        member_profile = {}

    try:
        _ensure_vertex_initialised()
        copy = _generate_copy(sku, member_profile)
        image_bytes = _generate_image_bytes(sku, copy, member_profile)
        image_url = _upload_to_gcs(image_bytes, sku)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Ad generation pipeline failed for SKU %s: %s", sku, exc)
        fallback = _fallback_payload(sku)
        return jsonify(fallback), 503

    return jsonify({
        "title": copy["title"],
        "subline": copy["subline"],
        "cta": copy["cta"],
        "image_url": image_url,
    })
