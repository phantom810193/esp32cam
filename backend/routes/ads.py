"""HTTP routes for composing and caching advertisement creatives."""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from ..ad_pipeline import compose_member_ad


ads_blueprint = Blueprint("ads", __name__)


@ads_blueprint.get("/ad/<member_id>/compose")
def compose_ad_endpoint(member_id: str):
    force_flag = str(request.args.get("force", "0")).lower() in {"1", "true", "yes"}
    try:
        payload = compose_member_ad(member_id, force=force_flag)
    except FileNotFoundError:
        return jsonify({"error": "底圖不存在或未配置"}), 404
    except Exception as exc:  # pylint: disable=broad-except
        current_app.logger.exception("Failed to compose advertisement for %s", member_id)
        return jsonify({"error": str(exc)}), 500

    return jsonify(payload)
