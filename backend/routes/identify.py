# -*- coding: utf-8 -*-
from __future__ import annotations
import io
from flask import Blueprint, request, jsonify

identify_bp = Blueprint("identify", __name__)

def _detect_member_id(image_bytes: bytes) -> str | None:
    """
    嘗試透過 backend.aws (Amazon Rekognition) 找到 member_id。
    若無此模組或比對不到，回 None。
    """
    try:
        from backend.aws import identify_face  # 你專案中如有提供
        result = identify_face(image_bytes)    # 預期回 dict | None，內含 member_id
        if isinstance(result, dict):
            return result.get("member_id") or result.get("face_id")
    except Exception:
        pass
    return None

@identify_bp.post("/identify")
def identify():
    """
    multipart/form-data:
      - file: 圖片
    或 application/json:
      - {"member_id":"ME0001"} / {"face_id":"MEMxxxx"}
    回傳：{"member_id":"ME0001"} 或 {"member_id":null}
    """
    # 1) JSON 直傳（方便自動化測）
    j = request.get_json(silent=True) or {}
    if "member_id" in j:
        return jsonify({"member_id": j["member_id"]})
    if "face_id" in j:
        # 若你 members 表以 face_id 對應 member_id，可在這裡轉換；暫時原樣回
        return jsonify({"member_id": j["face_id"]})

    # 2) 上傳圖片
    f = request.files.get("file")
    if not f:
        return jsonify({"member_id": None, "error": "no file"}), 400
    image_bytes = f.read()
    member_id = _detect_member_id(image_bytes)
    return jsonify({"member_id": member_id})
