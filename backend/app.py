"""Flask backend for the ESP32-CAM retail advertising MVP."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

from flask import Flask, jsonify, render_template, request, url_for

from .advertising import build_ad_context
from .ai import AzureFaceError, AzureFaceService
from .database import Database
from .recognizer import FaceEncoding, FaceRecognizer

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "mvp.sqlite3"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["JSON_AS_ASCII"] = False

face_service = AzureFaceService()
recognizer = FaceRecognizer(face_service)
database = Database(DB_PATH)
database.ensure_demo_data()


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.route("/person-group", methods=["GET", "POST"])
def person_group_trainer():
    """Render a simple UI to upload faces for Azure Person Group training."""

    context = {
        "azure_enabled": face_service.can_manage_person_group,
        "azure_configured": face_service.can_describe_faces,
        "azure_person_group_error": getattr(face_service, "person_group_error", None),
        "member_id": "",
        "results": [],
        "errors": [],
        "azure_person_id": None,
        "faces_registered": 0,
        "created_member": False,
    }

    if request.method == "POST":
        member_id = (request.form.get("member_id") or "").strip()
        context["member_id"] = member_id

        if not member_id:
            context["errors"].append("請輸入會員 ID。")

        uploads = []
        if request.files:
            for field in ("images", "image", "files", "photos"):
                for storage in request.files.getlist(field):
                    data = storage.read()
                    if not data:
                        continue
                    uploads.append(
                        {
                            "name": storage.filename,
                            "bytes": data,
                            "mimetype": storage.mimetype or request.mimetype or "image/jpeg",
                        }
                    )
        if not uploads:
            context["errors"].append("請選擇至少一張要上傳的照片。")

        if not face_service.can_manage_person_group:
            if face_service.person_group_error:
                context["errors"].append(face_service.person_group_error)
            else:
                context["errors"].append(
                    "Azure Face Person Group 功能未啟用，請設定 AZURE_FACE_ENDPOINT / AZURE_FACE_KEY。"
                )

        azure_person_id: str | None = None
        existing_encoding: FaceEncoding | None = None
        created_person = False
        persisted_face_ids: list[str] = []

        if not context["errors"]:
            existing_encoding = database.get_member_encoding(member_id)
            if existing_encoding:
                azure_person_id = existing_encoding.azure_person_id
                if not existing_encoding.azure_person_name:
                    existing_encoding.azure_person_name = member_id

            if azure_person_id is None:
                try:
                    azure_person_id = face_service.find_person_id_by_name(member_id)
                except AzureFaceError as exc:
                    context["errors"].append(f"查詢 Azure Person 失敗：{exc}")

        encoding: FaceEncoding | None = existing_encoding

        if not context["errors"] and encoding is None and uploads:
            primary_upload = uploads[0]
            try:
                encoding = recognizer.encode(primary_upload["bytes"], mime_type=primary_upload["mimetype"])
            except ValueError as exc:
                context["errors"].append(f"無法產生臉部特徵：{exc}")

        start_index = 0
        if not context["errors"] and uploads:
            primary_upload = uploads[0]
            if azure_person_id is None:
                try:
                    azure_person_id = face_service.register_person(
                        member_id,
                        primary_upload["bytes"],
                        user_data=member_id,
                    )
                except AzureFaceError as exc:
                    context["errors"].append(f"建立 Azure Person 失敗：{exc}")
                else:
                    context["faces_registered"] += 1
                    context["results"].append(primary_upload["name"] or "face-1.jpg")
                    start_index = 1
                    created_person = True
                    if encoding is None:
                        vector = FaceRecognizer._vector_from_signature(azure_person_id)
                        encoding = FaceEncoding(
                            vector=vector,
                            face_description=member_id,
                            source="azure-person-group",
                        )
                if face_service.can_use_face_list:
                    try:
                        persisted_id = face_service.add_face_to_face_list(
                            member_id,
                            primary_upload["bytes"],
                            user_data=member_id,
                        )
                    except AzureFaceError as exc:
                        context["errors"].append(f"{primary_upload['name'] or 'face-1.jpg'} 加入 Face List 失敗：{exc}")
                    else:
                        persisted_face_ids.append(persisted_id)

            if azure_person_id:
                for upload in uploads[start_index:]:
                    try:
                        face_service.add_face_to_person(azure_person_id, upload["bytes"])
                    except AzureFaceError as exc:
                        context["errors"].append(
                            f"{upload['name'] or '照片'} 加入 Person Group 時失敗：{exc}"
                        )
                    else:
                        context["faces_registered"] += 1
                        context["results"].append(upload["name"] or "face.jpg")
                    if face_service.can_use_face_list:
                        try:
                            persisted_id = face_service.add_face_to_face_list(
                                member_id,
                                upload["bytes"],
                                user_data=member_id,
                            )
                        except AzureFaceError as exc:
                            context["errors"].append(
                                f"{upload['name'] or '照片'} 加入 Face List 時失敗：{exc}"
                            )
                        else:
                            persisted_face_ids.append(persisted_id)

                if encoding is not None:
                    encoding.azure_person_id = azure_person_id
                    encoding.azure_person_name = member_id
                    if not encoding.face_description:
                        encoding.face_description = member_id
                    encoding.source = "azure-person-group"
                    if persisted_face_ids:
                        encoding.azure_persisted_face_id = encoding.azure_persisted_face_id or persisted_face_ids[0]
                        if encoding.source != "azure-person-group":
                            encoding.source = "azure-face-list"
                    if existing_encoding is None:
                        database.create_member(encoding, member_id=member_id)
                        context["created_member"] = True
                    else:
                        database.update_member_encoding(member_id, encoding)

                additional_faces = context["faces_registered"] - (1 if created_person else 0)
                if additional_faces > 0:
                    try:
                        face_service.train_person_group(suppress_errors=False)
                    except AzureFaceError as exc:
                        context["errors"].append(f"訓練 Person Group 失敗：{exc}")

        context["azure_person_id"] = azure_person_id

    return render_template("person_group.html", context=context)


@app.post("/upload_face")
def upload_face():
    """Receive an image from the ESP32-CAM and return the member identifier."""

    try:
        image_bytes, mime_type = _extract_image_payload(request)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    try:
        encoding = recognizer.encode(image_bytes, mime_type=mime_type)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 422

    member_id: str | None = None
    distance: float | None = None
    persisted_face_id: str | None = None

    if encoding.azure_person_name:
        stored_encoding = database.get_member_encoding(encoding.azure_person_name)
        if stored_encoding is not None:
            member_id = encoding.azure_person_name
            distance = 0.0

    if (
        member_id is None
        and face_service.can_use_face_list
        and encoding.azure_face_id
    ):
        try:
            matches = face_service.find_similar_faces(
                encoding.azure_face_id,
                max_candidates=3,
            )
        except AzureFaceError as exc:
            logging.warning("Azure Face findSimilar unavailable: %s", exc)
        else:
            for match in matches:
                persisted = match.get("persisted_face_id")
                if not isinstance(persisted, str) or not persisted:
                    continue
                matched_member, stored_encoding = database.find_member_by_persisted_face_id(
                    persisted
                )
                if matched_member:
                    member_id = matched_member
                    distance = 0.0
                    persisted_face_id = persisted
                    confidence = match.get("confidence")
                    try:
                        encoding.azure_confidence = float(confidence) if confidence is not None else None
                    except (TypeError, ValueError):
                        encoding.azure_confidence = None
                    encoding.azure_persisted_face_id = persisted
                    if encoding.source != "azure-person-group":
                        encoding.source = "azure-face-list"
                    if stored_encoding is not None:
                        updated = False
                        if stored_encoding.azure_persisted_face_id != persisted:
                            stored_encoding.azure_persisted_face_id = persisted
                            updated = True
                        if (
                            encoding.azure_confidence is not None
                            and stored_encoding.azure_confidence != encoding.azure_confidence
                        ):
                            stored_encoding.azure_confidence = encoding.azure_confidence
                            updated = True
                        if (
                            encoding.face_description
                            and not stored_encoding.face_description
                        ):
                            stored_encoding.face_description = encoding.face_description
                            updated = True
                        if (
                            encoding.source != "hash"
                            and stored_encoding.source != encoding.source
                        ):
                            stored_encoding.source = encoding.source
                            updated = True
                        if updated:
                            database.update_member_encoding(matched_member, stored_encoding)
                    break

    if member_id is None:
        member_id, distance = database.find_member_by_encoding(encoding, recognizer)

    new_member = False
    if member_id is None:
        member_seed = recognizer.derive_member_id(encoding)
        if face_service.can_manage_person_group:
            try:
                azure_person_id = face_service.register_person(member_seed, image_bytes)
                encoding.azure_person_id = azure_person_id
                encoding.azure_person_name = member_seed
                encoding.source = "azure-person-group"
            except AzureFaceError as exc:
                logging.warning("Azure Face registration unavailable: %s", exc)
        if face_service.can_use_face_list:
            try:
                persisted_face_id = face_service.add_face_to_face_list(
                    member_seed,
                    image_bytes,
                    user_data=member_seed,
                )
            except AzureFaceError as exc:
                logging.warning("Azure Face face list unavailable: %s", exc)
            else:
                encoding.azure_persisted_face_id = persisted_face_id
                if encoding.source != "azure-person-group":
                    encoding.source = "azure-face-list"
        member_id = database.create_member(encoding, member_seed)
        _create_welcome_purchase(member_id)
        new_member = True
    else:
        # Update stored metadata if Azure supplied additional information.
        stored_encoding = database.get_member_encoding(member_id)
        if stored_encoding is not None:
            changed = False
            if encoding.azure_person_id and stored_encoding.azure_person_id != encoding.azure_person_id:
                stored_encoding.azure_person_id = encoding.azure_person_id
                changed = True
            if encoding.azure_person_name and stored_encoding.azure_person_name != encoding.azure_person_name:
                stored_encoding.azure_person_name = encoding.azure_person_name
                changed = True
            elif encoding.azure_person_id and not stored_encoding.azure_person_name:
                stored_encoding.azure_person_name = member_id
                changed = True
            if (
                persisted_face_id
                and stored_encoding.azure_persisted_face_id != persisted_face_id
            ):
                stored_encoding.azure_persisted_face_id = persisted_face_id
                changed = True
            if encoding.azure_confidence is not None:
                stored_encoding.azure_confidence = encoding.azure_confidence
                changed = True
            if encoding.face_description and not stored_encoding.face_description:
                stored_encoding.face_description = encoding.face_description
                changed = True
            if encoding.source != "hash" and stored_encoding.source != encoding.source:
                stored_encoding.source = encoding.source
                changed = True
            if changed:
                database.update_member_encoding(member_id, stored_encoding)
        if encoding.azure_person_id and not encoding.azure_person_name:
            encoding.azure_person_name = member_id
        if (
            persisted_face_id is None
            and face_service.can_use_face_list
            and not encoding.azure_persisted_face_id
        ):
            try:
                persisted_face_id = face_service.add_face_to_face_list(
                    member_id,
                    image_bytes,
                    user_data=member_id,
                )
            except AzureFaceError as exc:
                logging.warning("Azure Face face list unavailable: %s", exc)
            else:
                encoding.azure_persisted_face_id = persisted_face_id
                if encoding.source != "azure-person-group":
                    encoding.source = "azure-face-list"
                stored_encoding = database.get_member_encoding(member_id)
                if stored_encoding is not None:
                    stored_encoding.azure_persisted_face_id = persisted_face_id
                    if stored_encoding.source == "hash":
                        stored_encoding.source = encoding.source
                    database.update_member_encoding(member_id, stored_encoding)

    payload = {
        "status": "ok",
        "member_id": member_id,
        "new_member": new_member,
        "ad_url": url_for("render_ad", member_id=member_id, _external=True),
    }
    if distance is not None:
        payload["distance"] = distance
    return jsonify(payload), 201 if new_member else 200


@app.get("/ad/<member_id>")
def render_ad(member_id: str):
    purchases = database.get_purchase_history(member_id)
    creative = None
    if face_service.can_generate_ads:
        try:
            creative = face_service.generate_ad_copy(
                member_id,
                [
                    {
                        "item": purchase.item,
                        "last_purchase": purchase.last_purchase,
                        "discount": purchase.discount,
                        "recommendation": purchase.recommendation,
                    }
                    for purchase in purchases
                ],
            )
        except AzureFaceError as exc:
            logging.warning("Azure Face ad generation unavailable: %s", exc)
    context = build_ad_context(member_id, purchases, creative=creative)
    return render_template("ad.html", context=context)


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_image_payload(req) -> Tuple[bytes, str]:
    if req.files:
        for key in ("image", "file", "photo"):
            if key in req.files:
                uploaded = req.files[key]
                data = uploaded.read()
                if data:
                    return data, uploaded.mimetype or req.mimetype or "image/jpeg"
    data = req.get_data()
    if not data:
        raise ValueError("No image data found in request")
    return data, req.mimetype or "image/jpeg"


def _create_welcome_purchase(member_id: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d")
    database.add_purchase(
        member_id,
        "歡迎禮盒",
        now,
        0.2,
        "AI 精選：咖啡豆 x 手工甜點組，今天下單享 8 折！",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

