"""Regression tests for the /upload_face endpoint."""

from __future__ import annotations

import io
import json

from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import app as backend_app
from backend.database import Database
from backend.recognizer import FaceEncoding, FaceRecognizer


class FlakyGemini:
    """Gemini stub that returns a different description on every call."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def can_describe_faces(self) -> bool:  # pragma: no cover - simple property
        return True

    def describe_face(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        self.calls += 1
        return f"desc-{self.calls}"


def _jpeg_bytes() -> bytes:
    """Generate a deterministic in-memory JPEG fixture for testing."""

    image = Image.new("RGB", (8, 8), color=(123, 45, 67))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_repeated_upload_reuses_member_id(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test.sqlite3"
    database = Database(test_db_path)
    gemini = FlakyGemini()
    recognizer = FaceRecognizer(gemini)

    monkeypatch.setattr(backend_app, "database", database)
    monkeypatch.setattr(backend_app, "recognizer", recognizer)

    client = backend_app.app.test_client()
    payload = _jpeg_bytes()

    first = client.post("/upload_face", data=payload, content_type="image/jpeg")
    assert first.status_code == 201
    first_data = first.get_json()
    assert first_data["new_member"] is True
    member_id = first_data["member_id"]

    second = client.post("/upload_face", data=payload, content_type="image/jpeg")
    assert second.status_code == 200
    second_data = second.get_json()
    assert second_data["new_member"] is False
    assert second_data["member_id"] == member_id

    assert gemini.calls == 2, "Gemini should be consulted on each upload"

    with database._connect() as conn:
        row = conn.execute(
            "SELECT encoding_json FROM members WHERE member_id = ?",
            (member_id,),
        ).fetchone()
    assert row is not None

    encoding = FaceEncoding.from_jsonable(json.loads(row["encoding_json"]))
    assert encoding.signature == FaceRecognizer._hash_signature(payload)
    assert encoding.gemini_description == "desc-1"
    assert encoding.source == "hash+gemini"
