from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from ..ai import AdCreative, AzureFaceError, FaceAnalysis
from ..app import app as flask_app
from ..database import Database
from ..recognizer import FaceEncoding, FaceRecognizer


class FlakyAzureFace:
    """Test double that mimics the Azure Face interface."""

    def __init__(self, description: str = "短髮微笑顧客", fail_first: bool = False) -> None:
        self.description = description
        self.fail_first = fail_first
        self.calls = 0
        self.register_calls = 0
        self.add_face_calls = 0
        self.trained_calls = 0
        self.person_id: str | None = None
        self.registered_member: str | None = None
        self.faces_added: list[bytes] = []

    @property
    def can_describe_faces(self) -> bool:  # pragma: no cover - simple proxy
        return True

    @property
    def can_generate_ads(self) -> bool:  # pragma: no cover - simple proxy
        return True

    @property
    def can_manage_person_group(self) -> bool:  # pragma: no cover - simple proxy
        return True

    @property
    def person_group_error(self) -> str | None:  # pragma: no cover - simple proxy
        return None

    def analyze_face(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> FaceAnalysis:
        del mime_type
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise AzureFaceError("transient failure")
        return FaceAnalysis(
            description=self.description,
            face_id=f"face-{self.calls}",
            person_id=self.person_id if self.registered_member else None,
            person_name=self.registered_member,
            confidence=0.9 if self.registered_member else None,
        )

    def describe_face(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        return self.analyze_face(image_bytes, mime_type).description

    def register_person(self, member_id: str, image_bytes: bytes, *, user_data: str | None = None) -> str:
        del user_data
        self.register_calls += 1
        self.person_id = f"person-{self.register_calls}"
        self.registered_member = member_id
        self.faces_added.append(image_bytes)
        self.train_person_group()
        return self.person_id

    def add_face_to_person(self, person_id: str, image_bytes: bytes) -> None:
        if not self.person_id or person_id != self.person_id:
            raise AzureFaceError("unknown person")
        self.add_face_calls += 1
        self.faces_added.append(image_bytes)

    def train_person_group(self, suppress_errors: bool = True) -> None:
        del suppress_errors
        self.trained_calls += 1

    def find_person_id_by_name(self, member_name: str) -> str | None:
        if self.registered_member == member_name:
            return self.person_id
        return None

    def generate_ad_copy(self, member_id: str, purchases):  # pragma: no cover - unused by test
        return AdCreative(
            headline=f"會員 {member_id}，歡迎回來！",
            subheading="專屬優惠等你",
            highlight="今日下單享免運",
        )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    database = Database(tmp_path / "test.sqlite3")
    service = FlakyAzureFace()
    recognizer = FaceRecognizer(service)

    monkeypatch.setattr("backend.app.database", database)
    monkeypatch.setattr("backend.app.face_service", service)
    monkeypatch.setattr("backend.app.recognizer", recognizer)

    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as test_client:
        yield test_client, service, database


def _post_image(test_client, payload: bytes):
    return test_client.post(
        "/upload_face",
        data=payload,
        content_type="image/jpeg",
    )


def test_reuse_member_id_and_serialisation(client):
    test_client, service, database = client
    image_payload = b"fake-image-data"

    response = _post_image(test_client, image_payload)
    assert response.status_code == 201
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["new_member"] is True
    member_id = data["member_id"]

    with database._connect() as conn:  # pylint: disable=protected-access
        row = conn.execute(
            "SELECT encoding_json FROM members WHERE member_id = ?",
            (member_id,),
        ).fetchone()
    assert row is not None
    stored_encoding = json.loads(row["encoding_json"])
    assert stored_encoding["face_description"] == service.description
    assert stored_encoding["source"] == "azure-person-group"
    assert stored_encoding["azure_person_id"] == "person-1"
    assert stored_encoding["azure_person_name"] == member_id
    assert "gemini_description" not in stored_encoding

    response_again = _post_image(test_client, image_payload)
    assert response_again.status_code == 200
    data_again = response_again.get_json()
    assert data_again["status"] == "ok"
    assert data_again["member_id"] == member_id
    assert data_again["new_member"] is False
    assert service.calls == 2
    assert service.register_calls == 1


def test_person_group_training_creates_member(client):
    test_client, service, database = client

    response = test_client.post(
        "/person-group",
        data={
            "member_id": "VIP001",
            "images": [
                (io.BytesIO(b"face-one"), "one.jpg", "image/jpeg"),
                (io.BytesIO(b"face-two"), "two.jpg", "image/jpeg"),
            ],
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert service.register_calls == 1
    assert service.add_face_calls == 1
    assert service.trained_calls >= 1

    encoding = database.get_member_encoding("VIP001")
    assert encoding is not None
    assert encoding.azure_person_id == "person-1"
    assert encoding.azure_person_name == "VIP001"
    assert encoding.source == "azure-person-group"


def test_person_group_training_updates_existing_person(client):
    test_client, service, database = client

    existing = FaceEncoding(
        vector=np.zeros(128, dtype=np.float32),
        face_description="existing",
        source="seed",
        azure_person_id="person-5",
        azure_person_name="VIP777",
    )
    database.create_member(existing, member_id="VIP777")

    service.person_id = "person-5"
    service.registered_member = "VIP777"

    response = test_client.post(
        "/person-group",
        data={
            "member_id": "VIP777",
            "images": [
                (io.BytesIO(b"face-a"), "a.jpg", "image/jpeg"),
                (io.BytesIO(b"face-b"), "b.jpg", "image/jpeg"),
            ],
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert service.register_calls == 0
    assert service.add_face_calls == 2
    assert service.trained_calls == 1

    updated = database.get_member_encoding("VIP777")
    assert updated is not None
    assert updated.azure_person_id == "person-5"
    assert updated.azure_person_name == "VIP777"
    assert updated.source == "azure-person-group"
