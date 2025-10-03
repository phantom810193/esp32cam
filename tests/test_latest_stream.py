import json
import sys
import types
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Stub the Google Gemini SDK to avoid network calls during tests.
# ---------------------------------------------------------------------------
google_stub = types.ModuleType("google")
generativeai_stub = types.ModuleType("google.generativeai")


class _GenerativeModel:  # pragma: no cover - simple stub
    def __init__(self, name: str):
        self._name = name

    def generate_content(self, prompt, generation_config=None):  # noqa: ARG002
        return types.SimpleNamespace(text="")


def _configure(api_key=None):  # pragma: no cover - mimic SDK signature
    return None


generativeai_stub.GenerativeModel = _GenerativeModel
generativeai_stub.configure = _configure

google_stub.generativeai = generativeai_stub

sys.modules.setdefault("google", google_stub)
sys.modules.setdefault("google.generativeai", generativeai_stub)

from backend.app import app, database


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    database.cleanup_upload_events(keep_latest=0)
    with app.test_client() as test_client:
        yield test_client
    database.cleanup_upload_events(keep_latest=0)


def _parse_sse_payload(body: str) -> dict:
    for line in body.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise AssertionError("SSE payload did not contain data line")


def test_latest_stream_emits_latest_event(client):
    member_id = f"MEM{uuid4().hex[:8]}"
    database.record_upload_event(
        member_id=member_id,
        image_filename=None,
        upload_duration=0.11,
        recognition_duration=0.22,
        ad_duration=0.33,
        total_duration=0.66,
    )

    response = client.get("/ad/latest/stream?once=1")
    body = b"".join(response.response).decode("utf-8")

    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/event-stream")

    payload = _parse_sse_payload(body)
    assert payload["status"] == "ok"
    assert payload["member_id"] == member_id
    assert payload["ad_url"].endswith(f"/ad/{member_id}")
    assert payload["offer_url"].endswith(f"/ad/{member_id}/offer")
    assert payload["event_id"] == database.get_latest_upload_event().id


def test_latest_stream_rejects_invalid_interval(client):
    response = client.get("/ad/latest/stream?interval=abc&once=1")

    assert response.status_code == 400
    payload = response.get_json()
    assert payload == {"error": "Invalid interval"}
