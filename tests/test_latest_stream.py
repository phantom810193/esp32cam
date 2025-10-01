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
# Stub cloud SDKs that are unavailable in the execution environment.
# ---------------------------------------------------------------------------
google_stub = types.ModuleType("google")
cloud_stub = types.ModuleType("google.cloud")
storage_stub = types.ModuleType("google.cloud.storage")
aiplatform_stub = types.ModuleType("google.cloud.aiplatform")
api_core_stub = types.ModuleType("google.api_core")
api_core_exceptions_stub = types.ModuleType("google.api_core.exceptions")
vertexai_stub = types.ModuleType("vertexai")
vertexai_generative_stub = types.ModuleType("vertexai.generative_models")
vertexai_preview_stub = types.ModuleType("vertexai.preview")
vertexai_preview_vision_stub = types.ModuleType("vertexai.preview.vision_models")
vertexai_vision_stub = types.ModuleType("vertexai.vision_models")


class _StorageClient:  # pragma: no cover - just a safety stub
    def __init__(self, *args, **kwargs):
        raise RuntimeError("google.cloud.storage is not available in tests")


def _noop(*_args, **_kwargs):  # pragma: no cover
    return None


class _GenerativeModel:  # pragma: no cover - simple stub
    def __init__(self, name: str):
        self._name = name

    def generate_content(self, parts, request_options=None):
        return types.SimpleNamespace(text="")


class _Part:  # pragma: no cover
    @classmethod
    def from_data(cls, **kwargs):
        return kwargs


class _GenerationConfig:  # pragma: no cover
    def __init__(self, **kwargs):
        self.values = kwargs


class _ImageGenerationModel:  # pragma: no cover
    @classmethod
    def from_pretrained(cls, name: str):
        return cls()


storage_stub.Client = _StorageClient
cloud_stub.storage = storage_stub
cloud_stub.aiplatform = aiplatform_stub
api_core_exceptions_stub.NotFound = type("NotFound", (Exception,), {})
api_core_stub.exceptions = api_core_exceptions_stub
vertexai_generative_stub.GenerativeModel = _GenerativeModel
vertexai_generative_stub.Part = _Part
vertexai_preview_vision_stub.ImageGenerationModel = _ImageGenerationModel
vertexai_vision_stub.ImageGenerationModel = _ImageGenerationModel
vertexai_preview_stub.vision_models = vertexai_preview_vision_stub
vertexai_generative_stub.GenerationConfig = _GenerationConfig
vertexai_stub.generative_models = vertexai_generative_stub
vertexai_stub.preview = vertexai_preview_stub
vertexai_stub.vision_models = vertexai_vision_stub

aiplatform_stub.init = _noop

google_stub.cloud = cloud_stub

sys.modules.setdefault("google", google_stub)
sys.modules.setdefault("google.cloud", cloud_stub)
sys.modules.setdefault("google.cloud.storage", storage_stub)
sys.modules.setdefault("google.cloud.aiplatform", aiplatform_stub)
sys.modules.setdefault("google.api_core", api_core_stub)
sys.modules.setdefault("google.api_core.exceptions", api_core_exceptions_stub)
sys.modules.setdefault("vertexai", vertexai_stub)
sys.modules.setdefault("vertexai.generative_models", vertexai_generative_stub)
sys.modules.setdefault("vertexai.preview", vertexai_preview_stub)
sys.modules.setdefault("vertexai.preview.vision_models", vertexai_preview_vision_stub)
sys.modules.setdefault("vertexai.vision_models", vertexai_vision_stub)

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
    assert payload["event_id"] == database.get_latest_upload_event().id


def test_latest_stream_rejects_invalid_interval(client):
    response = client.get("/ad/latest/stream?interval=abc&once=1")

    assert response.status_code == 400
    payload = response.get_json()
    assert payload == {"error": "Invalid interval"}
