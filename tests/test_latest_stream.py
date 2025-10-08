import json
import sys
import types
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports that rely on the project root being on sys.path
from backend.advertising import CTA_JOIN_MEMBER, TEMPLATE_IMAGE_BY_ID, build_ad_context
from backend.database import NEW_GUEST_MEMBER_ID, MemberProfile, Purchase

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


def test_guest_cta_uses_prediction_template_image():
    profile = MemberProfile(
        profile_id=1,
        profile_label="wellness-gourmet",
        name="未註冊客戶",
        member_id="MEMHEALTH2025",
        mall_member_id=None,
        member_status=None,
        joined_at=None,
        points_balance=None,
        gender=None,
        birth_date=None,
        phone=None,
        email=None,
        address=None,
        occupation=None,
        first_image_filename=None,
    )
    purchase = Purchase(
        member_id="MEMHEALTH2025",
        member_code="",
        product_category="健康食品",
        internal_item_code="SKU-0001",
        item="高纖燕麥片",
        purchased_at="2025-09-01 10:00",
        unit_price=360.0,
        quantity=1.0,
        total_price=360.0,
    )

    context = build_ad_context(
        "MEMHEALTH2025",
        [purchase],
        profile=profile,
        prediction_items=[],
        audience="guest",
    )

    expected_filename = TEMPLATE_IMAGE_BY_ID.get("ME0003", "ME0003.jpg")
    assert context.cta_href == f"/static/images/ads/{expected_filename}"
    assert context.cta_href.startswith("/static/images/ads/")
    assert not context.cta_href.startswith("#")
    assert context.cta_text in {CTA_JOIN_MEMBER, "立即加入會員"}


def test_new_guest_cta_uses_registration_image():
    profile = MemberProfile(
        profile_id=2,
        profile_label="brand-new-guest",
        name="新客",
        member_id=NEW_GUEST_MEMBER_ID,
        mall_member_id="",
        member_status="未入會",
        joined_at=None,
        points_balance=0.0,
        gender=None,
        birth_date=None,
        phone=None,
        email=None,
        address=None,
        occupation=None,
        first_image_filename=None,
    )

    context = build_ad_context(
        NEW_GUEST_MEMBER_ID,
        [],
        profile=profile,
        prediction_items=[],
        audience="guest",
    )

    expected_filename = TEMPLATE_IMAGE_BY_ID.get("ME0000", "ME0000.jpg")
    assert context.cta_href == f"/static/images/ads/{expected_filename}"
    assert context.cta_text in {CTA_JOIN_MEMBER, "立即加入會員"}


def test_dashboard_new_guest_has_blank_sections(client):
    response = client.get(f"/dashboard?member_id={NEW_GUEST_MEMBER_ID}")

    assert response.status_code == 200

    html = response.get_data(as_text=True)
    assert "<tr data-prediction-row" not in html
    assert "目前無可用的推薦商品" not in html
    assert "<tr data-month=" not in html
    assert "尚未有消費紀錄" not in html


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
