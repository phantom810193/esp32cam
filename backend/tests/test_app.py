import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import app as app_module


def test_upload_face_records_last_ip(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test.sqlite3"
    test_database = app_module.Database(test_db_path)
    test_database.ensure_demo_data()
    monkeypatch.setattr(app_module, "database", test_database)

    app_module.last_uploader_ip = None

    client = app_module.app.test_client()
    headers = {"X-Forwarded-For": "203.0.113.42"}
    response = client.post(
        "/upload_face",
        data=b"test-bytes",
        content_type="image/jpeg",
        headers=headers,
    )

    assert response.status_code in (200, 201)
    payload = json.loads(response.get_data(as_text=True))
    assert payload["status"] == "ok"

    assert app_module.last_uploader_ip == "203.0.113.42"

    landing_page = client.get("/")
    assert landing_page.status_code == 200
    body = landing_page.get_data(as_text=True)
    assert "203.0.113.42" in body
    assert "http://203.0.113.42:81/stream" in body
