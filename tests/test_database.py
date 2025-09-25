import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.database import Database
from backend.recognizer import FaceEncoding


class StubRecognizer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def find_best_match(self, candidates, encoding):
        self.calls.append((candidates, encoding))
        return self.result


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "test.sqlite3"


def _load_member_encoding(path, member_id):
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT encoding_json FROM members WHERE member_id = ?",
            (member_id,),
        ).fetchone()
    assert row is not None, "member not stored"
    return FaceEncoding.from_jsonable(json.loads(row[0]))


def test_matching_same_signature_updates_encoding(db_path):
    database = Database(db_path)
    base_vec = np.ones(128, dtype=np.float32)
    base_encoding = FaceEncoding(vector=base_vec, signature="sig", source="insightface")
    database.create_member(base_encoding, member_id="MEM1")

    incoming_vec = np.full(128, 0.5, dtype=np.float32)
    incoming = FaceEncoding(vector=incoming_vec, signature="sig", source="insightface")

    member_id, distance = database.find_member_by_encoding(incoming, StubRecognizer((None, None)))

    assert member_id == "MEM1"
    assert distance == 0.0

    stored = _load_member_encoding(db_path, "MEM1")
    assert pytest.approx(stored.vector[0], rel=1e-3) == 0.7
    assert stored.source == "insightface"


def test_best_match_blends_vectors(db_path):
    database = Database(db_path)
    base_vec = np.ones(128, dtype=np.float32)
    base_encoding = FaceEncoding(vector=base_vec, signature="base", source="insightface")
    database.create_member(base_encoding, member_id="MEM1")

    incoming_vec = np.full(128, 0.2, dtype=np.float32)
    incoming = FaceEncoding(vector=incoming_vec, signature="other", source="insightface")

    recognizer = StubRecognizer(("MEM1", 0.25))
    member_id, distance = database.find_member_by_encoding(incoming, recognizer)

    assert recognizer.calls, "recognizer should be invoked"
    assert member_id == "MEM1"
    assert distance == 0.25

    stored = _load_member_encoding(db_path, "MEM1")
    assert pytest.approx(stored.vector[0], rel=1e-3) == 0.52
    assert stored.source == "insightface"
