from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.database import Database
from backend.recognizer import FaceEncoding, FaceRecognizer


@pytest.fixture()
def temp_db(tmp_path: Path) -> Database:
    db_path = tmp_path / 'mvp.sqlite3'
    return Database(db_path)


def _face_encoding(vector_value: float, signature: str, source: str = 'hash') -> FaceEncoding:
    vector = np.zeros(128, dtype=np.float32)
    vector[0] = vector_value
    return FaceEncoding(vector=vector, signature=signature, source=source)


def test_resolve_member_id_auto_reuses_existing_member(temp_db: Database) -> None:
    recognizer = FaceRecognizer(None, tolerance=0.32)
    existing_encoding = _face_encoding(0.0, 'seed-signature')
    member_id = temp_db.create_member(existing_encoding, member_id='MEMEXIST001')

    candidate_encoding = _face_encoding(0.35, 'new-candidate')

    result = temp_db.resolve_member_id(candidate_encoding, recognizer)

    assert result.member_id == member_id
    assert result.new_member is False
    assert result.encoding_updated is False
    assert result.distance == pytest.approx(0.35, rel=1e-6)

    with temp_db._connect() as conn:
        count = conn.execute('SELECT COUNT(*) FROM members').fetchone()[0]
    assert count == 1


def test_resolve_member_id_creates_new_member_when_far(temp_db: Database) -> None:
    recognizer = FaceRecognizer(None, tolerance=0.32)
    existing_encoding = _face_encoding(0.0, 'seed-signature')
    temp_db.create_member(existing_encoding, member_id='MEMEXIST002')

    candidate_encoding = _face_encoding(0.6, 'brand-new')

    result = temp_db.resolve_member_id(candidate_encoding, recognizer)

    assert result.new_member is True
    assert result.member_id != 'MEMEXIST002'
    assert result.distance == pytest.approx(0.6, rel=1e-6)

    with temp_db._connect() as conn:
        rows = conn.execute('SELECT member_id FROM members ORDER BY member_id').fetchall()
    assert len(rows) == 2
    assert any(row[0] == result.member_id for row in rows)


def test_resolve_member_id_refreshes_encoding_from_rekognition(temp_db: Database) -> None:
    recognizer = FaceRecognizer(None, tolerance=0.32)
    # Stored encoding without a signature should be replaced by one coming from Rekognition.
    existing_encoding = _face_encoding(0.0, '', source='hash')
    member_id = temp_db.create_member(existing_encoding, member_id='MEMEMPTY')

    improved_encoding = _face_encoding(0.0, 'rek-sig', source='rekognition-search')

    result = temp_db.resolve_member_id(improved_encoding, recognizer)

    assert result.member_id == member_id
    assert result.new_member is False
    assert result.encoding_updated is True

    stored = temp_db.get_member_encoding(member_id)
    assert stored is not None
    assert stored.signature == 'rek-sig'
    assert stored.source == 'rekognition-search'

