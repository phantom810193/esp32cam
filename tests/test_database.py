from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.database import Database, PROFILE_LABEL_TO_SEED_MEMBER, SEED_MEMBER_IDS
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
    assert result.auto_merged_source is None
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
    assert result.auto_merged_source is None
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
    assert result.auto_merged_source is None

    stored = temp_db.get_member_encoding(member_id)
    assert stored is not None
    assert stored.signature == 'rek-sig'
    assert stored.source == 'rekognition-search'


def test_resolve_member_id_merges_into_seed_member(temp_db: Database) -> None:
    recognizer = FaceRecognizer(None, tolerance=0.32)
    seed_id = SEED_MEMBER_IDS[0]
    seed_encoding = _face_encoding(0.0, seed_id, source='seed')
    temp_db.create_member(seed_encoding, member_id=seed_id)

    candidate_encoding = _face_encoding(0.8, 'brand-new-seed')

    result = temp_db.resolve_member_id(candidate_encoding, recognizer)

    assert result.member_id == seed_id
    assert result.new_member is False
    assert result.auto_merged_source is not None
    assert result.auto_merged_source != seed_id

    with temp_db._connect() as conn:
        rows = conn.execute('SELECT member_id FROM members').fetchall()
    member_ids = {row[0] for row in rows}
    assert member_ids == {seed_id}


def test_merge_members_preserves_unique_profile_constraint(temp_db: Database) -> None:
    target_id = temp_db.create_member(_face_encoding(0.0, 'target'), member_id='MEMTARGET')
    source_id = temp_db.create_member(_face_encoding(0.1, 'source'), member_id='MEMSOURCE')

    with temp_db._connect() as conn:
        conn.execute(
            "INSERT INTO member_profiles (profile_label, name, member_id, first_image_filename) VALUES (?, ?, ?, ?)",
            ('target-profile', 'Target', target_id, None),
        )
        conn.execute(
            "INSERT INTO member_profiles (profile_label, name, member_id, first_image_filename) VALUES (?, ?, ?, ?)",
            ('source-profile', 'Source', source_id, 'face.png'),
        )
        conn.commit()

    temp_db.merge_members(source_id, target_id)

    with temp_db._connect() as conn:
        rows = conn.execute(
            "SELECT member_id, first_image_filename FROM member_profiles ORDER BY profile_id"
        ).fetchall()

    assert any(row["member_id"] == target_id for row in rows)
    assert any(row["member_id"] is None for row in rows)
    for row in rows:
        if row["member_id"] == target_id:
            assert row["first_image_filename"] == 'face.png'


def test_ensure_demo_data_populates_seed_profiles(tmp_path: Path) -> None:
    db = Database(tmp_path / 'seed.sqlite3')
    db.ensure_demo_data()

    dessert = db.get_member_profile('MEME0383FE3AA')
    family = db.get_member_profile('MEM692FFD0824')

    assert dessert is not None
    assert dessert.name == '李函霏'
    assert dessert.mall_member_id == 'ME0001'

    assert family is not None
    assert family.name == '林位青'
    assert family.mall_member_id == 'ME0002'


def test_ensure_demo_data_repairs_seed_profiles(tmp_path: Path) -> None:
    db = Database(tmp_path / 'seed-repair.sqlite3')
    db.ensure_demo_data()

    with db._connect() as conn:
        conn.execute(
            "UPDATE member_profiles SET name = ?, mall_member_id = ? WHERE member_id = ?",
            ('林悅心', 'WRONG', 'MEME0383FE3AA'),
        )
        conn.execute(
            "UPDATE member_profiles SET name = ?, member_id = NULL WHERE profile_label = ?",
            ('錯誤資料', 'family-groceries'),
        )
        conn.commit()

    db.ensure_demo_data()

    dessert = db.get_member_profile('MEME0383FE3AA')
    family = db.get_member_profile('MEM692FFD0824')

    assert dessert is not None
    assert dessert.name == '李函霏'
    assert dessert.mall_member_id == 'ME0001'

    assert family is not None
    assert family.name == '林位青'
    assert family.member_id == 'MEM692FFD0824'


def test_profile_label_mapping_supports_aliases() -> None:
    assert PROFILE_LABEL_TO_SEED_MEMBER['dessert-lover'] == 'MEME0383FE3AA'
    assert PROFILE_LABEL_TO_SEED_MEMBER['dessert_lover'] == 'MEME0383FE3AA'
