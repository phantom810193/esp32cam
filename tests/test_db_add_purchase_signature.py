import pytest
from unittest.mock import patch

from backend.database import Database


def _make_db(tmp_path):
    return Database(tmp_path / "test.db")


def test_add_purchase_positional_member_id(tmp_path):
    db = _make_db(tmp_path)
    with patch.object(db, "_insert_purchase", return_value=None) as spy:
        db.add_purchase(
            "ME0001",
            item="可頌",
            purchased_at="2024-01-01",
            unit_price=55,
            quantity=1,
            total_price=55,
        )
    spy.assert_called_once()


def test_add_purchase_keyword_member_id(tmp_path):
    db = _make_db(tmp_path)
    with patch.object(db, "_insert_purchase", return_value=None) as spy:
        db.add_purchase(
            member_id="ME0001",
            item="可頌",
            purchased_at="2024-01-01",
            unit_price=55,
            quantity=1,
            total_price=55,
        )
    spy.assert_called_once()


def test_add_purchase_duplicate_member_id_raises(tmp_path):
    db = _make_db(tmp_path)
    with pytest.raises(TypeError):
        db.add_purchase(
            "ME0001",
            member_id="ME0001",
            item="可頌",
            purchased_at="2024-01-01",
            unit_price=55,
            quantity=1,
            total_price=55,
        )
