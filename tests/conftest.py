import builtins
import datetime as dt
import pytest

_BASE = dt.date.today()

_DEF_HISTORIES = {
    "dessert_history": [
        {
            "member_id": "ME0001",
            "category": "dessert",
            "item": "布朗尼",
            "price": 60,
            "purchased_at": str(_BASE - dt.timedelta(days=2)),
        },
        {
            "member_id": "ME0001",
            "category": "dessert",
            "item": "可頌",
            "price": 55,
            "purchased_at": str(_BASE - dt.timedelta(days=8)),
        },
        {
            "member_id": "ME0001",
            "category": "dessert",
            "item": "司康",
            "price": 50,
            "purchased_at": str(_BASE - dt.timedelta(days=15)),
        },
    ],
    "kids_history": [
        {
            "member_id": "ME0002",
            "category": "groceries",
            "item": "有機蘋果",
            "price": 120,
            "purchased_at": str(_BASE - dt.timedelta(days=5)),
        }
    ],
    "fitness_history": [
        {
            "member_id": "ME0003",
            "category": "fitness",
            "item": "乳清蛋白",
            "price": 980,
            "purchased_at": str(_BASE - dt.timedelta(days=3)),
        }
    ],
    "homemaker_history": [
        {
            "member_id": "ME0004",
            "category": "home",
            "item": "香氛蠟燭",
            "price": 450,
            "purchased_at": str(_BASE - dt.timedelta(days=6)),
        }
    ],
    "health_history": [
        {
            "member_id": "ME0005",
            "category": "wellness",
            "item": "草本茶",
            "price": 320,
            "purchased_at": str(_BASE - dt.timedelta(days=10)),
        }
    ],
}

for _name, _value in _DEF_HISTORIES.items():
    if not hasattr(builtins, _name):  # pragma: no cover
        setattr(builtins, _name, _value)

if not hasattr(builtins, "insert_statements"):  # pragma: no cover
    builtins.insert_statements = []  # type: ignore[attr-defined]

# 僅在專案內尚未定義同名 fixture 時新增
try:
    dessert_history  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    @pytest.fixture
    def dessert_history():
        """提供近30天內的甜點購買紀錄，讓預測流程可跑通。"""
        return list(_DEF_HISTORIES["dessert_history"])
