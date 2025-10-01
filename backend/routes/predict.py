# -*- coding: utf-8 -*-
"""
Predictions & history endpoints
- GET /members/<member_id>/predictions?ym=YYYY-MM
- GET /members/<member_id>/history?ym=YYYY-MM
說明：
- 以 SQLite 為主（預設路徑 backend/data/mvp.sqlite3，亦可由環境變數 DB_PATH 指定）
- 推估規則：取「當年度上一個月」的行為，推本月「尚未購買」的 7 筆候選
- 欄位：商品編號/名稱/類別/價格/查閱率/預估購買機率(%)
- 若缺少 views 表，view_rate 以購買次數做近似估計並正規化
"""
from __future__ import annotations
import os, sqlite3, math, datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any
from flask import Blueprint, jsonify, request

predict_bp = Blueprint("predict", __name__)

def db_path() -> str:
    return os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "mvp.sqlite3"))

def connect():
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    return conn

def ym_prev(ym: str | None) -> str:
    # ym = "YYYY-MM"; 若空，取今天的上個月
    if not ym:
        today = dt.date.today().replace(day=1)
        prev = (today - dt.timedelta(days=1)).replace(day=1)
    else:
        y, m = map(int, ym.split("-"))
        base = dt.date(y, m, 1)
        prev = (base - dt.timedelta(days=1)).replace(day=1)
    return f"{prev.year:04d}-{prev.month:02d}"

@dataclass
class ItemRow:
    sku: str
    name: str
    category: str
    price: float
    view_rate: float
    prob_pct: float

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def _get_last_month_range(ym: str) -> tuple[str, str]:
    y, m = map(int, ym.split("-"))
    start = dt.date(y, m, 1)
    if m == 12:
        end = dt.date(y+1, 1, 1)
    else:
        end = dt.date(y, m+1, 1)
    return (start.isoformat(), end.isoformat())

def has_table(conn, name: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cur.fetchone() is not None

def fetch_catalog(conn) -> Dict[str, Dict[str, Any]]:
    # 需要 items 表：sku(name, category, price)
    if not has_table(conn, "items"):
        return {}
    out = {}
    for r in conn.execute("SELECT sku, name, category, price FROM items;"):
        out[r["sku"]] = dict(sku=r["sku"], name=r["name"], category=r["category"], price=_safe_float(r["price"]))
    return out

def last_month_purchases(conn, member_id: str, ym_last: str) -> Dict[str, int]:
    start, end = _get_last_month_range(ym_last)
    if not has_table(conn, "purchases"):
        return {}
    cur = conn.execute(
        """
        SELECT sku, COUNT(*) AS cnt
        FROM purchases
        WHERE member_id = ? AND purchased_at >= ? AND purchased_at < ?
        GROUP BY sku
        """,
        (member_id, start, end)
    )
    return {r["sku"]: r["cnt"] for r in cur.fetchall()}

def last_month_views(conn, ym_last: str) -> Dict[str, int]:
    # 可選 views 表：views(sku, viewed_at)
    if not has_table(conn, "views"):
        return {}
    start, end = _get_last_month_range(ym_last)
    cur = conn.execute(
        """
        SELECT sku, COUNT(*) AS v
        FROM views
        WHERE viewed_at >= ? AND viewed_at < ?
        GROUP BY sku
        """,
        (start, end)
    )
    return {r["sku"]: r["v"] for r in cur.fetchall()}

def member_level(conn, member_id: str) -> str:
    if not has_table(conn, "members"):
        return "guest"
    cur = conn.execute("SELECT COALESCE(level, 'general') AS lv FROM members WHERE member_id=?;", (member_id,))
    r = cur.fetchone()
    return (r["lv"] if r else "guest") or "guest"

@predict_bp.get("/members/<member_id>/history")
def history(member_id: str):
    ym = request.args.get("ym")  # 指定月份
    ym_last = ym_prev(ym)
    start, end = _get_last_month_range(ym_last)
    with connect() as conn:
        if not has_table(conn, "purchases"):
            return jsonify({"ym": ym_last, "rows": []})
        cur = conn.execute(
            """
            SELECT purchased_at, sku, item AS name, category, unit_price AS price, quantity, total_price
            FROM purchases
            WHERE member_id=? AND purchased_at >= ? AND purchased_at < ?
            ORDER BY purchased_at DESC
            """,
            (member_id, start, end)
        )
        rows = [dict(r) for r in cur.fetchall()]
    return jsonify({"ym": ym_last, "rows": rows})

@predict_bp.get("/members/<member_id>/predictions")
def predictions(member_id: str):
    """
    回傳 7 筆候選 + 機率(%)
    公式（可簡化理解）：
      score = normalize( view_rate * conv_rate * time_decay * member_weight )
    其中：
      - view_rate 來自 views 表；若無，改以上月整體購買次數估計
      - conv_rate 以「上月購買次數 / max(1, 上月瀏覽)」估計
      - time_decay 若該會員對該 SKU 曾在過去 90 天買過則衰減
      - member_weight 依會員等級做 0.9~1.2 微調
    """
    ym = request.args.get("ym")
    ym_last = ym_prev(ym)

    with connect() as conn:
        catalog = fetch_catalog(conn)
        if not catalog:
            return jsonify({"ym": ym_last, "rows": []})

        # 基礎統計
        v_all = last_month_views(conn, ym_last)
        p_all_member_last = last_month_purchases(conn, member_id, ym_last)

        # 本月已購商品 (避免推薦)
        this_ym = ym or dt.date.today().strftime("%Y-%m")
        p_this_month = last_month_purchases(conn, member_id, this_ym)  # 便利用同函式抓今月，命名沿用
        already = set(p_this_month.keys())

        # 估計整體 view 分母
        total_views = sum(v_all.values())
        if total_views == 0:
            # 無 views 表或皆為 0：以全站上月購買次數近似 view
            all_purchases = {}
            if has_table(conn, "purchases"):
                start, end = _get_last_month_range(ym_last)
                cur = conn.execute(
                    """SELECT sku, COUNT(*) AS cnt FROM purchases
                       WHERE purchased_at >= ? AND purchased_at < ?
                       GROUP BY sku;""", (start, end))
                all_purchases = {r["sku"]: r["cnt"] for r in cur.fetchall()}
            v_all = all_purchases
            total_views = max(1, sum(v_all.values()))

        # 會員權重
        lv = member_level(conn, member_id)
        mw = {"vip": 1.2, "gold": 1.1, "general": 1.0, "guest": 0.95}.get(lv, 1.0)

        scored: List[ItemRow] = []
        # 時近衰減需要該會員近 90 天的最後一次購買日
        recent_last: Dict[str, str] = {}
        if has_table(conn, "purchases"):
            cur = conn.execute(
                """SELECT sku, MAX(purchased_at) AS last_at
                   FROM purchases
                   WHERE member_id=?
                     AND purchased_at >= date('now','-180 day')
                   GROUP BY sku;""",
                (member_id,)
            )
            recent_last = {r["sku"]: r["last_at"] for r in cur.fetchall() if r["last_at"]}

        # 計分
        for sku, meta in catalog.items():
            if sku in already:
                continue  # 本月已買就不推
            views = v_all.get(sku, 0)
            conv = p_all_member_last.get(sku, 0)
            view_rate = views / total_views if total_views else 0.0
            conv_rate = conv / max(1, views)
            # time decay: 最近買過者衰減（越近衰減越大，避免連續推相同品）
            decay = 1.0
            if sku in recent_last:
                days = max(0, (dt.date.today() - dt.date.fromisoformat(recent_last[sku])).days)
                lam = 1.0 / 45.0
                decay = math.exp(-lam * (90 - min(90, days)))
            score = max(0.0, view_rate * conv_rate * decay * mw)
            scored.append(ItemRow(
                sku=sku,
                name=str(meta.get("name","")),
                category=str(meta.get("category","")),
                price=float(meta.get("price") or 0.0),
                view_rate=round(view_rate*100, 2),
                prob_pct=0.0  # 暫存，稍後正規化
            ))
        # 正規化為百分比
        ssum = sum([i.view_rate for i in scored]) or 1.0
        for i in scored:
            # 這裡用 view_rate 近似分母，若要更嚴謹可保存 score 再正規化
            i.prob_pct = round((i.view_rate / ssum) * 100.0, 1)

        # 取前 7
        top7 = sorted(scored, key=lambda x: x.prob_pct, reverse=True)[:7]
        rows = [dict(
            sku=i.sku, name=i.name, category=i.category, price=i.price,
            view_rate=i.view_rate, probability_percent=i.prob_pct
        ) for i in top7]
        return jsonify({"ym": ym_last, "rows": rows, "member_level": lv})
