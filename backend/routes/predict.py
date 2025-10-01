# -*- coding: utf-8 -*-
"""
Predictions & history endpoints
- GET /members/<member_id>/predictions?ym=YYYY-MM&limit=7
- GET /members/<member_id>/history?ym=YYYY-MM
說明：
- 以 SQLite 為主（預設路徑 backend/data/mvp.sqlite3，亦可由環境變數 DB_PATH 指定）
- 推估規則：取「當年度上一個月」的行為，推本月「尚未購買」的候選（預設 7 筆）
- 欄位：商品編號/名稱/類別/價格/查閱率(%)/預估購買機率(%)
- 若缺少 views 表，view_rate 以購買次數做近似估計並正規化
- 任何錯誤皆回 200 並帶 ok: false 與可讀錯誤訊息，避免 500
"""
from __future__ import annotations

import os
import math
import sqlite3
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from flask import Blueprint, jsonify, request, current_app

predict_bp = Blueprint("predict", __name__)

# -----------------------------
# Utilities
# -----------------------------

def _project_root() -> Path:
    # backend/routes/predict.py -> 專案根目錄預期在 backend/ 的上一層
    here = Path(__file__).resolve()
    return here.parent.parent  # backend/

def db_path() -> str:
    env = os.getenv("DB_PATH")
    if env:
        return env
    # 預設：backend/data/mvp.sqlite3
    return str((_project_root() / "data" / "mvp.sqlite3").resolve())

def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    return conn

def ym_prev(ym: Optional[str]) -> str:
    # ym = "YYYY-MM"; 若空，取今天的上個月
    if not ym:
        today = dt.date.today().replace(day=1)
        prev = (today - dt.timedelta(days=1)).replace(day=1)
    else:
        y, m = map(int, ym.split("-"))
        base = dt.date(y, m, 1)
        prev = (base - dt.timedelta(days=1)).replace(day=1)
    return f"{prev.year:04d}-{prev.month:02d}"

def _get_month_range(ym: str) -> Tuple[str, str]:
    y, m = map(int, ym.split("-"))
    start = dt.date(y, m, 1)
    end = dt.date(y + (m // 12), (m % 12) + 1, 1)
    return (start.isoformat(), end.isoformat())

def has_table(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cur.fetchone() is not None

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

@dataclass
class ItemRow:
    sku: str
    name: str
    category: str
    price: float
    view_rate_pct: float  # 0~100
    score: float          # 用於正規化
    prob_pct: float       # 0~100

# -----------------------------
# Data fetchers
# -----------------------------

def fetch_catalog(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    """items(sku, name, category, price)"""
    if not has_table(conn, "items"):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for r in conn.execute("SELECT sku, name, category, price FROM items;"):
        out[r["sku"]] = {
            "sku": r["sku"],
            "name": r["name"],
            "category": r["category"],
            "price": _safe_float(r["price"]),
        }
    return out

def purchases_count_by_sku(
    conn: sqlite3.Connection,
    member_id: Optional[str],
    ym: str,
) -> Dict[str, int]:
    """回傳某月份的購買次數聚合；member_id=None 表示全站"""
    if not has_table(conn, "purchases"):
        return {}
    start, end = _get_month_range(ym)
    if member_id:
        sql = """
            SELECT sku, COUNT(*) AS cnt
            FROM purchases
            WHERE member_id = ? AND purchased_at >= ? AND purchased_at < ?
            GROUP BY sku
        """
        args = (member_id, start, end)
    else:
        sql = """
            SELECT sku, COUNT(*) AS cnt
            FROM purchases
            WHERE purchased_at >= ? AND purchased_at < ?
            GROUP BY sku
        """
        args = (start, end)
    cur = conn.execute(sql, args)
    return {r["sku"]: int(r["cnt"]) for r in cur.fetchall()}

def views_count_by_sku(conn: sqlite3.Connection, ym: str) -> Dict[str, int]:
    """views(sku, viewed_at)；缺表回空 dict"""
    if not has_table(conn, "views"):
        return {}
    start, end = _get_month_range(ym)
    cur = conn.execute(
        "SELECT sku, COUNT(*) AS v FROM views WHERE viewed_at >= ? AND viewed_at < ? GROUP BY sku",
        (start, end),
    )
    return {r["sku"]: int(r["v"]) for r in cur.fetchall()}

def member_level(conn: sqlite3.Connection, member_id: str) -> str:
    if not has_table(conn, "members"):
        return "guest"
    cur = conn.execute("SELECT COALESCE(level, 'general') AS lv FROM members WHERE member_id=?;", (member_id,))
    r = cur.fetchone()
    return (r["lv"] if r else "guest") or "guest"

# -----------------------------
# Endpoints
# -----------------------------

@predict_bp.get("/members/<member_id>/history")
def history(member_id: str):
    try:
        ym = request.args.get("ym")
        ym_last = ym_prev(ym)
        start, end = _get_month_range(ym_last)

        with connect() as conn:
            if not has_table(conn, "purchases"):
                return jsonify({"ok": True, "ym": ym_last, "rows": []})
            # 欄位名稱做兼容：若沒有 item/category/unit_price/quantity/total_price 也不應炸
            sql = """
                SELECT
                    purchased_at,
                    sku,
                    COALESCE(item, '') AS name,
                    COALESCE(category, '') AS category,
                    COALESCE(unit_price, 0) AS price,
                    COALESCE(quantity, 1) AS quantity,
                    COALESCE(total_price, COALESCE(unit_price,0) * COALESCE(quantity,1)) AS total_price
                FROM purchases
                WHERE member_id=? AND purchased_at >= ? AND purchased_at < ?
                ORDER BY purchased_at DESC
            """
            cur = conn.execute(sql, (member_id, start, end))
            rows = [dict(r) for r in cur.fetchall()]
        return jsonify({"ok": True, "ym": ym_last, "rows": rows})
    except Exception as e:
        current_app.logger.exception("history failed: %s", e)
        return jsonify({"ok": False, "error": str(e), "ym": request.args.get("ym")}), 200

@predict_bp.get("/members/<member_id>/predictions")
def predictions(member_id: str):
    """
    回傳 top-N 候選 + 機率(%)
    score = view_rate * conv_rate * time_decay * member_weight
    - 若無 views 表，以「上月全站購買數」近似 view 分母
    - 本月已購商品不再推薦
    """
    try:
        ym = request.args.get("ym")
        ym_last = ym_prev(ym)

        try:
            limit = int(request.args.get("limit", "7"))
            if limit <= 0 or limit > 50:
                limit = 7
        except Exception:
            limit = 7

        with connect() as conn:
            catalog = fetch_catalog(conn)
            if not catalog:
                return jsonify({"ok": True, "ym": ym_last, "rows": [], "member_level": "guest"})

            # 上月：全站瀏覽 / 會員購買
            v_all = views_count_by_sku(conn, ym_last)
            p_member_last = purchases_count_by_sku(conn, member_id, ym_last)

            # 本月已購（避免重推）
            this_ym = ym or dt.date.today().strftime("%Y-%m")
            p_this_month = purchases_count_by_sku(conn, member_id, this_ym)
            already = set(p_this_month.keys())

            # 若沒有 views，改用全站上月購買數當作近似
            if not v_all:
                v_all = purchases_count_by_sku(conn, None, ym_last)

            total_views = max(1, sum(v_all.values()))

            # 會員權重
            lv = member_level(conn, member_id)
            mw = {"vip": 1.2, "gold": 1.1, "general": 1.0, "guest": 0.95}.get(lv, 1.0)

            # 會員近 180 天最後一次購買（做 time decay）
            recent_last: Dict[str, str] = {}
            if has_table(conn, "purchases"):
                cur = conn.execute(
                    """SELECT sku, MAX(purchased_at) AS last_at
                       FROM purchases
                       WHERE member_id=? AND purchased_at >= date('now','-180 day')
                       GROUP BY sku;""",
                    (member_id,),
                )
                recent_last = {r["sku"]: r["last_at"] for r in cur.fetchall() if r["last_at"]}

            scored: List[ItemRow] = []
            for sku, meta in catalog.items():
                if sku in already:
                    continue  # 本月已購買就不推
                views = int(v_all.get(sku, 0))
                conv = int(p_member_last.get(sku, 0))

                view_rate = views / total_views if total_views else 0.0
                conv_rate = conv / max(1, views)  # 會員上月對該 SKU 的購買/瀏覽

                # time decay：最近 90 天買過衰減（越近衰減越大）
                decay = 1.0
                if sku in recent_last:
                    try:
                        last_date = dt.date.fromisoformat(recent_last[sku])
                        days = max(0, (dt.date.today() - last_date).days)
                        lam = 1.0 / 45.0
                        decay = math.exp(-lam * (90 - min(90, days)))
                    except Exception:
                        decay = 1.0

                score = max(0.0, view_rate * conv_rate * decay * mw)
                scored.append(
                    ItemRow(
                        sku=sku,
                        name=str(meta.get("name", "")),
                        category=str(meta.get("category", "")),
                        price=float(meta.get("price") or 0.0),
                        view_rate_pct=round(view_rate * 100.0, 2),
                        score=score,
                        prob_pct=0.0,
                    )
                )

            # 用「score」做正規化成百分比
            ssum = sum(i.score for i in scored) or 1.0
            for i in scored:
                i.prob_pct = round((i.score / ssum) * 100.0, 1)

            # 取前 N
            topn = sorted(scored, key=lambda x: x.prob_pct, reverse=True)[:limit]
            rows = [
                {
                    "sku": i.sku,
                    "name": i.name,
                    "category": i.category,
                    "price": i.price,
                    "view_rate": i.view_rate_pct,
                    "probability_percent": i.prob_pct,
                }
                for i in topn
            ]
            return jsonify({"ok": True, "ym": ym_last, "rows": rows, "member_level": lv})
    except Exception as e:
        current_app.logger.exception("predictions failed: %s", e)
        return jsonify({"ok": False, "error": str(e), "ym": request.args.get("ym")}), 200
