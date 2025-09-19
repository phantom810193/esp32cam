"""SQLite helper utilities for the ESP32-CAM MVP backend."""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .recognizer import FaceEncoding, FaceRecognizer

_LOGGER = logging.getLogger(__name__)


@dataclass
class Purchase:
    member_id: str
    item: str
    last_purchase: str
    discount: float
    recommendation: str


class Database:
    """Light-weight wrapper around SQLite used by the Flask backend."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS members (
                    member_id TEXT PRIMARY KEY,
                    encoding_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS purchases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    member_id TEXT NOT NULL,
                    item TEXT NOT NULL,
                    last_purchase TEXT NOT NULL,
                    discount REAL NOT NULL DEFAULT 0.0,
                    recommendation TEXT NOT NULL,
                    FOREIGN KEY(member_id) REFERENCES members(member_id)
                );
                """
            )

    # ------------------------------------------------------------------
    def find_member_by_encoding(
        self, encoding: FaceEncoding, recognizer: FaceRecognizer
    ) -> tuple[str | None, float | None]:
        """Return the best matching member for the provided encoding.

        Returns a tuple of ``(member_id, distance)``.  If no sufficiently close match
        is found the ``member_id`` will be ``None``.
        """

        best_member: str | None = None
        best_distance: float | None = None
        with self._connect() as conn:
            for row in conn.execute("SELECT member_id, encoding_json FROM members"):
                stored = FaceEncoding.from_jsonable(json.loads(row["encoding_json"]))
                distance = recognizer.distance(stored, encoding)
                if recognizer.is_match(stored, encoding):
                    if best_distance is None or distance < best_distance:
                        best_member = row["member_id"]
                        best_distance = distance
        return best_member, best_distance

    def create_member(self, encoding: FaceEncoding) -> str:
        member_id = self._generate_member_id()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO members (member_id, encoding_json) VALUES (?, ?)",
                (member_id, json.dumps(encoding.to_jsonable())),
            )
            conn.commit()
        _LOGGER.info("Created new member %s", member_id)
        return member_id

    def _generate_member_id(self) -> str:
        with self._connect() as conn:
            total_members = conn.execute("SELECT COUNT(*) FROM members").fetchone()[0]
        return f"MEM{total_members + 1:03d}"

    # ------------------------------------------------------------------
    def add_purchase(
        self, member_id: str, item: str, last_purchase: str, discount: float, recommendation: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO purchases (member_id, item, last_purchase, discount, recommendation)
                VALUES (?, ?, ?, ?, ?)
                """,
                (member_id, item, last_purchase, discount, recommendation),
            )
            conn.commit()

    def get_purchase_history(self, member_id: str) -> list[Purchase]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT member_id, item, last_purchase, discount, recommendation"
                " FROM purchases WHERE member_id = ? ORDER BY id DESC",
                (member_id,),
            ).fetchall()
        return [
            Purchase(
                member_id=row["member_id"],
                item=row["item"],
                last_purchase=row["last_purchase"],
                discount=float(row["discount"]),
                recommendation=row["recommendation"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    def ensure_demo_data(self) -> None:
        """Populate the database with a few demo purchases if empty."""

        with self._connect() as conn:
            has_members = conn.execute("SELECT COUNT(*) FROM members").fetchone()[0] > 0
            has_purchases = conn.execute("SELECT COUNT(*) FROM purchases").fetchone()[0] > 0

        if has_members and has_purchases:
            return

        if not has_members:
            # Create a synthetic member using a deterministic encoding so it works both
            # with real and hash based recognition.
            from .recognizer import FaceEncoding
            import numpy as np

            encoding = FaceEncoding(np.zeros(128, dtype=np.float32))
            demo_member = self.create_member(encoding)
        else:
            with self._connect() as conn:
                demo_member = conn.execute(
                    "SELECT member_id FROM members LIMIT 1"
                ).fetchone()[0]

        if not has_purchases:
            self.add_purchase(
                demo_member,
                "有機牛奶",
                "2023-12-18",
                0.1,
                "會員專屬 9 折優惠，今日加購麥片再折 20 元！",
            )
            self.add_purchase(
                demo_member,
                "手工吐司",
                "2023-11-02",
                0.15,
                "早餐搭配現磨咖啡可享第二杯半價。",
            )
            _LOGGER.info("Seeded demo purchase history for %s", demo_member)

