"""SQLite helper utilities for the ESP32-CAM MVP backend."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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

        candidates: list[tuple[str, FaceEncoding]] = []
        with self._connect() as conn:
            for row in conn.execute("SELECT member_id, encoding_json FROM members"):
                stored = FaceEncoding.from_jsonable(json.loads(row["encoding_json"]))
                if (
                    stored.signature
                    and encoding.signature
                    and stored.signature == encoding.signature
                ):
                    self._update_member_encoding(row["member_id"], stored, encoding)
                    return row["member_id"], 0.0
                candidates.append((row["member_id"], stored))
        candidate_lookup = {member_id: stored for member_id, stored in candidates}
        member_id, distance = recognizer.find_best_match(candidates, encoding)
        if member_id is not None and member_id in candidate_lookup and distance is not None:
            self._update_member_encoding(member_id, candidate_lookup[member_id], encoding)
        return member_id, distance

    def create_member(self, encoding: FaceEncoding, member_id: str | None = None) -> str:
        member_id = member_id or self._generate_member_id(encoding)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO members (member_id, encoding_json) VALUES (?, ?)",
                (member_id, json.dumps(encoding.to_jsonable(), ensure_ascii=False)),
            )
            conn.commit()
        _LOGGER.info("Created new member %s", member_id)
        return member_id

    def _generate_member_id(self, encoding: FaceEncoding | None = None) -> str:
        if encoding and encoding.signature:
            hashed = hashlib.sha1(encoding.signature.encode("utf-8")).hexdigest().upper()
            return f"MEM{hashed[:10]}"

        with self._connect() as conn:
            total_members = conn.execute("SELECT COUNT(*) FROM members").fetchone()[0]
        return f"MEM{total_members + 1:03d}"

    def _update_member_encoding(
        self,
        member_id: str,
        existing: FaceEncoding,
        new_encoding: FaceEncoding,
        blend_weight: float = 0.6,
    ) -> None:
        try:
            weight = float(np.clip(blend_weight, 0.0, 1.0))
        except (TypeError, ValueError):
            weight = 0.6

        try:
            if new_encoding.vector.size == 0:
                return

            if existing.vector.size > 0 and existing.vector.shape == new_encoding.vector.shape:
                base = existing.vector.astype(np.float32, copy=False)
                incoming = new_encoding.vector.astype(np.float32, copy=False)
                blended_vector = base * (1.0 - weight) + incoming * weight
            else:
                blended_vector = new_encoding.vector.astype(np.float32, copy=False)

            signature = existing.signature or new_encoding.signature
            if not signature and blended_vector.size > 0:
                signature = FaceRecognizer._hash_signature(blended_vector.tobytes())

            sources = [existing.source, new_encoding.source]
            merged_sources = "+".join(sorted({s for s in sources if s})) or "hash"

            updated = FaceEncoding(
                vector=blended_vector.astype(np.float32, copy=False),
                signature=signature,
                source=merged_sources,
            )

            with self._connect() as conn:
                conn.execute(
                    "UPDATE members SET encoding_json = ? WHERE member_id = ?",
                    (json.dumps(updated.to_jsonable(), ensure_ascii=False), member_id),
                )
                conn.commit()
            _LOGGER.info("更新會員 %s 的臉部向量 (來源=%s)", member_id, merged_sources)
        except Exception as exc:  # pragma: no cover - 保護性記錄
            _LOGGER.warning("更新會員 %s 向量失敗：%s", member_id, exc)

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
            import numpy as np

            encoding = FaceEncoding(
                np.zeros(128, dtype=np.float32),
                signature="demo-member",
                source="seed",
            )
            demo_member = self.create_member(encoding, member_id="MEMDEMO001")
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
