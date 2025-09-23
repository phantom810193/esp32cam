"""SQLite helper utilities for the ESP32-CAM MVP backend."""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .recognizer import FaceEncoding, FaceRecognizer

_LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA_PATH = Path(__file__).resolve().parent / "demo_members.json"


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
                if stored.signature and stored.signature == encoding.signature:
                    return row["member_id"], 0.0
                distance = recognizer.distance(stored, encoding)
                if recognizer.is_match(stored, encoding):
                    if best_distance is None or distance < best_distance:
                        best_member = row["member_id"]
                        best_distance = distance
        return best_member, best_distance

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
    def ensure_demo_data(self, recognizer: FaceRecognizer | None = None) -> None:
        """Populate the database with a few demo purchases if empty."""

        with self._connect() as conn:
            has_members = conn.execute("SELECT COUNT(*) FROM members").fetchone()[0] > 0
            has_purchases = (
                conn.execute("SELECT COUNT(*) FROM purchases").fetchone()[0] > 0
            )

        dataset = self._load_demo_dataset()
        if dataset:
            if not has_members:
                for entry in dataset:
                    member_id = entry["member_id"]
                    encoding = self._encoding_from_demo(entry, recognizer)
                    if encoding is None:
                        encoding = self._fallback_encoding(member_id)
                    self.create_member(encoding, member_id=member_id)

            if not has_purchases:
                for entry in dataset:
                    for purchase in entry.get("purchases", []):
                        try:
                            discount = float(purchase.get("discount", 0.0))
                        except (TypeError, ValueError):
                            discount = 0.0
                        self.add_purchase(
                            entry["member_id"],
                            str(purchase.get("item", "")) or "人氣商品",
                            str(purchase.get("last_purchase", "2024-01-01")),
                            discount,
                            str(purchase.get("recommendation", "歡迎再次光臨！")),
                        )
                if dataset:
                    _LOGGER.info(
                        "Seeded demo dataset with %d member(s)",
                        len(dataset),
                    )
            return

        if not has_members:
            encoding = self._fallback_encoding("MEMDEMO001")
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

    # ------------------------------------------------------------------
    def _load_demo_dataset(self) -> list[dict[str, Any]]:
        if not DEMO_DATA_PATH.exists():
            return []

        try:
            with DEMO_DATA_PATH.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            _LOGGER.warning("Unable to read demo dataset: %s", exc)
            return []

        if not isinstance(payload, list):
            _LOGGER.warning("Demo dataset must be a list of entries")
            return []

        entries: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            member_id = str(item.get("member_id", "")).strip()
            if not member_id:
                continue
            entry: dict[str, Any] = {
                "member_id": member_id,
                "image": item.get("image"),
                "purchases": item.get("purchases", []),
            }
            entries.append(entry)
        return entries

    def _encoding_from_demo(
        self, entry: dict[str, Any], recognizer: FaceRecognizer | None
    ) -> FaceEncoding | None:
        if recognizer is None:
            return None

        image_ref = entry.get("image")
        if not image_ref:
            return None

        image_path = Path(str(image_ref))
        if not image_path.is_absolute():
            candidate = PROJECT_ROOT / image_path
            if candidate.exists():
                image_path = candidate
            else:
                image_path = PROJECT_ROOT / "pictures" / image_path.name

        if not image_path.exists():
            _LOGGER.warning("Demo image not found: %s", image_path)
            return None

        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            _LOGGER.warning("Unable to read demo image %s: %s", image_path, exc)
            return None

        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        try:
            return recognizer.encode(image_bytes, mime_type=mime_type)
        except ValueError as exc:
            _LOGGER.warning("Unable to encode demo image %s: %s", image_path, exc)
            return None

    @staticmethod
    def _fallback_encoding(member_id: str) -> FaceEncoding:
        import numpy as np

        signature = f"demo-{member_id}"
        return FaceEncoding(
            np.zeros(128, dtype=np.float32),
            signature=signature,
            source="seed",
        )

