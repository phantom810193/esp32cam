"""SQLite helper utilities for the ESP32-CAM MVP backend."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
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
                if (
                    encoding.azure_persisted_face_id
                    and stored.azure_persisted_face_id
                    and stored.azure_persisted_face_id == encoding.azure_persisted_face_id
                ):
                    return row["member_id"], 0.0
                if encoding.azure_person_id and stored.azure_person_id:
                    if stored.azure_person_id == encoding.azure_person_id:
                        return row["member_id"], 0.0
                if encoding.azure_person_name:
                    if row["member_id"] == encoding.azure_person_name:
                        return row["member_id"], 0.0
                    if stored.azure_person_name == encoding.azure_person_name:
                        return row["member_id"], 0.0
                if (
                    stored.face_description
                    and stored.face_description == encoding.face_description
                ):
                    return row["member_id"], 0.0
                distance = recognizer.distance(stored, encoding)
                if recognizer.is_match(stored, encoding):
                    if best_distance is None or distance < best_distance:
                        best_member = row["member_id"]
                        best_distance = distance
        return best_member, best_distance

    def find_member_by_persisted_face_id(
        self, persisted_face_id: str
    ) -> tuple[str | None, FaceEncoding | None]:
        """Return the member with the given Azure persisted face identifier."""

        if not persisted_face_id:
            return None, None

        with self._connect() as conn:
            for row in conn.execute("SELECT member_id, encoding_json FROM members"):
                stored = FaceEncoding.from_jsonable(json.loads(row["encoding_json"]))
                if stored.azure_persisted_face_id == persisted_face_id:
                    return row["member_id"], stored
        return None, None

    def find_member_by_person_id(
        self, person_id: str
    ) -> tuple[str | None, FaceEncoding | None]:
        """Return the member registered with the specified Azure person identifier."""

        if not person_id:
            return None, None

        with self._connect() as conn:
            for row in conn.execute("SELECT member_id, encoding_json FROM members"):
                stored = FaceEncoding.from_jsonable(json.loads(row["encoding_json"]))
                if stored.azure_person_id == person_id:
                    return row["member_id"], stored
        return None, None

    def get_member_encoding(self, member_id: str) -> FaceEncoding | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT encoding_json FROM members WHERE member_id = ?",
                (member_id,),
            ).fetchone()
        if row is None:
            return None
        return FaceEncoding.from_jsonable(json.loads(row["encoding_json"]))

    def update_member_encoding(self, member_id: str, encoding: FaceEncoding) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE members SET encoding_json = ? WHERE member_id = ?",
                (json.dumps(encoding.to_jsonable(), ensure_ascii=False), member_id),
            )
            conn.commit()

    def create_member(self, encoding: FaceEncoding, member_id: str | None = None) -> str:
        """Persist a new member record, generating a unique ID when necessary."""

        payload = json.dumps(encoding.to_jsonable(), ensure_ascii=False)
        candidate_id = member_id or self._generate_member_id(encoding)

        try:
            self._insert_member(candidate_id, payload)
        except sqlite3.IntegrityError as exc:
            _LOGGER.warning("Member ID %s already exists, generating a new ID", candidate_id)
            last_error: sqlite3.IntegrityError | None = exc

            fallback_candidates = []
            sequential_id = self._generate_member_id(None)
            if sequential_id and sequential_id != candidate_id:
                fallback_candidates.append(sequential_id)

            # Guarantee progress even under concurrent inserts by appending
            # a few UUID based identifiers.
            for _ in range(5):
                fallback_candidates.append(self._generate_random_member_id())

            for fallback in fallback_candidates:
                try:
                    self._insert_member(fallback, payload)
                except sqlite3.IntegrityError as inner_exc:
                    last_error = inner_exc
                    continue
                else:
                    _LOGGER.info("Created new member %s", fallback)
                    return fallback

            raise last_error
        else:
            _LOGGER.info("Created new member %s", candidate_id)
            return candidate_id

    def _insert_member(self, member_id: str, payload: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO members (member_id, encoding_json) VALUES (?, ?)",
                (member_id, payload),
            )
            conn.commit()

    def _generate_random_member_id(self) -> str:
        return f"MEM{uuid.uuid4().hex[:10].upper()}"

    def _generate_member_id(self, encoding: FaceEncoding | None = None) -> str:
        if encoding and encoding.face_description:
            hashed = hashlib.sha1(encoding.face_description.encode("utf-8")).hexdigest().upper()
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
                face_description="demo-member",
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

