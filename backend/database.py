"""SQLite helper utilities for the ESP32-CAM MVP backend."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .recognizer import FaceEncoding, FaceRecognizer

_LOGGER = logging.getLogger(__name__)


MEMBER_CODE_OVERRIDES: dict[str, str] = {
    "MEME0383FE3AA": "ME0001",
    "MEM692FFD0824": "ME0002",
    "MEMFITNESS2025": "ME0003",
    "MEMHOMECARE2025": "",
    "MEMHEALTH2025": "",
}


_TAIPEI_TZ = ZoneInfo("Asia/Taipei")


@dataclass
class Purchase:
    member_id: str
    member_code: str
    item: str
    purchased_at: str
    unit_price: float
    quantity: float
    total_price: float


@dataclass
class MemberProfile:
    profile_id: int
    profile_label: str
    member_id: str | None
    mall_member_id: str | None
    member_status: str
    joined_at: str
    points_balance: float
    gender: str
    birth_date: str | None
    phone: str
    email: str
    address: str | None
    occupation: str | None


@dataclass
class UploadEvent:
    id: int
    created_at: str
    member_id: str
    member_code: str
    image_filename: str | None
    upload_duration: float
    recognition_duration: float
    ad_duration: float
    total_duration: float


def _build_seed_purchases(
    start_timestamp: str,
    member_code: str,
    items: list[tuple[str, float, float]],
) -> list[dict[str, float | str]]:
    base = datetime.fromisoformat(start_timestamp)
    purchases: list[dict[str, float | str]] = []
    for index, (name, unit_price, quantity) in enumerate(items):
        scheduled = base + timedelta(days=index * 3 + (index % 4), hours=index % 5, minutes=(index * 11) % 60)
        purchases.append(
            {
                "member_code": member_code,
                "item": name,
                "purchased_at": scheduled.strftime("%Y-%m-%d %H:%M"),
                "unit_price": float(unit_price),
                "quantity": float(quantity),
                "total_price": round(unit_price * quantity, 2),
            }
        )
    return purchases


def _blend_persona_items(
    persona_items: list[tuple[str, float, float]],
    lifestyle_items: list[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """Interleave persona-specific and general lifestyle goods at a 60/40 ratio."""

    blended: list[tuple[str, float, float]] = []
    persona_index = 0
    lifestyle_index = 0

    while persona_index < len(persona_items) or lifestyle_index < len(lifestyle_items):
        for _ in range(3):
            if persona_index >= len(persona_items):
                break
            blended.append(persona_items[persona_index])
            persona_index += 1
        for _ in range(2):
            if lifestyle_index >= len(lifestyle_items):
                break
            blended.append(lifestyle_items[lifestyle_index])
            lifestyle_index += 1

    return blended


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

    @staticmethod
    def _get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cursor.fetchall()]

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS members (
                    member_id TEXT PRIMARY KEY,
                    encoding_json TEXT NOT NULL
                );
                """
            )

            profile_columns = self._get_table_columns(conn, "member_profiles")
            expected_profile_columns = [
                "profile_id",
                "profile_label",
                "member_id",
                "mall_member_id",
                "member_status",
                "joined_at",
                "points_balance",
                "gender",
                "birth_date",
                "phone",
                "email",
                "address",
                "occupation",
            ]
            if profile_columns and profile_columns != expected_profile_columns:
                conn.execute("DROP TABLE IF EXISTS member_profiles")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS member_profiles (
                    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_label TEXT NOT NULL UNIQUE,
                    member_id TEXT UNIQUE,
                    mall_member_id TEXT,
                    member_status TEXT NOT NULL DEFAULT '有效',
                    joined_at TEXT NOT NULL,
                    points_balance REAL NOT NULL DEFAULT 0,
                    gender TEXT NOT NULL,
                    birth_date TEXT,
                    phone TEXT NOT NULL,
                    email TEXT NOT NULL,
                    address TEXT,
                    occupation TEXT,
                    FOREIGN KEY(member_id) REFERENCES members(member_id)
                );
                """
            )

            purchase_columns = self._get_table_columns(conn, "purchases")
            expected_columns = [
                "id",
                "member_id",
                "member_code",
                "purchased_at",
                "item",
                "unit_price",
                "quantity",
                "total_price",
            ]
            if purchase_columns and purchase_columns != expected_columns:
                conn.execute("DROP TABLE IF EXISTS purchases")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS purchases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    member_id TEXT NOT NULL,
                    member_code TEXT NOT NULL,
                    purchased_at TEXT NOT NULL,
                    item TEXT NOT NULL,
                    unit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    total_price REAL NOT NULL,
                    FOREIGN KEY(member_id) REFERENCES members(member_id)
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS upload_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    member_id TEXT NOT NULL,
                    image_filename TEXT,
                    upload_duration REAL NOT NULL,
                    recognition_duration REAL NOT NULL,
                    ad_duration REAL NOT NULL,
                    total_duration REAL NOT NULL,
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
                if encoding.signature and encoding.signature == row["member_id"]:
                    return row["member_id"], 0.0
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
        self._claim_unassigned_profile(member_id)
        return member_id

    def update_member_encoding(self, member_id: str, encoding: FaceEncoding) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE members SET encoding_json = ? WHERE member_id = ?",
                (json.dumps(encoding.to_jsonable(), ensure_ascii=False), member_id),
            )
            conn.commit()
        _LOGGER.info("Updated Rekognition encoding for member %s", member_id)

    def get_member_encoding(self, member_id: str) -> FaceEncoding | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT encoding_json FROM members WHERE member_id = ?",
                (member_id,),
            ).fetchone()
        if not row:
            return None
        return FaceEncoding.from_jsonable(json.loads(row["encoding_json"]))

    def delete_member(self, member_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM members WHERE member_id = ?",
                (member_id,),
            )
            conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            _LOGGER.info("Deleted member %s", member_id)
        return deleted

    def merge_members(self, source_id: str, target_id: str) -> tuple[FaceEncoding, FaceEncoding]:
        """Merge ``source_id`` into ``target_id`` and return their encodings."""

        if source_id == target_id:
            raise ValueError("source_id 與 target_id 不可相同")

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            source_row = conn.execute(
                "SELECT encoding_json FROM members WHERE member_id = ?",
                (source_id,),
            ).fetchone()
            target_row = conn.execute(
                "SELECT encoding_json FROM members WHERE member_id = ?",
                (target_id,),
            ).fetchone()

            if source_row is None:
                raise ValueError(f"找不到來源會員 {source_id}")
            if target_row is None:
                raise ValueError(f"找不到目標會員 {target_id}")

            source_encoding = FaceEncoding.from_jsonable(
                json.loads(source_row["encoding_json"])
            )
            target_encoding = FaceEncoding.from_jsonable(
                json.loads(target_row["encoding_json"])
            )

            conn.execute(
                "UPDATE purchases SET member_id = ?, member_code = ? WHERE member_id = ?",
                (target_id, self.get_member_code(target_id), source_id),
            )
            conn.execute(
                "UPDATE member_profiles SET member_id = ? WHERE member_id = ?",
                (target_id, source_id),
            )
            conn.execute(
                "DELETE FROM members WHERE member_id = ?",
                (source_id,),
            )
            conn.commit()

        _LOGGER.info("Merged member %s into %s", source_id, target_id)
        return source_encoding, target_encoding

    def _generate_member_id(self, encoding: FaceEncoding | None = None) -> str:
        if encoding and encoding.signature:
            hashed = hashlib.sha1(encoding.signature.encode("utf-8")).hexdigest().upper()
            return f"MEM{hashed[:10]}"

        with self._connect() as conn:
            total_members = conn.execute("SELECT COUNT(*) FROM members").fetchone()[0]
        return f"MEM{total_members + 1:03d}"

    # ------------------------------------------------------------------
    def _claim_unassigned_profile(self, member_id: str) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_id FROM member_profiles WHERE member_id IS NULL ORDER BY profile_id LIMIT 1"
            ).fetchone()
            if row is None:
                return

            profile_id = int(row["profile_id"])
            conn.execute(
                "UPDATE member_profiles SET member_id = ? WHERE profile_id = ?",
                (member_id, profile_id),
            )
            conn.commit()
        _LOGGER.info("Assigned new member %s to pre-seeded profile %s", member_id, profile_id)

    def get_member_profile(self, member_id: str) -> MemberProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT profile_id,
                       profile_label,
                       member_id,
                       mall_member_id,
                       member_status,
                       joined_at,
                       points_balance,
                       gender,
                       birth_date,
                       phone,
                       email,
                       address,
                       occupation
                FROM member_profiles
                WHERE member_id = ?
                """,
                (member_id,),
            ).fetchone()

        if row is None:
            return None

        return MemberProfile(
            profile_id=int(row["profile_id"]),
            profile_label=str(row["profile_label"]),
            member_id=str(row["member_id"]) if row["member_id"] else None,
            mall_member_id=str(row["mall_member_id"]) if row["mall_member_id"] else None,
            member_status=str(row["member_status"]),
            joined_at=str(row["joined_at"]),
            points_balance=float(row["points_balance"]),
            gender=str(row["gender"]),
            birth_date=str(row["birth_date"]) if row["birth_date"] else None,
            phone=str(row["phone"]),
            email=str(row["email"]),
            address=str(row["address"]) if row["address"] else None,
            occupation=str(row["occupation"]) if row["occupation"] else None,
        )

    def list_member_profiles(self) -> list[MemberProfile]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT profile_id,
                       profile_label,
                       member_id,
                       mall_member_id,
                       member_status,
                       joined_at,
                       points_balance,
                       gender,
                       birth_date,
                       phone,
                       email,
                       address,
                       occupation
                FROM member_profiles
                ORDER BY profile_id
                """
            ).fetchall()

        profiles: list[MemberProfile] = []
        for row in rows:
            profiles.append(
                MemberProfile(
                    profile_id=int(row["profile_id"]),
                    profile_label=str(row["profile_label"]),
                    member_id=str(row["member_id"]) if row["member_id"] else None,
                    mall_member_id=str(row["mall_member_id"]) if row["mall_member_id"] else None,
                    member_status=str(row["member_status"]),
                    joined_at=str(row["joined_at"]),
                    points_balance=float(row["points_balance"]),
                    gender=str(row["gender"]),
                    birth_date=str(row["birth_date"]) if row["birth_date"] else None,
                    phone=str(row["phone"]),
                    email=str(row["email"]),
                    address=str(row["address"]) if row["address"] else None,
                    occupation=str(row["occupation"]) if row["occupation"] else None,
                )
            )

        return profiles

    def get_member_code(self, member_id: str) -> str:
        profile = self.get_member_profile(member_id)
        if profile:
            if profile.mall_member_id:
                return profile.mall_member_id
            return ""

        override = MEMBER_CODE_OVERRIDES.get(member_id)
        if override is not None:
            return override
        with self._connect() as conn:
            row = conn.execute(
                "SELECT member_code FROM purchases WHERE member_id = ? ORDER BY purchased_at DESC, id DESC LIMIT 1",
                (member_id,),
            ).fetchone()
        if row is not None:
            code = row["member_code"]
            if code:
                return str(code)
        # 新產生的匿名會員預設沒有商場註冊代號，因此這裡不再回傳內部
        # ``MEM`` 編號的推導值，而是以空字串表示尚未綁定。
        return ""

    # ------------------------------------------------------------------
    def add_purchase(
        self,
        member_id: str,
        *,
        member_code: str | None = None,
        item: str,
        purchased_at: str,
        unit_price: float,
        quantity: float,
        total_price: float,
    ) -> None:
        if member_code is None:
            resolved_code = self.get_member_code(member_id)
        else:
            resolved_code = member_code
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO purchases (
                    member_id,
                    member_code,
                    purchased_at,
                    item,
                    unit_price,
                    quantity,
                    total_price
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    member_id,
                    resolved_code,
                    purchased_at,
                    item,
                    float(unit_price),
                    float(quantity),
                    float(total_price),
                ),
            )
            conn.commit()

    def get_purchase_history(self, member_id: str) -> list[Purchase]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT member_id, member_code, purchased_at, item, unit_price, quantity, total_price"
                " FROM purchases WHERE member_id = ? ORDER BY purchased_at DESC, id DESC",
                (member_id,),
            ).fetchall()
        return [
            Purchase(
                member_id=row["member_id"],
                member_code=row["member_code"],
                item=row["item"],
                purchased_at=row["purchased_at"],
                unit_price=float(row["unit_price"]),
                quantity=float(row["quantity"]),
                total_price=float(row["total_price"]),
            )
            for row in rows
        ]

    def record_upload_event(
        self,
        *,
        member_id: str,
        image_filename: str | None,
        upload_duration: float,
        recognition_duration: float,
        ad_duration: float,
        total_duration: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO upload_events (
                    created_at,
                    member_id,
                    image_filename,
                    upload_duration,
                    recognition_duration,
                    ad_duration,
                    total_duration
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(_TAIPEI_TZ).isoformat(timespec="seconds"),
                    member_id,
                    image_filename,
                    float(upload_duration),
                    float(recognition_duration),
                    float(ad_duration),
                    float(total_duration),
                ),
            )
            conn.commit()

    def cleanup_upload_events(self, keep_latest: int = 1) -> list[str]:
        """Trim stored upload events and return filenames that should be deleted.

        Only the ``keep_latest`` most recent entries are retained.  Any associated
        image filenames for older events are returned so the caller can remove the
        corresponding files from disk.
        """

        keep_latest = max(0, int(keep_latest))
        with self._connect() as conn:
            ids_to_keep: list[int] = []
            if keep_latest:
                rows = conn.execute(
                    "SELECT id FROM upload_events ORDER BY id DESC LIMIT ?",
                    (keep_latest,),
                ).fetchall()
                ids_to_keep = [int(row["id"]) for row in rows]

            params: tuple[int, ...] | None
            if ids_to_keep:
                placeholders = ",".join("?" for _ in ids_to_keep)
                params = tuple(ids_to_keep)
                query = (
                    "SELECT id, image_filename FROM upload_events "
                    f"WHERE id NOT IN ({placeholders})"
                )
            else:
                query = "SELECT id, image_filename FROM upload_events"
                params = None

            if params is None:
                rows = conn.execute(query).fetchall()
            else:
                rows = conn.execute(query, params).fetchall()
            if not rows:
                return []

            ids_to_delete = [int(row["id"]) for row in rows]
            filenames = [row["image_filename"] for row in rows if row["image_filename"]]

            delete_placeholders = ",".join("?" for _ in ids_to_delete)
            conn.execute(
                f"DELETE FROM upload_events WHERE id IN ({delete_placeholders})",
                tuple(ids_to_delete),
            )
            conn.commit()

        return filenames

    def get_latest_upload_event(self) -> UploadEvent | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id,
                       created_at,
                       member_id,
                       image_filename,
                       upload_duration,
                       recognition_duration,
                       ad_duration,
                       total_duration
                FROM upload_events
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None

        created_at = str(row["created_at"])
        try:
            created_dt = datetime.fromisoformat(created_at)
        except ValueError:
            created_dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=_TAIPEI_TZ)

        return UploadEvent(
            id=int(row["id"]),
            created_at=created_dt.astimezone(_TAIPEI_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            member_id=str(row["member_id"]),
            member_code=self.get_member_code(str(row["member_id"])),
            image_filename=row["image_filename"],
            upload_duration=float(row["upload_duration"]),
            recognition_duration=float(row["recognition_duration"]),
            ad_duration=float(row["ad_duration"]),
            total_duration=float(row["total_duration"]),
        )

    # ------------------------------------------------------------------
    def ensure_demo_data(self) -> None:
        """Seed deterministic purchase histories for demo members."""

        dessert_persona_items: list[tuple[str, float, float]] = [
            ("草莓千層蛋糕", 320.0, 1),
            ("香草可麗露禮盒", 480.0, 1),
            ("抹茶生乳捲", 280.0, 1),
            ("焦糖海鹽布蕾", 95.0, 2),
            ("蜂蜜檸檬磅蛋糕", 210.0, 1),
            ("法式莓果塔", 260.0, 1),
            ("生巧克力布朗尼", 180.0, 2),
            ("芒果生乳酪杯", 150.0, 2),
            ("伯爵茶瑪德蓮", 85.0, 4),
            ("紅絲絨杯子蛋糕", 120.0, 2),
            ("巴斯克乳酪蛋糕", 420.0, 1),
            ("桂花烏龍奶酪", 110.0, 2),
            ("藍莓優格慕斯", 135.0, 2),
            ("榛果可麗餅捲", 160.0, 2),
            ("柚香乳酪塔", 240.0, 1),
            ("抹茶紅豆鬆餅", 150.0, 2),
            ("花生黑糖奶酪", 105.0, 2),
            ("玫瑰荔枝蛋糕", 360.0, 1),
            ("太妃焦糖蘋果塔", 280.0, 1),
            ("香蕉核桃麵包布丁", 180.0, 2),
            ("海鹽奶油司康", 90.0, 4),
            ("柑橘乳酪生乳捲", 295.0, 1),
            ("法芙娜巧克力塔", 340.0, 1),
            ("葡萄柚優格杯", 140.0, 2),
            ("黑糖波士頓派", 330.0, 1),
            ("餅乾奶油杯", 95.0, 3),
            ("草莓生乳酪塔", 260.0, 1),
            ("伯爵奶茶布丁", 110.0, 2),
            ("楓糖肉桂捲", 85.0, 4),
            ("抹茶巴菲杯", 165.0, 2),
        ]
        dessert_lifestyle_items: list[tuple[str, float, float]] = [
            ("精品手沖咖啡豆", 520.0, 1),
            ("手作果醬三入組", 450.0, 1),
            ("嚴選花草茶禮盒", 680.0, 1),
            ("有機燕麥早餐罐", 180.0, 2),
            ("冷萃咖啡瓶裝禮盒", 360.0, 1),
            ("產地直送有機蔬菜箱", 880.0, 1),
            ("季節鮮採綜合水果箱", 720.0, 1),
            ("當季鮮採藍莓盒", 250.0, 1),
            ("溫室小黃瓜三入組", 95.0, 1),
            ("冷凍鮭魚切片家庭包", 560.0, 1),
            ("家用環保洗碗精補充包", 150.0, 2),
            ("柔感棉質廚房紙巾組", 320.0, 1),
            ("天然海鹽烹飪罐", 150.0, 1),
            ("舒眠香氛蠟燭", 320.0, 1),
            ("有機鮮乳家庭箱", 260.0, 1),
            ("無糖優酪乳六入", 220.0, 1),
            ("不沾煎鍋28CM", 880.0, 1),
            ("家用濾水壺", 520.0, 1),
            ("多功能香料罐組", 360.0, 1),
            ("全麥吐司家庭包", 120.0, 2),
        ]
        dessert_specs = _blend_persona_items(dessert_persona_items, dessert_lifestyle_items)

        kids_persona_items: list[tuple[str, float, float]] = [
            ("幼兒律動課體驗券", 680.0, 1),
            ("親子烘焙下午茶套票", 1180.0, 1),
            ("益智積木組", 450.0, 1),
            ("幼幼圖卡教材", 320.0, 1),
            ("幼兒園夏令營報名費", 5200.0, 1),
            ("親子瑜伽月票", 1680.0, 1),
            ("木製拼圖組", 560.0, 1),
            ("幼兒繪本套書", 1380.0, 1),
            ("幼兒科學實驗盒", 680.0, 1),
            ("幼兒園延托服務時數", 420.0, 5),
            ("兒童才藝試上課程", 780.0, 1),
            ("親子劇場週末票", 980.0, 1),
            ("幼兒律動課教材包", 460.0, 1),
            ("幼兒園制服組", 890.0, 1),
            ("家庭園遊會餐券", 350.0, 3),
            ("學前美語體驗課", 880.0, 1),
            ("幼兒陶土手作課", 720.0, 1),
            ("幼兒園校車月票", 2800.0, 1),
            ("幼兒籃球體驗營", 1650.0, 1),
            ("兒童營養午餐組", 120.0, 10),
            ("幼兒園畢業紀念冊預購", 550.0, 1),
            ("親子攝影紀念套組", 2680.0, 1),
            ("幼兒戶外探索課程", 1350.0, 1),
            ("兒童舞蹈公演票", 880.0, 2),
            ("幼兒安全防走失背包", 960.0, 1),
            ("幼兒園科學週材料包", 420.0, 1),
            ("親子閱讀早午餐套票", 920.0, 1),
            ("幼兒園節慶禮盒", 680.0, 1),
            ("幼兒園音樂會門票", 750.0, 2),
            ("親子共學木工課", 1450.0, 1),
        ]
        kids_lifestyle_items: list[tuple[str, float, float]] = [
            ("家庭健康維他命組", 850.0, 1),
            ("週末市集有機蔬菜箱", 980.0, 1),
            ("家用濾水壺替換濾芯", 450.0, 2),
            ("智能體重計", 1650.0, 1),
            ("無線吸塵器濾網組", 620.0, 1),
            ("家庭常備洗衣精補充包", 320.0, 3),
            ("旅行收納壓縮袋組", 560.0, 1),
            ("全家早餐穀物禮盒", 420.0, 2),
            ("季節鮮果禮盒", 880.0, 1),
            ("家庭露營炊具套組", 1980.0, 1),
            ("家庭號沐浴乳三入組", 360.0, 1),
            ("親子保溫水壺雙入", 520.0, 1),
            ("多用途餐桌防水墊", 280.0, 1),
            ("家庭常備繃帶組", 180.0, 1),
            ("親子戶外防曬乳", 450.0, 1),
            ("智慧家電延長線", 320.0, 1),
            ("客廳香氛擴香瓶", 420.0, 1),
            ("天然洗手乳補充包", 260.0, 2),
            ("居家整理收納箱組", 580.0, 1),
            ("家庭號即食玉米濃湯", 150.0, 3),
        ]
        kids_specs = _blend_persona_items(kids_persona_items, kids_lifestyle_items)

        fitness_specs: list[tuple[str, float, float]] = [
            ("高蛋白乳清粉", 1280.0, 1),
            ("肌力訓練彈力帶組", 420.0, 1),
            ("健身房月卡", 1680.0, 1),
            ("運動機能背心", 780.0, 1),
            ("速乾運動毛巾三入組", 360.0, 1),
            ("低脂優格家庭箱", 480.0, 1),
            ("能量燕麥棒禮盒", 280.0, 1),
            ("跑步智能手錶", 3680.0, 1),
            ("筋膜按摩滾筒", 950.0, 1),
            ("壺鈴訓練組15KG", 1680.0, 1),
            ("室內跳繩防滑墊", 520.0, 1),
            ("舒肥雞胸冷凍餐", 320.0, 3),
            ("防滑瑜珈墊", 980.0, 1),
            ("運動壓縮襪", 420.0, 2),
            ("健康蔬果汁禮盒", 620.0, 1),
            ("健身料理玻璃保鮮盒組", 780.0, 1),
            ("BCAA胺基酸飲", 680.0, 1),
            ("登山補給凍乾水果", 360.0, 1),
            ("強化護腕", 450.0, 1),
            ("多功能健腹輪", 820.0, 1),
            ("無糖豆漿箱", 360.0, 1),
            ("健康即食藜麥包", 420.0, 2),
            ("室內腳踏車訓練器租賃", 1980.0, 1),
            ("全穀饅頭六入", 150.0, 2),
            ("運動水壺雙入組", 520.0, 1),
            ("極速冷感運動帽", 680.0, 1),
            ("防滑健身手套", 420.0, 1),
            ("戶外越野襪三入", 380.0, 1),
            ("植物蛋白飲禮盒", 560.0, 1),
            ("冷壓橄欖油家庭瓶", 680.0, 1),
            ("有機藍莓盒", 220.0, 1),
            ("智能體脂計", 1480.0, 1),
            ("高纖蔬菜湯組", 360.0, 2),
            ("雞蛋白營養飲", 420.0, 1),
            ("運動耳機防汗版", 2980.0, 1),
            ("夜跑LED臂帶", 320.0, 1),
            ("山藥雞湯即食包", 260.0, 2),
            ("家用洗碗機洗劑", 450.0, 1),
            ("天然洗衣精補充包", 320.0, 2),
            ("旅行收納健身包", 880.0, 1),
            ("居家伸展彈力椅", 1350.0, 1),
            ("碳酸鎂粉補充包", 220.0, 1),
            ("海鹽堅果能量包", 260.0, 2),
            ("綜合沙拉葉家庭包", 180.0, 2),
            ("舒眠草本茶禮盒", 520.0, 1),
            ("多功能果汁機", 2280.0, 1),
            ("冬季保暖運動外套", 1980.0, 1),
            ("戶外蛋白餅乾", 180.0, 2),
            ("家用保溫餐盒", 560.0, 1),
            ("伸展瑜珈磚", 360.0, 1),
        ]

        homemaker_specs: list[tuple[str, float, float]] = [
            ("多功能電鍋", 1680.0, 1),
            ("家庭保鮮盒12件組", 520.0, 1),
            ("天然洗衣精補充包", 320.0, 3),
            ("兒童學習餐具組", 420.0, 1),
            ("客廳防滑地墊", 680.0, 1),
            ("智能掃地機濾網", 450.0, 1),
            ("廚房去油劑雙入", 260.0, 1),
            ("家庭號衛生紙箱", 320.0, 1),
            ("有機雞蛋家庭盒", 180.0, 2),
            ("當季蔬菜禮籃", 520.0, 1),
            ("兒童英文繪本組", 680.0, 1),
            ("烘焙常備粉組", 260.0, 2),
            ("廚房紙巾超值組", 360.0, 1),
            ("家庭常備藥品組", 480.0, 1),
            ("保溫便當盒雙層", 420.0, 1),
            ("兒童益智拼圖", 380.0, 1),
            ("天然酵素清潔液", 450.0, 1),
            ("家庭車用收納箱", 520.0, 1),
            ("香氛洗手乳三入", 320.0, 1),
            ("客廳抱枕套組", 420.0, 1),
            ("烘碗機除菌濾網", 280.0, 1),
            ("兒童室內拖鞋", 260.0, 2),
            ("家庭急救包", 620.0, 1),
            ("氣炸鍋烘烤紙", 150.0, 2),
            ("親子烹飪課程券", 1280.0, 1),
            ("冷凍餃子家庭包", 320.0, 2),
            ("早餐穀片超值箱", 360.0, 1),
            ("保鮮袋100入", 180.0, 1),
            ("兒童成長牛奶", 420.0, 2),
            ("家用滅菌噴霧", 380.0, 1),
            ("天然蜂蜜禮盒", 560.0, 1),
            ("客廳收納籃組", 450.0, 1),
            ("家庭號冷凍鮭魚", 520.0, 1),
            ("季節水果拼盤", 420.0, 1),
            ("小家庭燒烤盤", 780.0, 1),
            ("兒童畫畫教材組", 360.0, 1),
            ("除濕包超值組", 280.0, 2),
            ("家庭用吸塵器濾網", 420.0, 1),
            ("親子桌遊禮盒", 680.0, 1),
            ("天然醬油組", 320.0, 1),
            ("家庭號優格桶", 280.0, 1),
            ("舒眠草本茶", 260.0, 2),
            ("有機米禮盒", 680.0, 1),
            ("保暖親子毛毯", 780.0, 1),
            ("兒童雨衣靴組", 620.0, 1),
            ("家庭常備電池組", 360.0, 1),
            ("廚房玻璃調味罐", 280.0, 1),
            ("居家香氛噴霧", 450.0, 1),
            ("蔬果保鮮盒組", 360.0, 1),
            ("家庭烘焙模具", 320.0, 1),
            ("親子野餐籃", 520.0, 1),
            ("天然洗碗皂", 220.0, 2),
        ]

        health_specs: list[tuple[str, float, float]] = [
            ("有機冷壓亞麻仁油", 620.0, 1),
            ("高纖燕麥片禮盒", 360.0, 1),
            ("綜合堅果禮罐", 520.0, 1),
            ("植物基蛋白飲", 680.0, 1),
            ("綠拿鐵冷壓汁", 260.0, 2),
            ("益生菌粉末盒", 820.0, 1),
            ("有機羽衣甘藍", 220.0, 2),
            ("低溫烘焙杏仁", 360.0, 1),
            ("糙米能量棒", 280.0, 2),
            ("高鈣無糖豆漿", 320.0, 1),
            ("藜麥綜合穀物飯", 380.0, 2),
            ("低GI紫米麵包", 260.0, 2),
            ("海藻鈣膠囊", 780.0, 1),
            ("有機甜菜根粉", 420.0, 1),
            ("天然莓果乾", 320.0, 2),
            ("膳食纖維飲品", 450.0, 1),
            ("有機小農蔬菜箱", 880.0, 1),
            ("優格發酵菌粉", 350.0, 1),
            ("天然蜂膠滴劑", 560.0, 1),
            ("低溫烘焙腰果", 420.0, 1),
            ("紅藜健康米", 420.0, 1),
            ("綠茶多酚飲", 320.0, 2),
            ("有機高麗菜", 160.0, 2),
            ("全穀燕麥奶", 280.0, 2),
            ("天然蔓越莓汁", 360.0, 2),
            ("有機黑芝麻粉", 380.0, 1),
            ("暖薑黑糖飲", 280.0, 2),
            ("高蛋白豆腐組", 220.0, 2),
            ("冷壓胡蘿蔔汁", 260.0, 2),
            ("綠色蔬果粉", 520.0, 1),
            ("膠原蛋白飲", 780.0, 1),
            ("天然薄荷茶", 260.0, 2),
            ("發芽糙米", 320.0, 2),
            ("有機酪梨禮盒", 620.0, 1),
            ("全植營養補充錠", 980.0, 1),
            ("健康烤地瓜片", 180.0, 2),
            ("有機藍莓醬", 320.0, 1),
            ("燕麥豆奶布丁", 260.0, 2),
            ("純淨礦泉水箱", 280.0, 1),
            ("天然葡萄籽油", 560.0, 1),
            ("無糖椰子水", 320.0, 2),
            ("高纖蒟蒻麵", 280.0, 2),
            ("有機南瓜", 180.0, 2),
            ("保溫隨行杯", 420.0, 1),
            ("天然洗衣粉", 320.0, 1),
            ("香草舒眠枕噴霧", 450.0, 1),
            ("健康烹飪蒸籠", 680.0, 1),
            ("有機鷹嘴豆", 260.0, 2),
            ("純素黑巧克力", 320.0, 1),
            ("天然洗碗精", 280.0, 1),
            ("冷壓椰子油", 520.0, 1),
            ("有機檸檬禮盒", 360.0, 1),
        ]

        dessert_history = _build_seed_purchases(
            "2025-01-04 10:30",
            self.get_member_code("MEME0383FE3AA"),
            dessert_specs,
        )
        kids_history = _build_seed_purchases(
            "2025-01-05 09:20",
            self.get_member_code("MEM692FFD0824"),
            kids_specs,
        )

        fitness_history = _build_seed_purchases(
            "2025-01-06 07:30",
            self.get_member_code("MEMFITNESS2025"),
            fitness_specs,
        )
        homemaker_history = _build_seed_purchases(
            "2025-01-08 08:45",
            self.get_member_code("MEMHOMECARE2025"),
            homemaker_specs,
        )
        health_history = _build_seed_purchases(
            "2025-01-09 09:10",
            self.get_member_code("MEMHEALTH2025"),
            health_specs,
        )

        self._seed_member_history("MEME0383FE3AA", dessert_history)
        self._seed_member_history("MEM692FFD0824", kids_history)
        self._seed_member_history("MEMFITNESS2025", fitness_history)
        self._seed_member_history("MEMHOMECARE2025", homemaker_history)
        self._seed_member_history("MEMHEALTH2025", health_history)

        self._seed_member_profile(
            profile_label="dessert-lover",
            member_id="MEME0383FE3AA",
            mall_member_id="ME0001",
            member_status="有效",
            joined_at="2021-06-12",
            points_balance=1520,
            gender="女",
            birth_date="1988-07-12",
            phone="0912-345-678",
            email="dessertlover@example.com",
            address="台北市信義區松壽路10號",
            occupation="甜點教室講師",
        )
        self._seed_member_profile(
            profile_label="family-groceries",
            member_id="MEM692FFD0824",
            mall_member_id="ME0002",
            member_status="有效",
            joined_at="2020-09-01",
            points_balance=980,
            gender="女",
            birth_date="1990-02-08",
            phone="0923-556-789",
            email="familybuyer@example.com",
            address="新北市板橋區文化路100號",
            occupation="幼兒園老師",
        )
        self._seed_member_profile(
            profile_label="fitness-enthusiast",
            member_id="MEMFITNESS2025",
            mall_member_id="ME0003",
            member_status="有效",
            joined_at="2019-11-20",
            points_balance=2040,
            gender="男",
            birth_date="1985-04-19",
            phone="0955-112-233",
            email="fitgoer@example.com",
            address="台中市西屯區市政北二路88號",
            occupation="企業健身顧問",
        )
        self._seed_member_profile(
            profile_label="home-manager",
            member_id="MEMHOMECARE2025",
            mall_member_id="",
            member_status="有效",
            joined_at="2023-03-18",
            points_balance=640,
            gender="女",
            birth_date=None,
            phone="0977-334-556",
            email="homemanager@example.com",
            address="桃園市桃園區同德五街66號",
            occupation=None,
        )
        self._seed_member_profile(
            profile_label="wellness-gourmet",
            member_id="MEMHEALTH2025",
            mall_member_id="",
            member_status="有效",
            joined_at="2024-01-05",
            points_balance=520,
            gender="男",
            birth_date="1992-10-02",
            phone="0966-778-990",
            email="healthbuyer@example.com",
            address=None,
            occupation="營養顧問",
        )

        with self._connect() as conn:
            has_members = conn.execute("SELECT COUNT(*) FROM members").fetchone()[0] > 0
            has_purchases = conn.execute("SELECT COUNT(*) FROM purchases").fetchone()[0] > 0

        if not has_members:
            import numpy as np

            encoding = FaceEncoding(
                np.zeros(128, dtype=np.float32),
                signature="demo-member",
                source="seed",
            )
            self.create_member(encoding, member_id="MEMDEMO001")

        if not has_purchases:
            now = datetime.now().replace(second=0, microsecond=0)
            self.add_purchase(
                "MEMDEMO001",
                item="有機牛奶",
                purchased_at=now.strftime("%Y-%m-%d %H:%M"),
                unit_price=120.0,
                quantity=1,
                total_price=120.0,
            )
            _LOGGER.info("Seeded fallback purchase history for MEMDEMO001")

    def _seed_member_history(
        self, member_id: str, purchases: list[dict[str, float | str]]
    ) -> None:
        with self._connect() as conn:
            exists = (
                conn.execute("SELECT 1 FROM members WHERE member_id = ?", (member_id,)).fetchone()
                is not None
            )

        if not exists:
            import numpy as np

            encoding = FaceEncoding(
                np.zeros(128, dtype=np.float32),
                signature=member_id,
                source="seed",
            )
            self.create_member(encoding, member_id=member_id)

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM purchases WHERE member_id = ? AND purchased_at LIKE '2025-%'",
                (member_id,),
            )
            conn.commit()

        for purchase in purchases:
            self.add_purchase(member_id, **purchase)

    def _seed_member_profile(
        self,
        *,
        profile_label: str,
        member_id: str | None,
        mall_member_id: str | None,
        member_status: str,
        joined_at: str,
        points_balance: float,
        gender: str,
        birth_date: str | None,
        phone: str,
        email: str,
        address: str | None,
        occupation: str | None,
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_id FROM member_profiles WHERE profile_label = ?",
                (profile_label,),
            ).fetchone()

            params = (
                member_id,
                mall_member_id,
                member_status,
                joined_at,
                float(points_balance),
                gender,
                birth_date,
                phone,
                email,
                address,
                occupation,
            )

            if row is None:
                conn.execute(
                    """
                    INSERT INTO member_profiles (
                        profile_label,
                        member_id,
                        mall_member_id,
                        member_status,
                        joined_at,
                        points_balance,
                        gender,
                        birth_date,
                        phone,
                        email,
                        address,
                        occupation
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (profile_label, *params),
                )
            else:
                conn.execute(
                    """
                    UPDATE member_profiles
                    SET member_id = ?,
                        mall_member_id = ?,
                        member_status = ?,
                        joined_at = ?,
                        points_balance = ?,
                        gender = ?,
                        birth_date = ?,
                        phone = ?,
                        email = ?,
                        address = ?,
                        occupation = ?
                    WHERE profile_id = ?
                    """,
                    (*params, row["profile_id"]),
                )
            conn.commit()

