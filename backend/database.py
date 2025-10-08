"""SQLite helper utilities for the ESP32-CAM MVP backend."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, TypedDict
from zoneinfo import ZoneInfo

from .recognizer import FaceEncoding, FaceRecognizer

_LOGGER = logging.getLogger(__name__)


MEMBER_CODE_OVERRIDES: dict[str, str] = {}

NEW_GUEST_MEMBER_ID = "MEM8CCB842A77"

SEED_MEMBER_IDS: tuple[str, ...] = (
    "MEME0383FE3AA",
    "MEM692FFD0824",
    "MEMFITNESS2025",
    "MEMHOMECARE2025",
    "MEMHEALTH2025",
    NEW_GUEST_MEMBER_ID,
)

SEED_PROFILE_LABELS: tuple[str, ...] = (
    "dessert-lover",
    "family-groceries",
    "fitness-enthusiast",
    "home-manager",
    "wellness-gourmet",
    "brand-new-guest",
)


PROFILE_LABEL_TO_SEED_MEMBER: dict[str, str] = {
    "dessert-lover": "MEME0383FE3AA",
    "family-groceries": "MEM692FFD0824",
    "fitness-enthusiast": "MEMFITNESS2025",
    "home-manager": "MEMHOMECARE2025",
    "wellness-gourmet": "MEMHEALTH2025",
}



_TAIPEI_TZ = ZoneInfo("Asia/Taipei")


class _PersonaPurchaseItem(TypedDict):
    item: str
    category: str
    price: tuple[int, int]


class _PersonaPurchaseConfig(TypedDict):
    prefix: str
    items: list[_PersonaPurchaseItem]


_SEPTEMBER_2025_PURCHASE_CONFIG: dict[str, _PersonaPurchaseConfig] = {
    "MEME0383FE3AA": {
        "prefix": "DES",
        "items": [
            {"item": "7-11 早餐", "category": "咖啡廳", "price": (65, 150)},
            {"item": "手作戚風蛋糕", "category": "甜點", "price": (180, 320)},
            {"item": "草莓生乳捲", "category": "甜點", "price": (260, 420)},
            {"item": "抹茶千層蛋糕", "category": "甜點", "price": (320, 520)},
            {"item": "蜜桃水果茶", "category": "飲料", "price": (90, 160)},
            {"item": "黑糖珍珠鮮奶", "category": "飲料", "price": (85, 160)},
            {"item": "下午茶點心盒", "category": "甜點", "price": (220, 420)},
            {"item": "焦糖布丁禮盒", "category": "甜點", "price": (260, 480)},
            {"item": "手沖單品咖啡豆", "category": "咖啡廳", "price": (350, 580)},
            {"item": "巷口咖啡拿鐵", "category": "咖啡廳", "price": (110, 190)},
            {"item": "可可奶霜鬆餅", "category": "甜點", "price": (210, 360)},
            {"item": "奶蓋紅茶", "category": "飲料", "price": (90, 150)},
            {"item": "手工餅乾禮盒", "category": "甜點", "price": (320, 560)},
            {"item": "冷萃咖啡瓶", "category": "咖啡廳", "price": (180, 260)},
            {"item": "生巧克力塔", "category": "甜點", "price": (240, 420)},
        ],
    },
    "MEM692FFD0824": {
        "prefix": "FAM",
        "items": [
            {"item": "全聯購買洗衣精", "category": "家庭開銷", "price": (180, 320)},
            {"item": "家樂福廚房紙巾", "category": "居家用品", "price": (120, 260)},
            {"item": "寶寶尿布箱購", "category": "親子用品", "price": (520, 980)},
            {"item": "兒童營養餅乾", "category": "親子用品", "price": (180, 320)},
            {"item": "保鮮盒組", "category": "日用品", "price": (280, 520)},
            {"item": "家庭洗衣精補充包", "category": "家庭開銷", "price": (180, 360)},
            {"item": "超市蔬果採購", "category": "家庭開銷", "price": (350, 850)},
            {"item": "兒童沐浴乳", "category": "親子用品", "price": (220, 420)},
            {"item": "早餐麥片組合", "category": "日用品", "price": (150, 280)},
            {"item": "週末家常菜食材", "category": "家庭開銷", "price": (420, 980)},
            {"item": "親子DIY手作包", "category": "親子用品", "price": (260, 460)},
            {"item": "家庭醫藥箱補充", "category": "日用品", "price": (320, 620)},
            {"item": "超商儲值電費", "category": "家庭開銷", "price": (600, 1200)},
            {"item": "兒童故事書組", "category": "親子用品", "price": (280, 520)},
            {"item": "家庭清潔用品大採購", "category": "日用品", "price": (360, 820)},
        ],
    },
    "MEMFITNESS2025": {
        "prefix": "FIT",
        "items": [
            {"item": "健身房月卡", "category": "運動服務", "price": (1200, 1800)},
            {"item": "高蛋白雞胸便當", "category": "健身餐", "price": (150, 250)},
            {"item": "運動用品店阻力帶", "category": "運動用品", "price": (320, 580)},
            {"item": "乳清蛋白補充罐", "category": "蛋白粉", "price": (950, 1500)},
            {"item": "運動飲料箱購", "category": "健身餐", "price": (350, 620)},
            {"item": "高蛋白燕麥棒", "category": "健身餐", "price": (220, 360)},
            {"item": "健身手套", "category": "運動用品", "price": (380, 680)},
            {"item": "BCAA 胺基酸", "category": "蛋白粉", "price": (780, 1300)},
            {"item": "私人教練課程", "category": "運動服務", "price": (1500, 2000)},
            {"item": "壓力褲", "category": "運動用品", "price": (680, 980)},
            {"item": "藍莓優格冰沙", "category": "健身餐", "price": (120, 220)},
            {"item": "高蛋白粉補充包", "category": "蛋白粉", "price": (880, 1400)},
            {"item": "運動毛巾組", "category": "運動用品", "price": (240, 420)},
            {"item": "能量膠補給", "category": "健身餐", "price": (90, 160)},
            {"item": "體組成檢測", "category": "運動服務", "price": (600, 900)},
        ],
    },
    "MEMHOMECARE2025": {
        "prefix": "HOM",
        "items": [
            {"item": "家樂福廚房紙巾", "category": "廚房用品", "price": (120, 260)},
            {"item": "IKEA 收納盒組", "category": "居家用品", "price": (320, 620)},
            {"item": "掃地機器人濾網", "category": "家庭電器", "price": (450, 780)},
            {"item": "香氛擴香補充瓶", "category": "居家用品", "price": (280, 520)},
            {"item": "廚房不沾鍋", "category": "廚房用品", "price": (780, 1500)},
            {"item": "智慧燈泡二入", "category": "家庭電器", "price": (420, 680)},
            {"item": "衣物柔軟精大罐", "category": "居家用品", "price": (220, 360)},
            {"item": "玻璃保鮮盒組", "category": "廚房用品", "price": (360, 620)},
            {"item": "電熱水壺", "category": "家庭電器", "price": (680, 980)},
            {"item": "烘碗機保養服務", "category": "家庭電器", "price": (950, 1600)},
            {"item": "洗衣機清潔劑", "category": "居家用品", "price": (260, 420)},
            {"item": "防蟲密封罐", "category": "廚房用品", "price": (180, 320)},
            {"item": "家用工具箱", "category": "居家用品", "price": (520, 880)},
            {"item": "除濕機濾網", "category": "家庭電器", "price": (480, 780)},
            {"item": "餐具收納架", "category": "廚房用品", "price": (240, 420)},
        ],
    },
    "MEMHEALTH2025": {
        "prefix": "HLT",
        "items": [
            {"item": "冷壓果汁", "category": "健康食品", "price": (180, 280)},
            {"item": "有機蔬菜箱", "category": "有機蔬果", "price": (520, 880)},
            {"item": "保健綜合維他命", "category": "保健品", "price": (680, 1200)},
            {"item": "無糖優格禮盒", "category": "健康食品", "price": (260, 420)},
            {"item": "植物性膳食纖維", "category": "保健品", "price": (450, 760)},
            {"item": "有機藍莓", "category": "有機蔬果", "price": (320, 520)},
            {"item": "發芽糙米", "category": "健康食品", "price": (240, 360)},
            {"item": "益生菌粉", "category": "保健品", "price": (680, 980)},
            {"item": "養生堅果禮盒", "category": "健康食品", "price": (420, 780)},
            {"item": "冷壓橄欖油", "category": "健康食品", "price": (520, 980)},
            {"item": "有機紅蘿蔔汁", "category": "健康食品", "price": (180, 300)},
            {"item": "養身漢方湯包", "category": "保健品", "price": (380, 650)},
            {"item": "無麩質能量棒", "category": "健康食品", "price": (180, 260)},
            {"item": "舒眠草本茶", "category": "健康食品", "price": (200, 320)},
            {"item": "保健酵素飲", "category": "保健品", "price": (520, 850)},
        ],
    },
}


def _generate_monthly_records(
    *,
    member_code: str,
    config: _PersonaPurchaseConfig,
    seed_key: str,
    code_suffix: str,
    year: int,
    month: int,
    day_upper: int,
    count: int,
) -> list[dict[str, float | str]]:
    rng = random.Random(seed_key)
    minute_choices = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)

    records: list[dict[str, float | str]] = []
    for index in range(count):
        choice = rng.choice(config["items"])
        price_range = choice["price"]
        low, high = int(price_range[0]), int(price_range[1])
        if high <= low:
            unit_price = float(low)
        else:
            step = 5 if high - low >= 5 else 1
            unit_price = float(rng.randrange(low, high + 1, step))

        purchase_time = datetime(year, month, 1) + timedelta(
            days=rng.randint(0, max(0, day_upper - 1)),
            hours=rng.randint(8, 21),
            minutes=rng.choice(minute_choices),
        )

        records.append(
            {
                "member_code": member_code,
                "product_category": choice["category"],
                "internal_item_code": f"{config['prefix']}-{code_suffix}{index + 1:03d}",
                "item": choice["item"],
                "purchased_at": purchase_time.strftime("%Y-%m-%d %H:%M"),
                "unit_price": unit_price,
                "quantity": 1.0,
                "total_price": float(round(unit_price, 2)),
            }
        )

    records.sort(key=lambda entry: entry["purchased_at"])
    return records


@dataclass
class Purchase:
    member_id: str
    member_code: str
    product_category: str
    internal_item_code: str
    item: str
    purchased_at: str
    unit_price: float
    quantity: float
    total_price: float


@dataclass
class MemberProfile:
    profile_id: int
    profile_label: str
    name: str | None
    member_id: str | None
    mall_member_id: str | None
    member_status: str | None
    joined_at: str | None
    points_balance: float | None
    gender: str | None
    birth_date: str | None
    phone: str | None
    email: str | None
    address: str | None
    occupation: str | None
    first_image_filename: str | None


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

AUTO_MERGE_DISTANCE_MIN = 0.32
AUTO_MERGE_DISTANCE_MAX = 0.40


@dataclass
class MemberMatch:
    matched_id: str | None
    matched_distance: float | None
    best_candidate_id: str | None
    best_candidate_distance: float | None


@dataclass
class ResolvedMember:
    member_id: str
    new_member: bool
    distance: float | None
    encoding_updated: bool = False
    auto_merged_source: str | None = None



def _build_seed_purchases(
    start_timestamp: str,
    items: list[tuple[str, float, float] | tuple[str, float, float, str] | tuple[str, float, float, str, str]],
    member_code: str | None = None,
    *,
    default_category: str = "一般商品",
    code_prefix: str = "SKU",
) -> list[dict[str, float | str]]:
    base = datetime.fromisoformat(start_timestamp)
    purchases: list[dict[str, float | str]] = []
    for index, spec in enumerate(items, start=1):
        name: str
        unit_price: float
        quantity: float
        category: str
        internal_code: str

        if len(spec) == 3:
            name, unit_price, quantity = spec  # type: ignore[misc]
            category = default_category
            internal_code = f"{code_prefix}-{index:03d}"
        elif len(spec) == 4:
            name, unit_price, quantity, category = spec  # type: ignore[misc]
            internal_code = f"{code_prefix}-{index:03d}"
        else:
            name, unit_price, quantity, category, internal_code = spec  # type: ignore[misc]
            if not internal_code:
                internal_code = f"{code_prefix}-{index:03d}"

        scheduled = base + timedelta(days=index * 3 + (index % 4), hours=index % 5, minutes=(index * 11) % 60)
        purchases.append(
            {
                "member_code": member_code,
                "item": name,
                "purchased_at": scheduled.strftime("%Y-%m-%d %H:%M"),
                "unit_price": float(unit_price),
                "quantity": float(quantity),
                "total_price": round(unit_price * quantity, 2),
                "product_category": category,
                "internal_item_code": internal_code,
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
        self._profile_purchase_templates: dict[
            str, list[dict[str, float | str]]
        ] = {}
        self._profile_history_seeded: set[str] = set()
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
                "name",
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
                "first_image_filename",
            ]

            needs_drop = False
            needs_rebuild = False

            if profile_columns:
                if profile_columns != expected_profile_columns:
                    needs_drop = True
                else:
                    profile_meta = conn.execute("PRAGMA table_info(member_profiles)").fetchall()
                    nullable_columns = {
                        "member_status",
                        "joined_at",
                        "points_balance",
                        "gender",
                        "phone",
                        "email",
                        "name",
                    }
                    if any(row["name"] in nullable_columns and row["notnull"] for row in profile_meta):
                        needs_drop = True
                    elif self._profile_label_has_unique_constraint(conn):
                        needs_rebuild = True

            if needs_drop:
                conn.execute("DROP TABLE IF EXISTS member_profiles")
            elif needs_rebuild:
                self._rebuild_member_profiles_without_unique(conn, expected_profile_columns)

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS member_profiles (
                    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_label TEXT NOT NULL,
                    name TEXT,
                    member_id TEXT UNIQUE,
                    mall_member_id TEXT,
                    member_status TEXT DEFAULT '有效',
                    joined_at TEXT,
                    points_balance REAL DEFAULT 0,
                    gender TEXT,
                    birth_date TEXT,
                    phone TEXT,
                    email TEXT,
                    address TEXT,
                    occupation TEXT,
                    first_image_filename TEXT,
                    FOREIGN KEY(member_id) REFERENCES members(member_id)
                );
                """
            )

            purchase_columns = self._get_table_columns(conn, "purchases")
            expected_columns = [
                "id",
                "member_id",
                "member_code",
                "product_category",
                "internal_item_code",
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
                    product_category TEXT NOT NULL DEFAULT '',
                    internal_item_code TEXT NOT NULL DEFAULT '',
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
    @staticmethod
    def _profile_label_has_unique_constraint(conn: sqlite3.Connection) -> bool:
        indexes = conn.execute("PRAGMA index_list(member_profiles)").fetchall()
        for index in indexes:
            if not index["unique"]:
                continue
            index_name = str(index["name"])
            columns = conn.execute(f"PRAGMA index_info({index_name})").fetchall()
            if len(columns) != 1:
                continue
            column_name = columns[0]["name"]
            if column_name == "profile_label":
                return True
        return False

    def _rebuild_member_profiles_without_unique(
        self, conn: sqlite3.Connection, expected_columns: Iterable[str]
    ) -> None:
        conn.execute("ALTER TABLE member_profiles RENAME TO member_profiles_legacy")
        conn.execute(
            """
            CREATE TABLE member_profiles (
                profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_label TEXT NOT NULL,
                name TEXT,
                member_id TEXT UNIQUE,
                mall_member_id TEXT,
                member_status TEXT DEFAULT '有效',
                joined_at TEXT,
                points_balance REAL DEFAULT 0,
                gender TEXT,
                birth_date TEXT,
                phone TEXT,
                email TEXT,
                address TEXT,
                occupation TEXT,
                first_image_filename TEXT,
                FOREIGN KEY(member_id) REFERENCES members(member_id)
            );
            """
        )
        columns_csv = ", ".join(expected_columns)
        conn.execute(
            f"INSERT INTO member_profiles ({columns_csv}) "
            f"SELECT {columns_csv} FROM member_profiles_legacy"
        )
        conn.execute("DROP TABLE member_profiles_legacy")
        conn.execute(
            """
            UPDATE sqlite_sequence
            SET seq = (SELECT MAX(profile_id) FROM member_profiles)
            WHERE name = 'member_profiles'
            """
        )

    def _normalize_placeholder_profiles(
        self,
        conn: sqlite3.Connection | None = None,
        *,
        ensure_placeholders: bool = True,
    ) -> None:
        if conn is None:
            with self._connect() as managed_conn:
                self._normalize_placeholder_profiles(
                    managed_conn, ensure_placeholders=ensure_placeholders
                )
            return

        conn.execute(
            """
            UPDATE member_profiles
            SET profile_label = 'unknown'
            WHERE profile_label LIKE 'staging-slot-%'
            """
        )

        if SEED_MEMBER_IDS:
            placeholders = ",".join("?" for _ in SEED_MEMBER_IDS)
            conn.execute(
                f"""
                UPDATE member_profiles
                SET member_id = NULL,
                    first_image_filename = NULL
                WHERE profile_label = 'unknown' AND member_id IN ({placeholders})
                """,
                SEED_MEMBER_IDS,
            )

        unassigned_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM member_profiles
            WHERE profile_label = 'unknown' AND member_id IS NULL
            """
        ).fetchone()[0]
        if ensure_placeholders:
            target_placeholders = 3
            missing = target_placeholders - int(unassigned_count)
            for _ in range(max(0, missing)):
                conn.execute(
                    """
                    INSERT INTO member_profiles (profile_label, name, member_status)
                    VALUES ('unknown', '未註冊客戶', '有效')
                    """
                )

    def find_member_by_encoding(
        self, encoding: FaceEncoding, recognizer: FaceRecognizer
    ) -> MemberMatch:
        """Return the closest matching member and fallback candidate for ``encoding``."""

        matched_id: str | None = None
        matched_distance: float | None = None
        best_candidate_id: str | None = None
        best_candidate_distance: float | None = None

        with self._connect() as conn:
            for row in conn.execute("SELECT member_id, encoding_json FROM members"):
                member_id = row["member_id"]
                stored = FaceEncoding.from_jsonable(json.loads(row["encoding_json"]))

                if encoding.signature and encoding.signature == member_id:
                    matched_id = member_id
                    matched_distance = 0.0
                    best_candidate_id = member_id
                    best_candidate_distance = 0.0
                    break

                if stored.signature and stored.signature == encoding.signature:
                    matched_id = member_id
                    matched_distance = 0.0
                    best_candidate_id = member_id
                    best_candidate_distance = 0.0
                    break

                distance = recognizer.distance(stored, encoding)

                if recognizer.is_match(stored, encoding):
                    if matched_distance is None or distance < matched_distance:
                        matched_id = member_id
                        matched_distance = distance

                if best_candidate_distance is None or distance < best_candidate_distance:
                    best_candidate_id = member_id
                    best_candidate_distance = distance

        return MemberMatch(
            matched_id=matched_id,
            matched_distance=matched_distance,
            best_candidate_id=best_candidate_id,
            best_candidate_distance=best_candidate_distance,
        )

    @staticmethod
    def _should_replace_encoding(current: FaceEncoding, candidate: FaceEncoding) -> bool:
        if not current.signature and candidate.signature:
            return True
        if (
            candidate.signature
            and candidate.source.startswith("rekognition")
            and not current.source.startswith("rekognition")
        ):
            return True
        return False

    def _maybe_merge_seed_member(
        self,
        provisional_member_id: str,
        *,
        candidate_id: str | None,
        encoding: FaceEncoding,
    ) -> tuple[str | None, bool]:
        canonical_id: str | None = None
        if candidate_id and candidate_id in SEED_MEMBER_IDS:
            canonical_id = candidate_id
        if canonical_id is None:
            profile = self.get_member_profile(provisional_member_id)
            if profile and profile.profile_label in SEED_PROFILE_TO_MEMBER_ID:
                canonical_id = SEED_PROFILE_TO_MEMBER_ID[profile.profile_label]
        if not canonical_id or canonical_id == provisional_member_id:
            return None, False
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM members WHERE member_id = ?", (canonical_id,)
            ).fetchone()
        if not exists:
            return None, False
        try:
            self.merge_members(provisional_member_id, canonical_id)
        except ValueError:
            return None, False
        encoding_updated = self.maybe_refresh_member_encoding(canonical_id, encoding)
        _LOGGER.info("Auto-merged provisional member %s into canonical seed %s", provisional_member_id, canonical_id)
        return canonical_id, encoding_updated


    def maybe_refresh_member_encoding(self, member_id: str, encoding: FaceEncoding) -> bool:
        existing = self.get_member_encoding(member_id)
        if existing is None:
            return False
        if self._should_replace_encoding(existing, encoding):
            self.update_member_encoding(member_id, encoding)
            return True
        return False

    def resolve_member_id(
        self,
        encoding: FaceEncoding,
        recognizer: FaceRecognizer,
        *,
        auto_merge_min_distance: float = AUTO_MERGE_DISTANCE_MIN,
        auto_merge_max_distance: float = AUTO_MERGE_DISTANCE_MAX,
    ) -> ResolvedMember:
        """Resolve ``encoding`` to a member id, auto-merging when safe."""

        match = self.find_member_by_encoding(encoding, recognizer)

        if match.matched_id:
            encoding_updated = self.maybe_refresh_member_encoding(match.matched_id, encoding)
            return ResolvedMember(
                member_id=match.matched_id,
                new_member=False,
                distance=match.matched_distance,
                encoding_updated=encoding_updated,
            )

        candidate_id = match.best_candidate_id
        candidate_distance = match.best_candidate_distance
        lower_bound = max(recognizer.tolerance, auto_merge_min_distance)
        upper_bound = auto_merge_max_distance

        if (
            candidate_id
            and candidate_distance is not None
            and lower_bound <= candidate_distance <= upper_bound
        ):
            encoding_updated = self.maybe_refresh_member_encoding(candidate_id, encoding)
            _LOGGER.info("Auto-resolved face to member %s (distance=%.3f)", candidate_id, candidate_distance)
            return ResolvedMember(
                member_id=candidate_id,
                new_member=False,
                distance=candidate_distance,
                encoding_updated=encoding_updated,
            )

        member_id = self.create_member(encoding, recognizer.derive_member_id(encoding))
        canonical_id, canonical_updated = self._maybe_merge_seed_member(
            member_id, candidate_id=candidate_id, encoding=encoding
        )
        if canonical_id:
            return ResolvedMember(
                member_id=canonical_id,
                new_member=False,
                distance=candidate_distance,
                encoding_updated=canonical_updated,
                auto_merged_source=member_id,
            )
        _LOGGER.info("Created new member %s after failing to auto-resolve", member_id)
        return ResolvedMember(
            member_id=member_id,
            new_member=True,
            distance=candidate_distance,
        )

    def create_member(self, encoding: FaceEncoding, member_id: str | None = None) -> str:
        candidate = member_id or self._generate_member_id(encoding)
        payload = json.dumps(encoding.to_jsonable(), ensure_ascii=False)

        with self._connect() as conn:
            while True:
                try:
                    conn.execute(
                        "INSERT INTO members (member_id, encoding_json) VALUES (?, ?)",
                        (candidate, payload),
                    )
                except sqlite3.IntegrityError:
                    _LOGGER.warning(
                        "Member id %s already exists, generating fallback id", candidate
                    )
                    candidate = self._generate_member_id()
                    continue
                else:
                    conn.commit()
                    break

        _LOGGER.info("Created new member %s", candidate)
        self._claim_unassigned_profile(candidate)
        return candidate

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
            source_profile_row = conn.execute(
                "SELECT first_image_filename FROM member_profiles WHERE member_id = ?",
                (source_id,),
            ).fetchone()
            target_profile_row = conn.execute(
                "SELECT first_image_filename FROM member_profiles WHERE member_id = ?",
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
            if (
                source_profile_row
                and source_profile_row["first_image_filename"]
                and (not target_profile_row or not target_profile_row["first_image_filename"])
            ):
                conn.execute(
                    "UPDATE member_profiles SET first_image_filename = ? WHERE member_id = ?",
                    (source_profile_row["first_image_filename"], target_id),
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
            next_index = conn.execute("SELECT COUNT(*) FROM members").fetchone()[0] + 1
            while True:
                candidate = f"MEM{next_index:03d}"
                exists = conn.execute(
                    "SELECT 1 FROM members WHERE member_id = ?", (candidate,)
                ).fetchone()
                if not exists:
                    return candidate
                next_index += 1

    # ------------------------------------------------------------------
    def _claim_unassigned_profile(self, member_id: str) -> None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT profile_id, profile_label
                FROM member_profiles
                WHERE member_id IS NULL
                ORDER BY profile_id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return

            profile_id = int(row["profile_id"])
            profile_label = str(row["profile_label"])
            conn.execute(
                "UPDATE member_profiles SET member_id = ? WHERE profile_id = ?",
                (member_id, profile_id),
            )
            conn.commit()
        _LOGGER.info(
            "Assigned new member %s to pre-seeded profile %s (%s)",
            member_id,
            profile_id,
            profile_label,
        )
        self._populate_profile_history(profile_label, member_id)

    def _populate_profile_history(self, profile_label: str, member_id: str) -> None:
        template = self._profile_purchase_templates.get(profile_label)
        if not template:
            return
        if profile_label in self._profile_history_seeded:
            return

        for purchase in template:
            self.add_purchase(member_id, **purchase)

        self._profile_history_seeded.add(profile_label)
        _LOGGER.info(
            "Seeded %d purchases for %s using %s persona",
            len(template),
            member_id,
            profile_label,
        )

    def _reset_seed_profiles(self) -> None:
        if not SEED_PROFILE_LABELS:
            return

        labels_placeholder = ",".join("?" for _ in SEED_PROFILE_LABELS)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT member_id
                FROM member_profiles
                WHERE profile_label IN ({labels_placeholder})
                """,
                SEED_PROFILE_LABELS,
            ).fetchall()

            purge_ids = {str(row["member_id"]) for row in rows if row["member_id"]}
            purge_ids.update(SEED_MEMBER_IDS)

            if purge_ids:
                purge_list = sorted(purge_ids)
                placeholders = ",".join("?" for _ in purge_list)
                conn.execute(
                    f"DELETE FROM purchases WHERE member_id IN ({placeholders})",
                    purge_list,
                )
                conn.execute(
                    f"DELETE FROM members WHERE member_id IN ({placeholders})",
                    purge_list,
                )
                conn.execute(
                    f"DELETE FROM upload_events WHERE member_id IN ({placeholders})",
                    purge_list,
                )

            conn.execute(
                f"""
                UPDATE member_profiles
                SET member_id = NULL,
                    first_image_filename = NULL
                WHERE profile_label IN ({labels_placeholder})
                """,
                SEED_PROFILE_LABELS,
            )
            conn.commit()

    def get_member_profile(self, member_id: str) -> MemberProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT profile_id,
                       profile_label,
                       name,
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
                       occupation,
                       first_image_filename
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
            name=str(row["name"]) if row["name"] else None,
            member_id=str(row["member_id"]) if row["member_id"] else None,
            mall_member_id=str(row["mall_member_id"]) if row["mall_member_id"] else None,
            member_status=str(row["member_status"]) if row["member_status"] else None,
            joined_at=str(row["joined_at"]) if row["joined_at"] else None,
            points_balance=float(row["points_balance"]) if row["points_balance"] is not None else None,
            gender=str(row["gender"]) if row["gender"] else None,
            birth_date=str(row["birth_date"]) if row["birth_date"] else None,
            phone=str(row["phone"]) if row["phone"] else None,
            email=str(row["email"]) if row["email"] else None,

            address=str(row["address"]) if row["address"] else None,
            occupation=str(row["occupation"]) if row["occupation"] else None,
            first_image_filename=
                str(row["first_image_filename"]) if row["first_image_filename"] else None,
        )

    def list_member_profiles(self) -> list[MemberProfile]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT profile_id,
                       profile_label,
                       name,
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
                       occupation,
                       first_image_filename
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
                    name=str(row["name"]) if row["name"] else None,
                    member_id=str(row["member_id"]) if row["member_id"] else None,
                    mall_member_id=str(row["mall_member_id"]) if row["mall_member_id"] else None,
                    member_status=str(row["member_status"]) if row["member_status"] else None,
                    joined_at=str(row["joined_at"]) if row["joined_at"] else None,
                    points_balance=float(row["points_balance"]) if row["points_balance"] is not None else None,
                    gender=str(row["gender"]) if row["gender"] else None,
                    birth_date=str(row["birth_date"]) if row["birth_date"] else None,
                    phone=str(row["phone"]) if row["phone"] else None,
                    email=str(row["email"]) if row["email"] else None,

                    address=str(row["address"]) if row["address"] else None,
                    occupation=str(row["occupation"]) if row["occupation"] else None,
                    first_image_filename=
                        str(row["first_image_filename"]) if row["first_image_filename"] else None,
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
        product_category: str | None = None,
        internal_item_code: str | None = None,
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
        resolved_category = product_category or ""
        resolved_internal_code = internal_item_code or ""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO purchases (
                    member_id,
                    member_code,
                    product_category,
                    internal_item_code,
                    purchased_at,
                    item,
                    unit_price,
                    quantity,
                    total_price
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    member_id,
                    resolved_code,
                    resolved_category,
                    resolved_internal_code,
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
                "SELECT member_id, member_code, product_category, internal_item_code, purchased_at, item, unit_price, quantity, total_price"
                " FROM purchases WHERE member_id = ? ORDER BY purchased_at DESC, id DESC",
                (member_id,),
            ).fetchall()
        return self._rows_to_purchases(rows)

    def get_purchase_history_page(
        self, member_id: str, limit: int, offset: int
    ) -> list[Purchase]:
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        if limit == 0:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT member_id, member_code, product_category, internal_item_code, purchased_at, item, unit_price, quantity, total_price"
                " FROM purchases WHERE member_id = ? ORDER BY purchased_at DESC, id DESC LIMIT ? OFFSET ?",
                (member_id, limit, offset),
            ).fetchall()
        return self._rows_to_purchases(rows)

    def count_purchase_history(self, member_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM purchases WHERE member_id = ?",
                (member_id,),
            ).fetchone()
        if row is None:
            return 0
        return int(row["total"]) if "total" in row.keys() else int(row[0])

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
            self._set_first_image_if_absent(conn, member_id, image_filename)
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
            reserved_rows = conn.execute(
                """
                SELECT first_image_filename
                FROM member_profiles
                WHERE first_image_filename IS NOT NULL
                """
            ).fetchall()
            protected_filenames = {
                str(row["first_image_filename"])
                for row in reserved_rows
                if row["first_image_filename"]
            }
            filenames = [
                row["image_filename"]
                for row in rows
                if row["image_filename"] and row["image_filename"] not in protected_filenames
            ]

            delete_placeholders = ",".join("?" for _ in ids_to_delete)
            conn.execute(
                f"DELETE FROM upload_events WHERE id IN ({delete_placeholders})",
                tuple(ids_to_delete),
            )
            conn.commit()

        return filenames

    def _set_first_image_if_absent(
        self, conn: sqlite3.Connection, member_id: str, image_filename: str | None
    ) -> None:
        if not image_filename:
            return
        row = conn.execute(
            """
            SELECT first_image_filename
            FROM member_profiles
            WHERE member_id = ?
            """,
            (member_id,),
        ).fetchone()
        if row is None:
            return
        if row["first_image_filename"]:
            return
        conn.execute(
            "UPDATE member_profiles SET first_image_filename = ? WHERE member_id = ?",
            (image_filename, member_id),
        )

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
        """Load deterministic demo data from ``data/mvp.sql`` when empty."""

        self._normalize_placeholder_profiles(ensure_placeholders=False)
        self._profile_purchase_templates = {}
        self._profile_history_seeded.clear()

        with self._connect() as conn:
            try:
                (count,) = conn.execute("SELECT COUNT(*) FROM member_profiles").fetchone()
            except sqlite3.OperationalError:
                count = 0
            if count:
                self._ensure_recent_history()
                self._ensure_new_guest_member()
                return

        sql_path = Path(__file__).resolve().parent / "data" / "mvp.sql"
        if not sql_path.exists():  # pragma: no cover - defensive guard
            _LOGGER.warning("Seed file %s not found; skipping demo data load", sql_path)
            return

        self._reset_seed_profiles()

        self._seed_member_profile(
            profile_label="dessert-lover",
            name="李函霏",
            member_id=None,
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
            name="林位青",
            member_id=None,
            mall_member_id="ME0002",
            member_status="有效",
            joined_at="2020-09-01",
            points_balance=980,
            gender="男",
            birth_date="1990-02-08",
            phone="0923-556-789",
            email="familybuyer@example.com",
            address="新北市板橋區文化路100號",
            occupation="幼兒園老師",
        )
        self._seed_member_profile(
            profile_label="fitness-enthusiast",
            name="范文華",
            member_id=None,
            mall_member_id="ME0003",
            member_status="有效",
            joined_at="2019-11-20",
            points_balance=2040,
            gender="女",
            birth_date="1985-04-19",
            phone="0955-112-233",
            email="fitgoer@example.com",
            address="台中市西屯區市政北二路88號",
            occupation="企業健身顧問",
        )
        self._seed_member_profile(
            profile_label="home-manager",
            name="未註冊客戶",
            member_id=None,
            mall_member_id="",
            member_status=None,
            joined_at=None,
            points_balance=None,
            gender=None,
            birth_date=None,
            phone=None,
            email=None,
            address=None,
            occupation=None,
        )
        self._seed_member_profile(
            profile_label="wellness-gourmet",
            name="未註冊客戶",
            member_id=None,
            mall_member_id="",
            member_status=None,
            joined_at=None,
            points_balance=None,
            gender=None,
            birth_date=None,
            phone=None,
            email=None,
            address=None,
            occupation=None,
        )

        self._seed_member_profile(
            profile_label="brand-new-guest",
            name="新客專屬體驗",
            member_id=None,
            mall_member_id="",
            member_status="未入會",
            joined_at=None,
            points_balance=0,
            gender=None,
            birth_date=None,
            phone=None,
            email=None,
            address=None,
            occupation=None,
        )

        insert_statements: list[str] = []
        with sql_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("--"):
                    continue
                if not line.lower().startswith("insert into"):
                    continue
                if not line.endswith(";"):
                    line = f"{line};"
                insert_statements.append(line)

        script = "\n".join(insert_statements)

        script = script.replace("INSERT INTO ", "INSERT OR IGNORE INTO ")

        with self._connect() as conn:
            conn.executescript(script)
            face_map = {
                "MEME0383FE3AA": "faces/dessert_lover.jpg",
                "MEM692FFD0824": "faces/family_groceries.jpg",
                "MEMFITNESS2025": "faces/fitness_enthusiast.jpg",
                "MEMHOMECARE2025": "faces/home_manager.jpg",
                "MEMHEALTH2025": "faces/wellness_gourmet.jpg",
                NEW_GUEST_MEMBER_ID: "faces/新顧客.png",
            }
            for member_id, filename in face_map.items():
                conn.execute(
                    "UPDATE member_profiles SET first_image_filename = ? WHERE member_id = ?",
                    (filename, member_id),
                )
            conn.commit()

        for profile_label, member_id in PROFILE_LABEL_TO_SEED_MEMBER.items():
            template = self._profile_purchase_templates.get(profile_label)
            if not template:
                continue
            self._seed_member_history(member_id, template)

        self._ensure_new_guest_member()

        self._normalize_placeholder_profiles()

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

        config = _SEPTEMBER_2025_PURCHASE_CONFIG.get(member_id)
        if not config:
            return

        member_code = self.get_member_code(member_id)
        september_records = _generate_monthly_records(
            member_code=member_code,
            config=config,
            seed_key=f"{member_id}-2025-09",
            code_suffix="S25",
            year=2025,
            month=9,
            day_upper=30,
            count=50,
        )
        for record in september_records:
            self.add_purchase(member_id, **record)

        self._ensure_month_history(
            member_id,
            config=config,
            year=2025,
            month=10,
            day_upper=10,
            count=10,
            code_suffix="O25",
        )

    def _ensure_month_history(
        self,
        member_id: str,
        *,
        config: _PersonaPurchaseConfig,
        year: int,
        month: int,
        day_upper: int,
        count: int,
        code_suffix: str,
    ) -> None:
        if count <= 0:
            return

        month_key = f"{year}-{month:02d}"
        like_pattern = f"{month_key}%"
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM purchases WHERE member_id = ? AND purchased_at LIKE ?",
                (member_id, like_pattern),
            ).fetchone()
            existing_total = int(row["total"]) if row is not None and "total" in row.keys() else int(row[0]) if row else 0

            existing_codes = {
                str(row["internal_item_code"])
                for row in conn.execute(
                    "SELECT internal_item_code FROM purchases WHERE member_id = ? AND purchased_at LIKE ?",
                    (member_id, like_pattern),
                ).fetchall()
            }

        missing = count - existing_total
        if missing <= 0:
            return

        member_code = self.get_member_code(member_id) or member_id
        seed_key = f"{member_id}-{month_key}"
        candidate_records = _generate_monthly_records(
            member_code=member_code,
            config=config,
            seed_key=seed_key,
            code_suffix=code_suffix,
            year=year,
            month=month,
            day_upper=day_upper,
            count=count,
        )

        for record in candidate_records:
            internal_code = str(record.get("internal_item_code", ""))
            if internal_code in existing_codes:
                continue
            self.add_purchase(member_id, **record)
            existing_codes.add(internal_code)
            missing -= 1
            if missing <= 0:
                break

    def _ensure_recent_history(self) -> None:
        for member_id, config in _SEPTEMBER_2025_PURCHASE_CONFIG.items():
            if member_id == NEW_GUEST_MEMBER_ID:
                continue
            self._ensure_month_history(
                member_id,
                config=config,
                year=2025,
                month=10,
                day_upper=10,
                count=10,
                code_suffix="O25",
            )

    def _ensure_new_guest_member(self) -> None:
        import numpy as np

        encoding = FaceEncoding(
            np.zeros(128, dtype=np.float32),
            signature=NEW_GUEST_MEMBER_ID,
            source="seed",
        )
        payload = json.dumps(encoding.to_jsonable(), ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO members (member_id, encoding_json) VALUES (?, ?)",
                (NEW_GUEST_MEMBER_ID, payload),
            )
            row = conn.execute(
                "SELECT profile_id, member_id, first_image_filename FROM member_profiles WHERE profile_label = ?",
                ("brand-new-guest",),
            ).fetchone()

            if row is None:
                conn.execute(
                    """
                    INSERT INTO member_profiles (
                        profile_label,
                        name,
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
                        occupation,
                        first_image_filename
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "brand-new-guest",
                        "新客專屬體驗",
                        NEW_GUEST_MEMBER_ID,
                        "",
                        "未入會",
                        None,
                        0.0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        "faces/新顧客.png",
                    ),
                )
            else:
                profile_id = int(row["profile_id"])
                if not row["member_id"]:
                    conn.execute(
                        "UPDATE member_profiles SET member_id = ? WHERE profile_id = ?",
                        (NEW_GUEST_MEMBER_ID, profile_id),
                    )
                conn.execute(
                    """
                    UPDATE member_profiles
                    SET name = COALESCE(NULLIF(name, ''), '新客專屬體驗'),
                        member_status = COALESCE(member_status, '未入會'),
                        points_balance = COALESCE(points_balance, 0),
                        first_image_filename = COALESCE(first_image_filename, 'faces/新顧客.png')
                    WHERE profile_id = ?
                    """,
                    (profile_id,),
                )
            conn.commit()

    def get_member_profile_by_label(self, profile_label: str) -> MemberProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT profile_id,
                       profile_label,
                       name,
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
                       occupation,
                       first_image_filename
                FROM member_profiles
                WHERE profile_label = ?
                LIMIT 1
                """,
                (profile_label,),
            ).fetchone()

        if row is None:
            return None

        return MemberProfile(
            profile_id=int(row["profile_id"]),
            profile_label=str(row["profile_label"]),
            name=str(row["name"]) if row["name"] else None,
            member_id=str(row["member_id"]) if row["member_id"] else None,
            mall_member_id=str(row["mall_member_id"]) if row["mall_member_id"] else None,
            member_status=str(row["member_status"]) if row["member_status"] else None,
            joined_at=str(row["joined_at"]) if row["joined_at"] else None,
            points_balance=float(row["points_balance"]) if row["points_balance"] is not None else None,
            gender=str(row["gender"]) if row["gender"] else None,
            birth_date=str(row["birth_date"]) if row["birth_date"] else None,
            phone=str(row["phone"]) if row["phone"] else None,
            email=str(row["email"]) if row["email"] else None,
            address=str(row["address"]) if row["address"] else None,
            occupation=str(row["occupation"]) if row["occupation"] else None,
            first_image_filename=
                str(row["first_image_filename"]) if row["first_image_filename"] else None,
        )

    def _seed_member_profile(
        self,
        *,
        profile_label: str,
        name: str | None,
        member_id: str | None,
        mall_member_id: str | None,
        member_status: str | None,
        joined_at: str | None,
        points_balance: float | None,
        gender: str | None,
        birth_date: str | None,
        phone: str | None,
        email: str | None,

        address: str | None,
        occupation: str | None,
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_id FROM member_profiles WHERE profile_label = ?",
                (profile_label,),
            ).fetchone()

            params = (
                name,
                member_id,
                mall_member_id,
                member_status,
                joined_at,
                float(points_balance) if points_balance is not None else None,
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
                        name,
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
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (profile_label, *params),
                )
            else:
                conn.execute(
                    """
                    UPDATE member_profiles
                    SET name = ?,
                        member_id = ?,
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

    @staticmethod
    def _rows_to_purchases(rows: Iterable[sqlite3.Row]) -> list[Purchase]:
        return [
            Purchase(
                member_id=row["member_id"],
                member_code=row["member_code"],
                product_category=str(row["product_category"] or ""),
                internal_item_code=str(row["internal_item_code"] or ""),
                item=row["item"],
                purchased_at=row["purchased_at"],
                unit_price=float(row["unit_price"]),
                quantity=float(row["quantity"]),
                total_price=float(row["total_price"]),
            )
            for row in rows
        ]


