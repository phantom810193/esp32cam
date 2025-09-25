"""SQLite helper utilities for the ESP32-CAM MVP backend."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .recognizer import FaceEncoding, FaceRecognizer

_LOGGER = logging.getLogger(__name__)


@dataclass
class Purchase:
    member_id: str
    item: str
    purchased_at: str
    unit_price: float
    quantity: float
    total_price: float


def _build_seed_purchases(
    start_timestamp: str, items: list[tuple[str, float, float]]
) -> list[dict[str, float | str]]:
    base = datetime.fromisoformat(start_timestamp)
    purchases: list[dict[str, float | str]] = []
    for index, (name, unit_price, quantity) in enumerate(items):
        scheduled = base + timedelta(days=index * 3 + (index % 4), hours=index % 5, minutes=(index * 11) % 60)
        purchases.append(
            {
                "item": name,
                "purchased_at": scheduled.strftime("%Y-%m-%d %H:%M"),
                "unit_price": float(unit_price),
                "quantity": float(quantity),
                "total_price": round(unit_price * quantity, 2),
            }
        )
    return purchases


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

            purchase_columns = self._get_table_columns(conn, "purchases")
            expected_columns = [
                "id",
                "member_id",
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
                    purchased_at TEXT NOT NULL,
                    item TEXT NOT NULL,
                    unit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    total_price REAL NOT NULL,
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
                "UPDATE purchases SET member_id = ? WHERE member_id = ?",
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
    def add_purchase(
        self,
        member_id: str,
        *,
        item: str,
        purchased_at: str,
        unit_price: float,
        quantity: float,
        total_price: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO purchases (
                    member_id,
                    purchased_at,
                    item,
                    unit_price,
                    quantity,
                    total_price
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    member_id,
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
                "SELECT member_id, purchased_at, item, unit_price, quantity, total_price"
                " FROM purchases WHERE member_id = ? ORDER BY purchased_at DESC, id DESC",
                (member_id,),
            ).fetchall()
        return [
            Purchase(
                member_id=row["member_id"],
                item=row["item"],
                purchased_at=row["purchased_at"],
                unit_price=float(row["unit_price"]),
                quantity=float(row["quantity"]),
                total_price=float(row["total_price"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    def ensure_demo_data(self) -> None:
        """Seed deterministic purchase histories for demo members."""

        dessert_specs: list[tuple[str, float, float]] = [
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
            ("椰香奶凍", 90.0, 3),
            ("楓糖肉桂捲", 85.0, 4),
            ("抹茶巴菲杯", 165.0, 2),
            ("焦糖堅果塔", 255.0, 1),
            ("蜂蜜舒芙蕾", 210.0, 1),
            ("綜合水果可麗餅", 190.0, 2),
            ("紫薯乳酪派", 270.0, 1),
            ("檸香優格凍", 105.0, 2),
            ("黑森林慕斯杯", 175.0, 2),
            ("藍莓乳酪可頌", 120.0, 3),
            ("萊姆羅勒塔", 245.0, 1),
            ("開心果達克瓦茲", 95.0, 4),
            ("橙酒巧克力捲", 320.0, 1),
            ("覆盆子雪白蛋糕", 340.0, 1),
            ("蜂蜜奶香巴巴露娃", 260.0, 1),
            ("伯爵茶冰淇淋三明治", 110.0, 3),
            ("波蘭酥皮蘋果派", 280.0, 1),
            ("馬斯卡彭提拉米蘇杯", 150.0, 2),
            ("西西里檸檬塔", 230.0, 1),
            ("焦糖布丁奶昔", 140.0, 2),
            ("乳酪香草泡芙", 95.0, 4),
            ("黑芝麻生乳捲", 295.0, 1),
        ]

        kids_specs: list[tuple[str, float, float]] = [
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
            ("幼兒感覺統合課程", 2380.0, 1),
            ("親子野餐活動餐盒", 280.0, 3),
            ("幼兒園游泳課程", 1980.0, 1),
            ("幼兒節慶表演服裝", 620.0, 1),
            ("幼兒園晨間活力課", 980.0, 1),
            ("親子手作香氛課", 1320.0, 1),
            ("幼兒園親師座談會餐點", 180.0, 6),
            ("幼兒足球體驗營", 1580.0, 1),
            ("幼兒園才藝發表DVD", 450.0, 1),
            ("親子探索農場門票", 660.0, 2),
            ("幼兒園戶外教學車資", 350.0, 3),
            ("幼兒園夏日水樂園日票", 880.0, 2),
            ("親子烘焙材料包", 540.0, 1),
            ("幼兒園畢典花束訂金", 320.0, 1),
            ("幼兒園閱讀角落捐書", 280.0, 2),
            ("親子晨跑活動補給", 150.0, 4),
            ("幼兒園防疫清潔組", 380.0, 1),
            ("幼兒園戲劇工作坊", 1680.0, 1),
            ("親子創意美術課", 1250.0, 1),
            ("幼兒園跨校交流活動券", 720.0, 1),
        ]

        dessert_history = _build_seed_purchases("2025-01-04 10:30", dessert_specs)
        kids_history = _build_seed_purchases("2025-01-05 09:20", kids_specs)

        self._seed_member_history("MEME0383FE3AA", dessert_history)
        self._seed_member_history("MEM692FFD0824", kids_history)

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

