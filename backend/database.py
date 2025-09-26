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
    "MEME0383FE3AA": "",
    "MEM692FFD0824": "",
    "MEMFITNESS2025": "",
    "MEMWESTERN2025": "",
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

    def get_member_code(self, member_id: str) -> str:
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
            if code is not None:
                return str(code)
        if member_id.startswith("MEM") and len(member_id) > 3:
            suffix = member_id[3:]
            if suffix:
                return f"ME{suffix}"
        return member_id

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
            ("楓糖肉桂捲", 85.0, 4),
            ("抹茶巴菲杯", 165.0, 2),
            ("焦糖堅果塔", 255.0, 1),
            ("蜂蜜舒芙蕾", 210.0, 1),
            ("綜合水果可麗餅", 190.0, 2),
            ("紫薯乳酪派", 270.0, 1),
            ("檸香優格凍", 105.0, 2),
            ("黑森林慕斯杯", 175.0, 2),
            ("藍莓乳酪可頌", 120.0, 3),
            ("橙酒巧克力捲", 320.0, 1),
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
        ]

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

        western_specs: list[tuple[str, float, float]] = [
            ("經典義大利麵禮盒", 520.0, 1),
            ("冷壓初榨橄欖油組", 980.0, 1),
            ("帕瑪森起司塊", 420.0, 1),
            ("手工拖鞋麵包", 180.0, 2),
            ("西班牙海鮮燉飯套組", 1280.0, 1),
            ("普羅旺斯香草罐", 260.0, 1),
            ("紅酒醋雙入組", 360.0, 1),
            ("法式奶油可頌", 220.0, 2),
            ("鴨胸排組合", 960.0, 1),
            ("爐烤牛排禮盒", 1580.0, 1),
            ("義式濃縮咖啡豆", 680.0, 1),
            ("奶油蘑菇濃湯包", 320.0, 2),
            ("香煎鱸魚片", 780.0, 1),
            ("地中海沙拉橄欖", 260.0, 2),
            ("義式香腸薄餅", 480.0, 1),
            ("進口番茄罐頭組", 350.0, 2),
            ("迷迭香烤雞套餐", 880.0, 1),
            ("松露鹽禮盒", 620.0, 1),
            ("義式乳酪拼盤", 780.0, 1),
            ("羅馬風提拉米蘇", 420.0, 1),
            ("精品紅酒", 1380.0, 1),
            ("義式冰淇淋組", 560.0, 1),
            ("焦糖布丁燉蛋", 260.0, 2),
            ("香烤波隆那香腸", 450.0, 1),
            ("手工香料麵包棒", 220.0, 2),
            ("義式起司火腿拼盤", 980.0, 1),
            ("法式奶油燉菜", 420.0, 1),
            ("義大利手工巧克力", 360.0, 1),
            ("有機芝麻葉", 180.0, 2),
            ("進口蘆筍束", 320.0, 1),
            ("松露義大利麵醬", 560.0, 1),
            ("帕尼尼三明治組", 320.0, 2),
            ("義式甜菜沙拉", 260.0, 2),
            ("義式燉牛膝", 1260.0, 1),
            ("地中海無花果果醬", 260.0, 1),
            ("精品氣泡水組", 420.0, 1),
            ("義式濃縮咖啡機清潔片", 280.0, 1),
            ("全麥佛卡夏麵包", 210.0, 2),
            ("家庭用烤盤紙", 150.0, 2),
            ("高級餐巾紙組", 180.0, 2),
            ("進口橄欖油噴霧", 450.0, 1),
            ("法式鑄鐵平底鍋", 2280.0, 1),
            ("海鹽黑巧克力", 260.0, 2),
            ("香檳氣泡酒", 1580.0, 1),
            ("進口奶油乳酪", 320.0, 1),
            ("義式蕃茄冷湯", 260.0, 2),
            ("精緻餐桌布", 520.0, 1),
            ("香氛蠟燭組", 680.0, 1),
            ("橄欖木砧板", 780.0, 1),
            ("義式奶油酥餅", 260.0, 2),
            ("精品濾掛咖啡組", 420.0, 1),
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
        western_history = _build_seed_purchases(
            "2025-01-07 12:15",
            self.get_member_code("MEMWESTERN2025"),
            western_specs,
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
        self._seed_member_history("MEMWESTERN2025", western_history)
        self._seed_member_history("MEMHOMECARE2025", homemaker_history)
        self._seed_member_history("MEMHEALTH2025", health_history)

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

