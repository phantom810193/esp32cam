"""Central catalogue definitions for the ESP32-CAM retail demo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class Product:
    """Immutable catalogue entry."""

    code: str
    name: str
    category: str
    price: float
    view_rate: float  # expressed as 0-1 ratio


CATEGORY_PREFIXES: dict[str, str] = {
    "dessert": "DES",
    "fitness": "FIT",
    "kindergarten": "KID",
    "homemaker": "HOM",
    "general": "GEN",
}

CATEGORY_LABELS: dict[str, str] = {
    "dessert": "精緻甜點",
    "fitness": "運動健身",
    "kindergarten": "幼兒教養",
    "homemaker": "家居生活",
    "general": "生活選品",
}

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dessert": (
        "蛋糕",
        "布蕾",
        "甜",
        "塔",
        "慕斯",
        "布丁",
        "餅乾",
        "乳酪",
        "可麗",
        "麵包",
    ),
    "fitness": (
        "瑜伽",
        "健身",
        "體能",
        "訓練",
        "肌力",
        "跑步",
        "能量",
        "運動",
        "瑜珈",
        "伸展",
    ),
    "kindergarten": (
        "幼兒",
        "親子",
        "兒童",
        "才藝",
        "園",
        "課",
        "積木",
        "繪本",
        "童",
    ),
    "homemaker": (
        "家用",
        "清潔",
        "廚房",
        "香氛",
        "收納",
        "濾水",
        "補充包",
        "沐浴",
        "家庭",
        "家居",
        "烹飪",
    ),
    "general": (
        "咖啡",
        "茶",
        "禮盒",
        "票",
        "市集",
        "旅行",
        "香氛",
    ),
}

_RAW_CATALOGUE: dict[str, Sequence[tuple[str, float, float]]] = {
    "dessert": (
        ("草莓千層蛋糕禮盒", 620.0, 0.72),
        ("焦糖海鹽布蕾雙入", 210.0, 0.61),
        ("抹茶生乳捲分享組", 560.0, 0.67),
        ("藍莓優格慕斯杯", 260.0, 0.58),
        ("法式莓果塔", 280.0, 0.64),
        ("伯爵茶可麗露禮盒", 520.0, 0.69),
        ("蜂蜜檸檬磅蛋糕", 320.0, 0.56),
        ("玫瑰荔枝蛋糕", 360.0, 0.55),
        ("榛果巧克力布朗尼", 240.0, 0.6),
    ),
    "fitness": (
        ("高強度間歇訓練月票", 2680.0, 0.48),
        ("智能飛輪體驗課程", 1680.0, 0.5),
        ("肌力核心進階班", 1980.0, 0.46),
        ("運動機能壓縮衣", 1280.0, 0.54),
        ("能量蛋白飲禮盒", 680.0, 0.57),
        ("瑜伽伸展一日營", 1480.0, 0.45),
        ("體態評估與飲食諮詢", 3200.0, 0.43),
        ("樂齡低衝擊體適能課", 1580.0, 0.47),
        ("戶外越野跑訓練營", 2380.0, 0.44),
    ),
    "kindergarten": (
        ("幼兒律動課體驗券", 720.0, 0.62),
        ("親子烘焙下午茶套票", 1280.0, 0.6),
        ("幼兒園夏令營報名", 5200.0, 0.58),
        ("幼兒科學探索盒", 680.0, 0.59),
        ("親子劇場週末票", 980.0, 0.55),
        ("幼兒園延托服務時數", 450.0, 0.57),
        ("幼兒才藝試上課程", 780.0, 0.6),
        ("幼兒園校車月票", 2800.0, 0.53),
        ("親子閱讀共學包", 920.0, 0.61),
    ),
    "homemaker": (
        ("家用濾水壺含濾芯組", 950.0, 0.52),
        ("智慧家電延長線", 420.0, 0.49),
        ("廚房收納保鮮組", 560.0, 0.51),
        ("家庭常備洗衣精補充包", 360.0, 0.55),
        ("香氛擴香瓶禮盒", 620.0, 0.5),
        ("不沾煎鍋三件組", 1880.0, 0.46),
        ("居家舒眠香氛蠟燭", 420.0, 0.48),
        ("智能掃拖機器人保養", 2800.0, 0.44),
        ("家庭露營炊具套組", 1980.0, 0.45),
    ),
    "general": (
        ("精品手沖咖啡體驗課", 980.0, 0.41),
        ("城市慢旅文化導覽", 1580.0, 0.38),
        ("當季市集嚴選蔬果箱", 880.0, 0.47),
        ("無酒精氣泡飲禮盒", 620.0, 0.4),
        ("手作香氛擴香課程", 1280.0, 0.39),
        ("節慶限定禮物卡", 500.0, 0.42),
        ("都會輕旅背包", 1380.0, 0.43),
        ("生活選品訂閱方案", 760.0, 0.37),
        ("永續生活工作坊", 1180.0, 0.36),
    ),
}


def _build_catalogue() -> list[Product]:
    catalogue: list[Product] = []
    for category, items in _RAW_CATALOGUE.items():
        prefix = CATEGORY_PREFIXES[category]
        for index, (name, price, view_rate) in enumerate(items, start=1):
            code = f"{prefix}{index:03d}"
            catalogue.append(
                Product(
                    code=code,
                    name=name,
                    category=category,
                    price=float(price),
                    view_rate=float(view_rate),
                )
            )
    return catalogue


_CATALOGUE: tuple[Product, ...] = tuple(_build_catalogue())


def get_catalogue() -> tuple[Product, ...]:
    """Return the immutable catalogue."""

    return _CATALOGUE


def iter_by_category(category: str) -> Iterator[Product]:
    normalized = category.strip().lower()
    return (product for product in _CATALOGUE if product.category == normalized)


def get_product_by_code(code: str) -> Product | None:
    normalized = code.strip().upper()
    for product in _CATALOGUE:
        if product.code == normalized:
            return product
    return None


def infer_category_from_item(name: str) -> str:
    """Best-effort category mapping derived from the product name."""

    item = name.strip()
    lowered = item.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in item or keyword in lowered for keyword in keywords):
            return category
    return "general"


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, "生活選品")


def purchased_product_codes(purchases: Iterable[str]) -> set[str]:
    known = {product.name: product.code for product in _CATALOGUE}
    codes = set()
    for item in purchases:
        code = known.get(item)
        if code:
            codes.add(code)
    return codes
