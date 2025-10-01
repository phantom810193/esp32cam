"""Static product catalogue used for purchase predictions."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

CATEGORY_LABELS: dict[str, str] = {
    "breakfast": "穀物 / 早餐",
    "wellness": "健康飲品",
    "fitness": "運動配件",
    "beauty": "美妝保養",
    "family": "親子生活",
    "home": "居家用品",
    "general": "生活選品",
}


@dataclass(frozen=True)
class Product:
    code: str
    name: str
    category: str
    price: float
    view_rate: float


_CATALOGUE: tuple[Product, ...] = (
    Product("SKU100234", "桂花牌有機澳洲燕麥片 500g", "breakfast", 250.0, 0.85),
    Product("SKU204233", "Light 能量蛋白棒｜醇濃巧克力", "wellness", 260.0, 0.8),
    Product("SKU401022", "冷壓綜合果汁 350ml", "wellness", 95.0, 0.75),
    Product("SKU305120", "瑜伽墊止滑清潔噴霧", "fitness", 240.0, 0.72),
    Product("SKU401023", "無糖優格 200ml", "wellness", 250.0, 0.74),
    Product("SKU823144", "冷壓蘋果果汁 350ml", "wellness", 95.0, 0.73),
    Product("SKU621412", "橡膠啞鈴 14kg", "fitness", 395.0, 0.68),
    Product("SKU510001", "高纖優格脆片 400g", "breakfast", 320.0, 0.69),
    Product("SKU612300", "薑黃能量飲 12入", "wellness", 520.0, 0.65),
    Product("SKU720450", "深層舒緩瑜伽滾筒", "fitness", 880.0, 0.6),
    Product("SKU830210", "莓果活力綜合果昔包", "wellness", 180.0, 0.66),
    Product("SKU910120", "植萃保濕面膜 5 入", "beauty", 350.0, 0.58),
    Product("SKU930450", "香氛舒眠枕頭噴霧", "home", 420.0, 0.54),
    Product("SKU950310", "親子手作烘焙組", "family", 560.0, 0.62),
)

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "breakfast": ("燕麥", "脆片", "穀", "早餐"),
    "wellness": ("果汁", "飲", "優格", "能量", "茶"),
    "fitness": ("瑜伽", "啞鈴", "運動", "健身", "滾筒"),
    "beauty": ("面膜", "保濕", "精華", "美妝"),
    "family": ("親子", "幼兒", "家庭", "手作"),
    "home": ("香氛", "居家", "枕頭", "噴霧"),
}


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, CATEGORY_LABELS["general"])


def get_catalogue() -> list[Product]:
    return list(_CATALOGUE)


def infer_category_from_item(item: str) -> str:
    normalized = item.lower()
    for product in _CATALOGUE:
        name = product.name.lower()
        if name in normalized or normalized in name:
            return product.category
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    return "general"


@lru_cache(maxsize=1)
def _normalized_lookup() -> dict[str, Product]:
    lookup: dict[str, Product] = {}
    for product in _CATALOGUE:
        lookup[product.name.lower()] = product
    return lookup


def purchased_product_codes(items: Iterable[str]) -> set[str]:
    lookup = _normalized_lookup()
    codes: set[str] = set()
    for item in items:
        normalized = item.lower()
        for name, product in lookup.items():
            if name in normalized or normalized in name:
                codes.add(product.code)
                break
    return codes
