"""
Clean, single-pass implementation of `generate_ad_with_vertex`.
Removes recursive wrappers and guarantees required keys.
"""
from __future__ import annotations
from typing import Any, Dict

# Re-export the current service (alias for legacy names)
from .ai import GeminiService as _GeminiService, GeminiUnavailableError  # type: ignore

# Legacy class aliases
GeminiService = _GeminiService
AdVertex = _GeminiService
AdGen = _GeminiService
AdGenerator = _GeminiService
VertexAdGen = _GeminiService

def _first_nonempty(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v not in (None, ""):
            return v
    return None
def generate_ad_with_vertex(*args, **kwargs) -> Dict[str, Any]:
    """
    Returns dict with at least:
      headline, subheading, highlights, item, category, score,
      image, image_name, cta, discount, status
    """
    member = kwargs.get("member") or {}
    member_id = member.get("id") or kwargs.get("member_id") or "UNKNOWN"
    registered = bool(kwargs.get("registered") or member.get("registered"))
    pred = kwargs.get("prediction") or {}
    # 1) 嘗試用服務產生文案（失敗就用空 dict）
    ad: Dict[str, Any] = {}
    try:
        svc = _GeminiService()
        if hasattr(svc, "generate_ad"):
            ad = svc.generate_ad(*args, **kwargs) or {}
        elif hasattr(svc, "generate_copy"):
            ad = dict(svc.generate_copy(*args, **kwargs) or {})
    except Exception:
        ad = {}
    if not isinstance(ad, dict):
        ad = {}
    # 2) 規範化必備鍵（避免下游 KeyError）
    ad.setdefault("headline", "本月熱銷・限時加碼")
    ad.setdefault("subheading", "加入會員即享專屬優惠")
    ad.setdefault("highlights", ["熱門品項推薦", "會員 88 折", "不限金額免運"])
    item = _first_nonempty(
        ad.get("item"),
        pred.get("item"),
        pred.get("name"),
        pred.get("product_name"),
        pred.get("sku_name"),
        pred.get("sku"),
    )
    ad["item"] = item or "精選商品"
    ad.setdefault("category", _first_nonempty(pred.get("category"), ad.get("category"), "general"))
    try:
        ad["score"] = float(_first_nonempty(ad.get("score"), pred.get("score"), 0) or 0)
    except Exception:
        ad["score"] = 0.0
    if ad.get("image") in (None, ""):
        ad["image"] = None if registered else "ME0000.jpg"

    default_image_name = f"ad_{member_id}.jpg" if registered else "ME0000.jpg"
    if ad.get("image_name") in (None, ""):
        ad["image_name"] = default_image_name

    ad.setdefault("cta", "立即加入會員")
    ad.setdefault("discount", "會員 88 折" if registered else "新客 95 折")
    ad.setdefault("status", "ok")
    return ad
