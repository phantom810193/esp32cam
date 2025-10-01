# backend/ai_gemini.py
# -*- coding: utf-8 -*-
"""
Gemini-based ad copy generation.
Requires env: GEMINI_API_KEY
"""
import os
from typing import Tuple

import google.generativeai as genai

GEMINI_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")

def _init():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)

_model = None
def _get_model():
    global _model
    if _model is None:
        _model = _init()
    return _model

def ad_copy_unregistered(product_name: str, category: str) -> str:
    """
    20字內，促加入會員（試用包/試吃包）
    """
    prompt = f"""以繁體中文寫一句20字內的廣告文案，鼓勵加入會員可獲得「{product_name}」試用包/試吃包。語氣活潑、有吸引力。避免標點過多。"""
    model = _get_model()
    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    return text[:40]  # 防呆

def ad_copy_registered(product_name: str, category: str) -> Tuple[str, str]:
    """
    1-2行，含會員專屬優惠折扣（自動產出折扣字句）
    回傳：(headline, subline)
    """
    prompt = f"""以繁體中文寫1至2行廣告文案，主打商品「{product_name}」，並給出會員專屬優惠折扣字句（例如85折或折\$50）。避免過度浮誇。"""
    model = _get_model()
    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) == 1:
        return lines[0], ""
    return lines[0], lines[1]
