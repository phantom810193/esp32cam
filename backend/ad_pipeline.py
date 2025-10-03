"""High level helper to compose personalised advertisements with caching."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

from .advertising import (
    AdCopy,
    compose_final_image,
    generate_ad_copy_via_gemini,
    get_member_scenario,
    member_to_base_image,
)
from .prediction import predict_for_member, recent_summary


_CACHE_FILENAME = "index.json"
_CACHE_TTL = timedelta(minutes=10)


def _out_dir() -> Path:
    configured = os.environ.get("AD_OUT_DIR")
    if configured:
        path = Path(configured)
    else:
        base_dir = Path(os.environ.get("AD_BASE_DIR", "/srv/esp32-ads"))
        path = base_dir / "out"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path(out_dir: Path) -> Path:
    return out_dir / _CACHE_FILENAME


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data  # type: ignore[return-value]


def _save_cache(path: Path, payload: Dict[str, Dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _is_fresh(entry: Dict[str, Any]) -> bool:
    timestamp = entry.get("generated_at")
    if not timestamp:
        return False
    try:
        generated = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return False
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return now - generated <= _CACHE_TTL


def _ensure_member_filename(path_str: str, member_id: str) -> str:
    path = Path(path_str)
    expected_prefix = f"{member_id}_"
    if path.name.startswith(expected_prefix):
        return str(path)
    new_path = path.with_name(expected_prefix + path.name)
    try:
        path.rename(new_path)
    except OSError:
        return str(path)
    return str(new_path)


def compose_member_ad(
    member_id: str,
    *,
    force: bool = False,
    limit: int = 3,
    svc=None,
) -> Dict[str, Any]:
    """Generate or return cached advertisement payload for the given member."""

    out_dir = _out_dir()
    cache_path = _index_path(out_dir)
    cache = _load_cache(cache_path)
    cache_entry = cache.get(member_id)
    if cache_entry and not force and _is_fresh(cache_entry):
        cached_payload = cache_entry.get("data")
        if isinstance(cached_payload, dict):
            return cached_payload

    scenario_key, scenario_label = get_member_scenario(member_id)
    base_image = member_to_base_image(member_id)

    top_predictions = predict_for_member(member_id, limit=limit)
    summary = recent_summary(member_id)

    gemini_start = perf_counter()
    copy: AdCopy = generate_ad_copy_via_gemini(
        member_id,
        scenario_label,
        top_predictions,
        summary,
        svc=svc,
    )
    gemini_duration = int((perf_counter() - gemini_start) * 1000)

    compose_start = perf_counter()
    output_path = compose_final_image(base_image, copy, out_dir=str(out_dir))
    output_path = _ensure_member_filename(output_path, member_id)
    compose_duration = int((perf_counter() - compose_start) * 1000)

    payload: Dict[str, Any] = {
        "member_id": member_id,
        "scenario": scenario_label,
        "scenario_key": scenario_key,
        "out_path": output_path,
        "copy": asdict(copy),
        "top_predictions": top_predictions,
        "timings": {"gemini_ms": gemini_duration, "compose_ms": compose_duration},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    cache[member_id] = {"generated_at": payload["generated_at"], "data": payload}
    try:
        _save_cache(cache_path, cache)
    except OSError:
        pass

    return payload
