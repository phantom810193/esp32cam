"""Utilities for composing personalised advertisement payloads."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping

from .advertising import AdContext, compose_final_image


def _normalise_copy(context: AdContext) -> dict[str, str]:
    """Extract the textual copy from an :class:`AdContext` instance.

    The helper keeps the module focussed on the composition responsibilities and
    ensures the returned payload has a predictable schema for both freshly
    generated and cached entries.
    """

    return {
        "headline": context.headline,
        "subheading": context.subheading,
        "highlight": context.highlight,
    }


def compose_member_ad(
    context: AdContext,
    base_image_path: str | Path,
    *,
    output_dir: str | Path,
    cached_payloads: MutableMapping[str, MutableMapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Compose the final advertisement artefacts for ``context``.

    The function produces a serialisable payload describing the assets required
    by the frontend.  When a ``cached_payloads`` mapping is provided the entry is
    updated in-place so repeated requests for the same member can reuse the same
    object without worrying about stale ``out_path`` values.
    """

    member_id = context.member_id
    base_path = Path(base_image_path)
    out_path = compose_final_image(base_path, member_id=member_id, output_dir=output_dir)

    payload: dict[str, Any] = {
        "member_id": member_id,
        "member_code": context.member_code,
        "scenario_key": context.scenario_key,
        "hero_path": str(base_path),
        "out_path": str(out_path),
    }
    payload.update(_normalise_copy(context))

    if cached_payloads is not None:
        cached = cached_payloads.get(member_id)
        if cached is None:
            cached_payloads[member_id] = payload
        else:
            cached.update(payload)
            payload = cached

    return payload


__all__ = ["compose_member_ad"]
