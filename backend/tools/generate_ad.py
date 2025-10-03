"""CLI tool to generate and cache personalised advertisements."""
from __future__ import annotations

import argparse
import json
from typing import Any

from ..ad_pipeline import compose_member_ad


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate advertisement for a member")
    parser.add_argument("--member", required=True, help="Member ID (e.g. ME0001)")
    parser.add_argument("--limit", type=int, default=3, help="Number of top predictions to include")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration instead of using cached result",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload: dict[str, Any] = compose_member_ad(
        args.member,
        limit=args.limit,
        force=bool(args.force),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(payload.get("out_path", ""))


if __name__ == "__main__":
    main()
