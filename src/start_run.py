"""Start (or resume) the wall-clock budget for a reproducible experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()
    path = args.root.resolve() / "artifacts" / "experiments" / "run_start.json"
    if not path.exists() or args.reset:
        save_json(path, {"started_utc": datetime.now(timezone.utc).isoformat(),
                         "budget_hours": 10,
                         "note": "Initial package/model download is excluded; computation starts here."})
    print(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
