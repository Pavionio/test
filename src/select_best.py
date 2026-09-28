"""Select the strongest measured branch that fits the ten-hour compute cap."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dateutil.parser import isoparse

from .common import save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--budget-hours", type=float, default=10.0)
    args = p.parse_args()
    root = args.root.resolve()
    exp = root / "artifacts" / "experiments"
    start_path = exp / "run_start.json"
    if not start_path.exists():
        raise FileNotFoundError("Record run_start.json before starting the experiment")
    start = datetime.fromisoformat(json.loads(start_path.read_text(encoding="utf-8"))["started_utc"])
    now = datetime.now(timezone.utc)
    wall_elapsed = (now - start).total_seconds()
    pause_file = exp / "compute_pauses.json"
    pauses = json.loads(pause_file.read_text(encoding="utf-8-sig")) if pause_file.exists() else []
    if isinstance(pauses, dict):
        pauses = [pauses]
    excluded_idle = 0.0
    for pause in pauses:
        left = max(start, isoparse(pause["pause_start_utc"]))
        right = min(now, isoparse(pause["pause_end_utc"]))
        excluded_idle += max(0.0, (right - left).total_seconds())
    elapsed = wall_elapsed - excluded_idle
    # Reserve 30 minutes for benchmark retrieval, submission verification and
    # small operational interruptions. The measured reranker speed is scaled
    # from 600 validation queries to all 2452 benchmark queries.
    available = args.budget_hours * 3600 - elapsed - 1800
    pilot = json.loads((exp / "pilot_validation_geo_off_B.json").read_text(encoding="utf-8"))
    # The pilot scores uncached pairs with the same cross encoder used by both
    # branches. Later validation passes can be mostly cached, so their
    # fresh-pairs/elapsed rate is not a reliable final-run estimate.
    pilot_rate = pilot["pairs_per_second"]
    alternatives = []
    for branch in ("A", "B"):
        for geo in ("geo_off", "geo_exact"):
            report = json.loads((exp / f"{branch}_{geo}.json").read_text(encoding="utf-8"))
            geo_timing = json.loads((root / "artifacts" / "cache" / "reranked" /
                                     f"validation_{geo}_{branch}" / "timing.json").read_text(encoding="utf-8"))
            target_pairs = 2452 * geo_timing["pairs"] / 600
            estimate = target_pairs / pilot_rate * 1.2
            alternatives.append({"branch": branch, "geo": geo,
                                 "fill_geo_off": False,
                                 "geo_bias": False,
                                 "holdout_recall_at_50": report["holdout_recall_at_50"],
                                 "estimated_final_seconds": estimate,
                                 "fits_budget": estimate <= available})
        # This supplemental variant preserves the exact-local top 50 and
        # spends otherwise empty slots on global results. Its time estimate is
        # deliberately conservative: it assumes both complete reranking runs.
        fill_path = exp / f"{branch}_geo_fill.json"
        if fill_path.exists():
            fill = json.loads(fill_path.read_text(encoding="utf-8"))
            local = next(x for x in alternatives if x["branch"] == branch and x["geo"] == "geo_exact")
            global_ = next(x for x in alternatives if x["branch"] == branch and x["geo"] == "geo_off")
            estimate = local["estimated_final_seconds"] + global_["estimated_final_seconds"]
            alternatives.append({"branch": branch, "geo": "geo_exact", "fill_geo_off": True,
                                 "geo_bias": False,
                                 "holdout_recall_at_50": fill["holdout_recall_at_50"],
                                 "estimated_final_seconds": estimate,
                                 "fits_budget": estimate <= available})
        if branch == "B" and (exp / "B_geo_bias.json").exists():
            bias_report = json.loads((exp / "B_geo_bias.json").read_text(encoding="utf-8"))
            local = next(x for x in alternatives if x["branch"] == "B" and
                         x["geo"] == "geo_exact" and not x["fill_geo_off"])
            global_ = next(x for x in alternatives if x["branch"] == "B" and
                           x["geo"] == "geo_off")
            estimate = local["estimated_final_seconds"] + global_["estimated_final_seconds"]
            alternatives.append({"branch": "B", "geo": "geo_exact",
                                 "fill_geo_off": False, "geo_bias": True,
                                 "holdout_recall_at_50": bias_report["holdout_recall_at_50"],
                                 "estimated_final_seconds": estimate,
                                 "fits_budget": estimate <= available})
    feasible = [x for x in alternatives if x["fits_budget"]]
    if not feasible:
        raise RuntimeError(f"No branch fits remaining budget ({available:.0f} seconds)")
    best_recall = max(x["holdout_recall_at_50"] for x in feasible)
    # Differences below 0.005 are treated as a tie; choose the faster run.
    near_best = [x for x in feasible if best_recall - x["holdout_recall_at_50"] <= 0.005]
    chosen = min(near_best, key=lambda x: x["estimated_final_seconds"])
    save_json(exp / "selection.json", {"chosen": chosen, "alternatives": alternatives,
                                        "elapsed_seconds": elapsed,
                                        "wall_elapsed_seconds": wall_elapsed,
                                        "excluded_idle_seconds": excluded_idle,
                                        "remaining_for_final_seconds": available,
                                        "budget_hours": args.budget_hours})
    print(chosen)


if __name__ == "__main__":
    main()
