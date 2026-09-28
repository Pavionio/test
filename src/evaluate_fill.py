"""Tune a local/global quota on 400 queries, then inspect the 200 holdouts.

The four planned branches remain untouched. A quota of 50 is the monotonic
empty-slot fill; lower quotas reserve a few slots for cross-location ads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import recall_at_k, save_json
from .evaluate import load_parts, rank_group


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", choices=["A", "B"], required=True)
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    exp = root / "artifacts" / "experiments"
    local = pd.read_parquet(exp / f"{args.branch}_geo_exact_predictions.parquet")
    global_ = load_parts(root / "artifacts" / "cache" / "reranked" /
                         f"validation_geo_off_{args.branch}")
    items = pd.read_parquet(root / "data" / "processed" / "train_items.parquet",
                            columns=["item_id"])
    # The deployed model uses one blend coefficient for the complete answer.
    config = json.loads((exp / f"{args.branch}_geo_exact.json").read_text(encoding="utf-8"))
    alpha = float(config["alpha"])
    queries = pd.read_parquet(root / "data" / "processed" / "validation_queries.parquet")
    qmap = {r.eval_id: r for r in queries.itertuples()}
    gmap = {qid: items.item_id.iloc[
        rank_group(group, args.branch, alpha, 1000).item_index.astype(int)].tolist()
        for qid, group in global_.groupby("eval_id", sort=False)}
    def combine(local_ids: list[str], global_ids: list[str], quota: int) -> list[str]:
        answer = local_ids[:quota]
        present = set(answer)
        for item_id in global_ids:
            if len(answer) >= 50:
                break
            if item_id not in present:
                answer.append(item_id)
                present.add(item_id)
        # A rare global shortage should not waste available local candidates.
        for item_id in local_ids[quota:]:
            if len(answer) >= 50:
                break
            if item_id not in present:
                answer.append(item_id)
                present.add(item_id)
        return answer

    quotas = [25, 35, 40, 45, 48, 50]
    tune_grid = {}
    for quota in quotas:
        values = []
        for row in local.itertuples():
            q = qmap[row.eval_id]
            if q.split == "tune":
                ids = combine(row.answer.split(), gmap.get(row.eval_id, []), quota)
                values.append(recall_at_k(ids, q.relevant))
        tune_grid[str(quota)] = float(sum(values) / len(values))
    quota = max(quotas, key=lambda x: (tune_grid[str(x)], x))
    rows = []
    for row in local.itertuples():
        q = qmap[row.eval_id]
        local_ids = row.answer.split()
        answer = combine(local_ids, gmap.get(row.eval_id, []), quota)
        rows.append({"eval_id": row.eval_id, "split": q.split,
                     "word_count": q.word_count, "has_filter": q.has_filter,
                     "local_count": len(local_ids), "answer": " ".join(answer),
                     "recall_at_50": recall_at_k(answer, q.relevant)})
    result = pd.DataFrame(rows)
    result.to_parquet(exp / f"{args.branch}_geo_fill_predictions.parquet", index=False)
    report = {"branch": args.branch, "method": "geo_exact_quota_then_geo_off",
              "local_quota": quota, "tune_grid": tune_grid,
              "tune_recall_at_50": result.loc[result.split.eq("tune"), "recall_at_50"].mean(),
              "holdout_recall_at_50": result.loc[result.split.eq("holdout"), "recall_at_50"].mean(),
              "by_filter_holdout": result[result.split.eq("holdout")]
                  .groupby("has_filter").recall_at_50.mean().to_dict(),
              "by_words_holdout": result[result.split.eq("holdout")]
                  .groupby("word_count").recall_at_50.mean().to_dict(),
              "queries_with_open_slots": int(result.local_count.lt(50).sum()),
              "empty_local_queries": int(result.local_count.eq(0).sum())}
    save_json(exp / f"{args.branch}_geo_fill.json", report)
    print(report)


if __name__ == "__main__":
    main()
