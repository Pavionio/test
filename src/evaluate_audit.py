"""One post-selection audit on requests not used to tune either branch.

The first 200 records of the retrieval-only split are fixed before scoring.
Their labels were used in the aggregate Recall@1000 diagnostic, but no
reranker coefficient, geography choice, or local quota was fitted on them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import recall_at_k, save_json
from .evaluate import load_parts, rank_group
from .fusion import geo_bias_top50


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", choices=["A", "B"], required=True)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    exp = root / "artifacts" / "experiments"
    queries = pd.read_parquet(root / "data" / "processed" / "validation_queries.parquet")
    queries = queries[queries.split.eq("retrieval")].iloc[:args.count]
    items = pd.read_parquet(root / "data" / "processed" / "train_items.parquet",
                            columns=["item_id"])
    fill = json.loads((exp / f"{args.branch}_geo_fill.json").read_text(encoding="utf-8"))
    quota = int(fill["local_quota"])
    selection = json.loads((exp / "selection.json").read_text(encoding="utf-8"))["chosen"]
    use_bias = args.branch == "B" and bool(selection.get("geo_bias", False))
    bias = float(json.loads((exp / "B_geo_bias.json").read_text(encoding="utf-8"))["bias"]) \
        if use_bias else None
    groups = {}
    raw_groups = {}
    for geo in ("geo_off", "geo_exact"):
        frame = load_parts(root / "artifacts" / "cache" / "reranked" /
                           f"validation_{geo}_{args.branch}" / f"audit_{args.count}")
        raw_groups[geo] = {qid: g for qid, g in frame.groupby("eval_id", sort=False)}
        config = json.loads((exp / f"{args.branch}_{geo}.json").read_text(encoding="utf-8"))
        alpha = float(config["alpha"])
        groups[geo] = {qid: items.item_id.iloc[
            rank_group(group, args.branch, alpha, 1000).item_index.astype(int)].tolist()
            for qid, group in frame.groupby("eval_id", sort=False)}
    # Deployment uses the exact-local blend coefficient for both groups. For B
    # this is always zero. Re-rank A global scores here if it is ever selected.
    if args.branch == "A":
        raise NotImplementedError("Audit of branch A needs its exact-local alpha on global scores")

    rows = []
    for q in queries.itertuples():
        local = groups["geo_exact"].get(q.eval_id, [])
        global_ = groups["geo_off"].get(q.eval_id, [])
        if use_bias:
            gg = raw_groups["geo_off"].get(q.eval_id)
            lg = raw_groups["geo_exact"].get(q.eval_id)
            answer = items.item_id.iloc[geo_bias_top50(gg, lg, bias)].tolist()
        else:
            answer = local[:quota]
            seen = set(answer)
            for item_id in global_ + local[quota:]:
                if len(answer) >= 50:
                    break
                if item_id not in seen:
                    answer.append(item_id)
                    seen.add(item_id)
        rows.append({"eval_id": q.eval_id,
                     "geo_off_recall": recall_at_k(global_, q.relevant),
                     "geo_exact_recall": recall_at_k(local, q.relevant),
                     "selected_recall": recall_at_k(answer, q.relevant),
                     "local_count": len(local), "answer": " ".join(answer)})
    result = pd.DataFrame(rows)
    result.to_parquet(exp / f"audit_{args.branch}_{args.count}_predictions.parquet", index=False)
    values = result.selected_recall.to_numpy()
    rng = np.random.default_rng(20260928)
    bootstrap = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    report = {"branch": args.branch, "count": len(result),
              "method": "geo_bias" if use_bias else "local_quota",
              "local_quota": quota if not use_bias else None, "geo_bias": bias,
              "geo_off_recall_at_50": float(result.geo_off_recall.mean()),
              "geo_exact_recall_at_50": float(result.geo_exact_recall.mean()),
              "selected_recall_at_50": float(values.mean()),
              "selected_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
              "empty_local_queries": int(result.local_count.eq(0).sum())}
    save_json(exp / f"audit_{args.branch}_{args.count}.json", report)
    print(report)


if __name__ == "__main__":
    main()
