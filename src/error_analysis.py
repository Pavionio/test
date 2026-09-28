"""Classify held-out misses as retrieval or final-selection losses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import recall_at_k, save_json
from .evaluate import load_parts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", choices=["A", "B"], default="B")
    p.add_argument("--method", choices=["fill", "bias"], default="fill")
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    exp = root / "artifacts" / "experiments"
    queries = pd.read_parquet(root / "data" / "processed" / "validation_queries.parquet")
    queries = queries[queries.split.eq("holdout")]
    qids = set(queries.eval_id)
    items = pd.read_parquet(root / "data" / "processed" / "train_items.parquet",
                            columns=["item_id"])
    reachable = {qid: set() for qid in qids}
    for geo in ("geo_off", "geo_exact"):
        folder = root / "artifacts" / "cache" / "candidates" / f"validation_{geo}"
        for path in sorted(folder.glob("part_*.parquet")):
            frame = pd.read_parquet(path, columns=["eval_id", "item_index"])
            frame = frame[frame.eval_id.isin(qids)]
            for qid, group in frame.groupby("eval_id", sort=False):
                reachable[qid].update(items.item_id.iloc[group.item_index.astype(int)])
    pred_file = (f"{args.branch}_geo_fill_predictions.parquet" if args.method == "fill"
                 else "B_geo_bias_predictions.parquet")
    if args.method == "bias" and args.branch != "B":
        p.error("bias fusion is implemented for branch B")
    predictions = pd.read_parquet(exp / pred_file)
    pred_map = {r.eval_id: r.answer.split() for r in predictions.itertuples()}
    rows = []
    for q in queries.itertuples():
        labels = set(q.relevant)
        found = labels & reachable[q.eval_id]
        selected = labels & set(pred_map[q.eval_id])
        rows.append({"eval_id": q.eval_id, "search_query": q.search_query,
                     "has_filter": q.has_filter, "word_count": q.word_count,
                     "relevant_count": len(labels), "reachable_count": len(found),
                     "selected_count": len(selected),
                     "retrieval_recall": len(found) / len(labels),
                     "final_recall": recall_at_k(pred_map[q.eval_id], q.relevant),
                     "retrieval_misses": len(labels - reachable[q.eval_id]),
                     "selection_misses": len(found - selected)})
    result = pd.DataFrame(rows)
    stem = f"error_analysis_{args.branch}_{args.method}_holdout"
    result.to_csv(exp / f"{stem}.csv", index=False)
    report = {"branch": args.branch, "method": args.method,
              "requests": len(result),
              "union_recall_at_1000_each": float(result.retrieval_recall.mean()),
              "final_recall_at_50": float(result.final_recall.mean()),
              "requests_with_any_retrieval_miss": int(result.retrieval_misses.gt(0).sum()),
              "requests_with_any_selection_miss": int(result.selection_misses.gt(0).sum()),
              "missed_positive_ads_at_retrieval": int(result.retrieval_misses.sum()),
              "missed_positive_ads_at_final_selection": int(result.selection_misses.sum())}
    save_json(exp / f"{stem}.json", report)
    print(report)


if __name__ == "__main__":
    main()
