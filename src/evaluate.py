"""Query-macro Recall@K and leak-free branch selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import recall_at_k, save_json
from .retrieve import CHANNELS


def load_parts(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("part_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No scored parts in {folder}")
    return pd.concat((pd.read_parquet(x) for x in files), ignore_index=True)


def metric_rows(frame: pd.DataFrame, queries: pd.DataFrame, items: pd.DataFrame,
                k: int, score_col: str | None = None) -> pd.DataFrame:
    labels = {r.eval_id: set(r.relevant) for r in queries.itertuples()}
    hits = {}
    for qid, group in frame.groupby("eval_id", sort=False):
        if score_col:
            group = group.nlargest(k, score_col)
        else:
            group = group.head(k)
        predicted = items.item_id.iloc[group.item_index.astype(int)].tolist()
        hits[qid] = recall_at_k(predicted, labels[qid], k)
    out = queries[["eval_id", "split", "word_count", "has_filter"]].copy()
    out["recall"] = out.eval_id.map(hits).fillna(0.0)
    return out


def retrieval(root: Path) -> None:
    queries = pd.read_parquet(root / "data" / "processed" / "validation_queries.parquet")
    items = pd.read_parquet(root / "data" / "processed" / "train_items.parquet", columns=["item_id"])
    report = {}
    for geo in ("geo_off", "geo_exact"):
        folder = root / "artifacts" / "cache" / "candidates" / f"validation_{geo}"
        frame = load_parts(folder)
        metrics = metric_rows(frame, queries, items, 1000)
        labels = {r.eval_id: set(r.relevant) for r in queries.itertuples()}
        channel_hits = {channel: {} for channel in CHANNELS}
        for qid, group in frame.groupby("eval_id", sort=False):
            for channel in CHANNELS:
                subset = group[group[f"{channel}_rank"].lt(10001)]
                ids = items.item_id.iloc[subset.item_index.astype(int)].tolist()
                channel_hits[channel][qid] = recall_at_k(ids, list(labels[qid]), 1000)
        report[geo] = {"recall_at_1000": metrics.recall.mean(),
                       "empty_queries": int((~queries.eval_id.isin(frame.eval_id)).sum()),
                       "by_filter": metrics.groupby("has_filter").recall.mean().to_dict(),
                       "by_query_words": metrics.groupby("word_count").recall.mean().to_dict(),
                       "retrieval_timing": json.loads((folder / "manifest.json").read_text(encoding="utf-8")),
                       "per_channel_recall_within_union": {
                           channel: float(np.mean([channel_hits[channel].get(qid, 0.0)
                                                   for qid in queries.eval_id]))
                           for channel in CHANNELS}}
        metrics.to_csv(root / "artifacts" / "experiments" / f"retrieval_{geo}_per_query.csv", index=False)
    save_json(root / "artifacts" / "experiments" / "retrieval.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


def rank_group(group: pd.DataFrame, branch: str, alpha: float, k: int = 50) -> pd.DataFrame:
    group = group.copy()
    group["ce_pct"] = group.ce_score.rank(pct=True, method="average")
    if branch == "A":
        group["cb_pct"] = group.cb_score.rank(pct=True, method="average")
        group["final_score"] = group.ce_pct + alpha * group.cb_pct
    else:
        group["final_score"] = group.ce_score
    return group.nlargest(k, "final_score")


def branch(root: Path, geo: str, name: str) -> None:
    folder = root / "artifacts" / "cache" / "reranked" / f"validation_{geo}_{name}"
    frame = load_parts(folder)
    queries = pd.read_parquet(root / "data" / "processed" / "validation_queries.parquet")
    queries = queries[queries.split.isin(["tune", "holdout"])]
    items = pd.read_parquet(root / "data" / "processed" / "train_items.parquet", columns=["item_id"])
    labels = {r.eval_id: set(r.relevant) for r in queries.itertuples()}
    grouped = {qid: group for qid, group in frame.groupby("eval_id", sort=False)}
    # Include strong CatBoost weights: its learned location and retrieval
    # features can be more reliable than the cross encoder on short queries.
    alphas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] if name == "A" else [0.0]
    tune_scores = {}
    for alpha in alphas:
        values = []
        for q in queries.itertuples():
            if q.split != "tune":
                continue
            g = grouped.get(q.eval_id)
            pred = [] if g is None else items.item_id.iloc[rank_group(g, name, alpha).item_index.astype(int)].tolist()
            values.append(recall_at_k(pred, list(labels[q.eval_id])))
        tune_scores[str(alpha)] = float(np.mean(values))
    alpha = max(alphas, key=lambda x: (tune_scores[str(x)], -x))
    rows, preds = [], []
    for q in queries.itertuples():
        g = grouped.get(q.eval_id)
        pred = [] if g is None else items.item_id.iloc[rank_group(g, name, alpha).item_index.astype(int)].tolist()
        rows.append({"eval_id": q.eval_id, "split": q.split,
                     "word_count": q.word_count, "has_filter": q.has_filter,
                     "recall_at_50": recall_at_k(pred, list(labels[q.eval_id]))})
        preds.append({"eval_id": q.eval_id, "split": q.split, "answer": " ".join(pred)})
    per_query = pd.DataFrame(rows)
    base = root / "artifacts" / "experiments" / f"{name}_{geo}"
    per_query.to_csv(base.with_name(base.name + "_per_query.csv"), index=False)
    pd.DataFrame(preds).to_parquet(base.with_name(base.name + "_predictions.parquet"), index=False)
    report = {"branch": name, "geo": geo, "alpha": alpha,
              "tune_grid": tune_scores,
              "tune_recall_at_50": per_query.loc[per_query.split.eq("tune"), "recall_at_50"].mean(),
              "holdout_recall_at_50": per_query.loc[per_query.split.eq("holdout"), "recall_at_50"].mean(),
              "by_filter_holdout": per_query[per_query.split.eq("holdout")]
                  .groupby("has_filter").recall_at_50.mean().to_dict(),
              "by_words_holdout": per_query[per_query.split.eq("holdout")]
                  .groupby("word_count").recall_at_50.mean().to_dict(),
              "empty_queries": int((~queries.eval_id.isin(frame.eval_id)).sum()),
              "reranker_timing": json.loads((folder / "timing.json").read_text(encoding="utf-8"))}
    if name == "A":
        cb = load_parts(root / "artifacts" / "cache" / "catboost" / f"validation_{geo}")
        cb = cb[cb.eval_id.isin(queries.eval_id)]
        pre = metric_rows(cb, queries, items, 150, score_col="cb_score")
        report["prefilter_recall_at_150"] = float(pre.recall.mean())
    save_json(base.with_suffix(".json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["retrieval", "branch"])
    p.add_argument("--branch", choices=["A", "B"])
    p.add_argument("--geo", choices=["geo_off", "geo_exact"])
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    (root / "artifacts" / "experiments").mkdir(parents=True, exist_ok=True)
    if args.stage == "retrieval":
        retrieval(root)
    else:
        if not args.branch or not args.geo:
            p.error("branch requires --branch and --geo")
        branch(root, args.geo, args.branch)


if __name__ == "__main__":
    main()
