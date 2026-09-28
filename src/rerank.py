"""Score query-item pairs with one local cross encoder in both branches."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from sentence_transformers import CrossEncoder

from .common import peak_memory_bytes, query_text, rerank_text, save_json
from .catboost_stage import parts


def cached_scores(db: sqlite3.Connection, dataset: str, qid: str) -> dict[int, float]:
    rows = db.execute("SELECT item_index, score FROM scores WHERE dataset=? AND eval_id=?",
                      (dataset, qid)).fetchall()
    return {int(i): float(s) for i, s in rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["validation", "benchmark"], required=True)
    p.add_argument("--geo", choices=["geo_off", "geo_exact"], required=True)
    p.add_argument("--branch", choices=["A", "B"], required=True)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--pilot-queries", type=int, default=0)
    p.add_argument("--audit-queries", type=int, default=0,
                   help="Score this many untouched retrieval-split queries for a final audit")
    args = p.parse_args()
    root = args.root.resolve()
    corpus = "benchmark" if args.dataset == "benchmark" else "train"
    items = pd.read_parquet(root / "data" / "processed" / f"{corpus}_items.parquet",
                            columns=["item_title_raw", "item_infm_params_text", "item_description_raw"])
    queries = pd.read_parquet(root / "data" / "processed" / f"{args.dataset}_queries.parquet")
    if args.audit_queries:
        if args.dataset != "validation" or args.pilot_queries:
            p.error("--audit-queries requires validation and cannot combine with a pilot")
        queries = queries[queries.split.eq("retrieval")].iloc[:args.audit_queries]
    elif args.dataset == "validation":
        queries = queries[queries.split.isin(["tune", "holdout"])]
    if args.pilot_queries:
        queries = queries.iloc[:args.pilot_queries]
    query_map = {r["eval_id"]: r for r in queries.to_dict("records")}
    out = root / "artifacts" / "cache" / "reranked" / f"{args.dataset}_{args.geo}_{args.branch}"
    if args.pilot_queries:
        out = out / f"pilot_{args.pilot_queries}"
    if args.audit_queries:
        out = out / f"audit_{args.audit_queries}"
    out.mkdir(parents=True, exist_ok=True)

    db_path = root / "artifacts" / "cache" / "reranker_scores.sqlite"
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS scores (dataset TEXT, eval_id TEXT, item_index INTEGER, "
               "score REAL, PRIMARY KEY (dataset, eval_id, item_index))")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(str(root / "models" / "bge-reranker-v2-m3"),
                         max_length=256, device=device)
    if device == "cuda":
        model.model.half()

    start_time = time.monotonic()
    pair_count = fresh_count = 0
    source = (root / "artifacts" / "cache" / "catboost" / f"{args.dataset}_{args.geo}") \
        if args.branch == "A" else None
    input_parts = sorted(source.glob("part_*.parquet")) if source else parts(root, args.dataset, args.geo)
    for part in input_parts:
        target = out / part.name
        if target.exists():
            continue
        frame = pd.read_parquet(part)
        frame = frame[frame.eval_id.isin(query_map)]
        scored = []
        for qid, group in frame.groupby("eval_id", sort=False):
            if args.branch == "A":
                group = group.nlargest(150, "cb_score")
            known = cached_scores(db, args.dataset, qid)
            missing = group[~group.item_index.isin(known)]
            if len(missing):
                q = query_text(query_map[qid])
                pairs = [(q, rerank_text(items.iloc[int(i)])) for i in missing.item_index]
                # Raw logits retain ordering among very low/high scores; a
                # sigmoid would saturate after FP16 inference.
                values = np.asarray(model.predict(pairs, batch_size=64,
                                                   activation_fn=torch.nn.Identity(),
                                                   show_progress_bar=False)).reshape(-1)
                entries = [(args.dataset, qid, int(i), float(s))
                           for i, s in zip(missing.item_index, values)]
                db.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?)", entries)
                db.commit()
                known.update({i: s for _, _, i, s in entries})
                fresh_count += len(entries)
            group = group.copy()
            group["ce_score"] = group.item_index.map(known)
            scored.append(group)
            pair_count += len(group)
        if scored:
            pd.concat(scored, ignore_index=True).to_parquet(target, index=False)
        print(f"{args.dataset}/{args.geo}/{args.branch}: {part.name}, "
              f"pairs={pair_count}, new={fresh_count}", flush=True)
    elapsed = time.monotonic() - start_time
    # Resume may skip previously written parts. Record the complete scored
    # population, while keeping fresh_count and elapsed for this invocation.
    complete_pairs = sum(pq.read_metadata(path).num_rows for path in out.glob("part_*.parquet"))
    timing = {"branch": args.branch, "geo": args.geo,
              "dataset": args.dataset, "pairs": complete_pairs,
              "newly_scored_pairs": fresh_count, "seconds": elapsed,
              "peak_process_memory_bytes": peak_memory_bytes(),
              "pairs_per_second": fresh_count / elapsed if elapsed else None,
              "pilot_queries": args.pilot_queries,
              "audit_queries": args.audit_queries}
    # A warm rerun may do no scoring at all. Preserve the measured cold-run
    # throughput instead of replacing it with a misleading zero-pair timing.
    if fresh_count == 0 and (out / "timing.json").exists():
        timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    save_json(out / "timing.json", timing)
    if args.pilot_queries:
        save_json(root / "artifacts" / "experiments" /
                  f"pilot_{args.dataset}_{args.geo}_{args.branch}.json", timing)
    else:
        # Cache directories are intentionally excluded from Git. Keep a small
        # timing manifest with each experiment so runtime and memory remain
        # available when the repository is cloned without score caches.
        name = (f"audit_{args.geo}_{args.branch}_timing.json" if args.audit_queries
                else f"{args.dataset}_{args.geo}_{args.branch}_timing.json")
        save_json(root / "artifacts" / "experiments" / name, timing)
    print(f"Scored {fresh_count} new pairs in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
