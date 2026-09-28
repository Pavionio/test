"""Hybrid candidate search. Every branch uses this identical 1000-item pool."""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import bm25s
import joblib
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from .common import normalize_query, peak_memory_bytes, query_text, save_json, tokens, topk

CHANNELS = ("title", "body", "char", "dense", "history")
FEATURES = ["rrf"] + [f"{c}_{v}" for c in CHANNELS for v in ("rank", "score")]


class HybridSearch:
    def __init__(self, root: Path, dataset: str, history: bool):
        self.root = root
        self.dataset = dataset
        items = pd.read_parquet(root / "data" / "processed" / f"{dataset}_items.parquet",
                                columns=["item_id", "item_location_id"])
        self.item_ids = items.item_id.astype(str).tolist()
        self.locations = items.item_location_id.to_numpy(dtype=np.int64)
        self.id_to_row = {s: i for i, s in enumerate(self.item_ids)}
        base = root / "artifacts" / "indexes" / dataset
        self.bm25 = {c: bm25s.BM25.load(str(base / c), load_corpus=False, show_progress=False)
                     for c in ("title", "body")}
        self.char_vectorizer, self.char_matrix = joblib.load(base / "title_char.joblib")
        self.client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"), grpc_port=6334,
                                   prefer_grpc=True, timeout=120)
        self.collection = f"avito_{dataset}_bge_m3"
        self.history = defaultdict(list)
        if history and dataset == "benchmark":
            pairs = pd.read_parquet(root / "train.parquet", columns=["search_query", "item_id"])
            counts = pairs.groupby(["search_query", "item_id"]).size().reset_index(name="clicks")
            counts["norm"] = counts.search_query.map(normalize_query)
            counts = counts.sort_values("clicks", ascending=False)
            for text, item_id in zip(counts.norm, counts.item_id):
                ix = self.id_to_row.get(item_id)
                if ix is not None and len(self.history[text]) < 100:
                    self.history[text].append(ix)

    def search(self, row: dict, vector: np.ndarray, geo: str) -> list[dict]:
        local = geo == "geo_exact" and not int(row["search_is_delivery_search"])
        loc = int(row["search_location_id"])
        mask = (self.locations == loc).astype(np.float32) if local else None
        results: dict[str, list[tuple[int, float]]] = {name: [] for name in CHANNELS}
        qt = tokens(query_text(row))
        for name in ("title", "body"):
            hit = self.bm25[name].retrieve([qt], k=min(400, len(self.item_ids)),
                                            weight_mask=mask, show_progress=False)
            for ix, score in zip(hit.documents[0], hit.scores[0]):
                ix = int(ix)
                if score > 0 and (not local or self.locations[ix] == loc):
                    results[name].append((ix, float(score)))

        # Sparse multiplication only visits titles sharing character n-grams.
        char_q = self.char_vectorizer.transform([str(row["search_query"])])
        sims = (char_q @ self.char_matrix.T).tocoo()
        ids, scores = sims.col, sims.data
        if local:
            keep = self.locations[ids] == loc
            ids, scores = ids[keep], scores[keep]
        if len(scores):
            order = topk(scores, 400)
            results["char"] = [(int(ids[i]), float(scores[i])) for i in order]

        qfilter = None
        if local:
            qfilter = models.Filter(must=[models.FieldCondition(
                key="item_location_id", match=models.MatchValue(value=loc))])
        dense = self.client.query_points(
            collection_name=self.collection, query=vector.astype(np.float32).tolist(),
            query_filter=qfilter, limit=1000, with_payload=False,
            search_params=models.SearchParams(hnsw_ef=128),
        ).points
        results["dense"] = [(int(hit.id), float(hit.score)) for hit in dense]

        for ix in self.history.get(normalize_query(row["search_query"]), []):
            if not local or self.locations[ix] == loc:
                results["history"].append((ix, 1.0))

        features = defaultdict(lambda: {"rrf": 0.0, **{f"{c}_rank": 10001 for c in CHANNELS},
                                        **{f"{c}_score": 0.0 for c in CHANNELS}})
        for channel, hits in results.items():
            for rank, (ix, score) in enumerate(hits, 1):
                feat = features[ix]
                feat["rrf"] += 1.0 / (60 + rank)
                feat[f"{channel}_rank"] = rank
                feat[f"{channel}_score"] = score
        ranked = sorted(features, key=lambda ix: (-features[ix]["rrf"], ix))[:1000]
        return [{"item_index": ix, **features[ix]} for ix in ranked]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fit", "validation", "benchmark"], required=True)
    p.add_argument("--geo", choices=["geo_off", "geo_exact"], required=True)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--chunk-queries", type=int, default=25)
    p.add_argument("--limit", type=int, default=0, help="Only for throughput pilot")
    args = p.parse_args()
    root = args.root.resolve()
    corpus = "benchmark" if args.dataset == "benchmark" else "train"
    queries = pd.read_parquet(root / "data" / "processed" / f"{args.dataset}_queries.parquet")
    if args.limit:
        queries = queries.iloc[:args.limit]
    vectors = np.load(root / "artifacts" / "embeddings" / f"{args.dataset}_queries.npy", mmap_mode="r")
    search = HybridSearch(root, corpus, history=args.dataset == "benchmark")
    started = time.monotonic()
    out = root / "artifacts" / "cache" / "candidates" / f"{args.dataset}_{args.geo}"
    if args.limit:
        out = out / f"pilot_{args.limit}"
    out.mkdir(parents=True, exist_ok=True)
    for left in range(0, len(queries), args.chunk_queries):
        target = out / f"part_{left:05d}.parquet"
        if target.exists():
            continue
        rows = []
        for j in range(left, min(left + args.chunk_queries, len(queries))):
            query = queries.iloc[j].to_dict()
            for candidate in search.search(query, vectors[j], args.geo):
                rows.append({"eval_id": query["eval_id"], **candidate})
        pd.DataFrame(rows, columns=["eval_id", "item_index"] + FEATURES).to_parquet(target, index=False)
        print(f"{args.dataset}/{args.geo}: {min(left+args.chunk_queries,len(queries))}/{len(queries)}", flush=True)
    save_json(out / "manifest.json", {"dataset": args.dataset, "geo": args.geo,
                                      "queries": len(queries), "corpus": corpus, "top_k": 1000,
                                      "seconds": time.monotonic() - started,
                                      "peak_process_memory_bytes": peak_memory_bytes()})


if __name__ == "__main__":
    main()
