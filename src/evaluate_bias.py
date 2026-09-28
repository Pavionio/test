"""Optional geography-aware fusion of already scored direct-rerank pools.

The cross encoder still scores every candidate from both 1000-item pools.
Only a fixed additive reward for ads retrieved by exact-local search is tuned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import recall_at_k, save_json
from .evaluate import load_parts
from .fusion import geo_bias_top50


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    exp = root / "artifacts" / "experiments"
    queries = pd.read_parquet(root / "data" / "processed" / "validation_queries.parquet")
    queries = queries[queries.split.isin(["tune", "holdout"])]
    items = pd.read_parquet(root / "data" / "processed" / "train_items.parquet",
                            columns=["item_id"])
    pools = {}
    for geo in ("geo_off", "geo_exact"):
        frame = load_parts(root / "artifacts" / "cache" / "reranked" /
                           f"validation_{geo}_B")
        pools[geo] = {qid: g for qid, g in frame.groupby("eval_id", sort=False)}
    fused = {}
    for q in queries.itertuples():
        global_ = pools["geo_off"].get(q.eval_id)
        local = pools["geo_exact"].get(q.eval_id)
        score = {} if global_ is None else dict(zip(global_.item_index, global_.ce_score))
        local_ix = set() if local is None else set(local.item_index)
        if local is not None:
            score.update(zip(local.item_index, local.ce_score))
        ix = np.fromiter(score.keys(), dtype=np.int64)
        ce = np.fromiter(score.values(), dtype=np.float32)
        is_local = np.fromiter((i in local_ix for i in ix), dtype=np.float32)
        fused[q.eval_id] = (ix, ce, is_local)

    def predict(qid: str, bias: float) -> list[str]:
        ix, ce, is_local = fused[qid]
        values = ce + bias * is_local
        if len(values) > 50:
            order = np.argpartition(values, -50)[-50:]
            order = order[np.argsort(values[order])[::-1]]
        else:
            order = np.argsort(values)[::-1]
        return items.item_id.iloc[ix[order]].tolist()

    biases = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
    grid = {}
    for bias in biases:
        scores = [recall_at_k(predict(q.eval_id, bias), q.relevant)
                  for q in queries.itertuples() if q.split == "tune"]
        grid[str(bias)] = float(np.mean(scores))
    selected = max(biases, key=lambda x: (grid[str(x)], -x))
    rows = []
    for q in queries.itertuples():
        pred = items.item_id.iloc[geo_bias_top50(
            pools["geo_off"].get(q.eval_id), pools["geo_exact"].get(q.eval_id),
            selected)].tolist()
        rows.append({"eval_id": q.eval_id, "split": q.split,
                     "word_count": q.word_count, "has_filter": q.has_filter,
                     "answer": " ".join(pred),
                     "recall_at_50": recall_at_k(pred, q.relevant)})
    result = pd.DataFrame(rows)
    result.to_parquet(exp / "B_geo_bias_predictions.parquet", index=False)
    report = {"method": "cross_encoder_score_plus_exact_local_bonus",
              "bias": selected, "tune_grid": grid,
              "tune_recall_at_50": float(result[result.split.eq("tune")].recall_at_50.mean()),
              "holdout_recall_at_50": float(result[result.split.eq("holdout")].recall_at_50.mean())}
    save_json(exp / "B_geo_bias.json", report)
    print(report)


if __name__ == "__main__":
    main()
