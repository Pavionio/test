"""Train and apply the CatBoost prefilter using retrieved hard negatives."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .common import peak_memory_bytes, save_json
from .features import FeatureBuilder, MODEL_FEATURES


def parts(root: Path, dataset: str, geo: str) -> list[Path]:
    return sorted((root / "artifacts" / "cache" / "candidates" /
                   f"{dataset}_{geo}").glob("part_*.parquet"))


def model_path(root: Path, geo: str, full: bool) -> Path:
    base = "catboost_final" if full else "catboost_prefilter"
    return root / "artifacts" / "models" / f"{base}{'_geo_exact' if geo == 'geo_exact' else ''}.cbm"


def train(root: Path, full: bool = False, geo: str = "geo_off") -> None:
    started = time.monotonic()
    rng = np.random.default_rng(20260928)
    xs, ys = [], []
    found, total = 0, 0
    datasets = ["fit", "validation"] if full else ["fit"]
    for dataset in datasets:
        builder = FeatureBuilder(root, "train", dataset)
        total += sum(len(q["relevant"]) for q in builder.queries.values())
        for part in parts(root, dataset, geo):
            candidates = pd.read_parquet(part)
            for qid, group in candidates.groupby("eval_id", sort=False):
                relevant = set(builder.queries[qid]["relevant"])
                positive = group.item_index.map(lambda i: builder.items.item_id.iat[int(i)] in relevant)
                found += positive.sum()
                if not positive.any():
                    continue
                neg = group.loc[~positive]
                # Both high-ranking and lower-ranking hard negatives teach the
                # prefilter to preserve positives beyond lexical matches.
                head = neg.head(25)
                tail = neg.iloc[25:]
                if len(tail) > 25:
                    tail = tail.iloc[rng.choice(len(tail), 25, replace=False)]
                sample = pd.concat([group.loc[positive], head, tail], ignore_index=True)
                label = sample.item_index.map(lambda i: builder.items.item_id.iat[int(i)] in relevant).astype(int)
                xs.append(builder.build(sample))
                ys.append(label)
    X = pd.concat(xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    model = CatBoostClassifier(iterations=700, depth=7, learning_rate=0.05,
                               loss_function="Logloss", eval_metric="AUC",
                               auto_class_weights="Balanced", random_seed=20260928,
                               thread_count=8, verbose=100)
    model.fit(X, y)
    path = model_path(root, geo, full)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    save_json(path.with_suffix(".json"), {
        "positive_candidates": int(found), "positives": int(total),
        "training_rows": len(X), "features": MODEL_FEATURES,
        "seed": 20260928, "iterations": 700, "depth": 7,
        "full": full, "geo": geo,
        "seconds": time.monotonic() - started,
        "peak_process_memory_bytes": peak_memory_bytes(),
    })
    print(f"Saved {path}; positive retrieval {found}/{total}")


def score(root: Path, dataset: str, geo: str, final: bool = False) -> None:
    started = time.monotonic()
    corpus = "benchmark" if dataset == "benchmark" else "train"
    builder = FeatureBuilder(root, corpus, dataset)
    model = CatBoostClassifier()
    model.load_model(str(model_path(root, geo, final)))
    out = root / "artifacts" / "cache" / "catboost" / f"{dataset}_{geo}"
    out.mkdir(parents=True, exist_ok=True)
    for part in parts(root, dataset, geo):
        target = out / part.name
        if target.exists():
            continue
        candidates = pd.read_parquet(part)
        if len(candidates):
            candidates["cb_score"] = model.predict_proba(builder.build(candidates))[:, 1]
        else:
            candidates["cb_score"] = pd.Series(dtype=float)
        candidates.to_parquet(target, index=False)
        print(f"Scored {target.name}", flush=True)
    save_json(out / "timing.json", {"dataset": dataset, "geo": geo, "final": final,
                                     "seconds": time.monotonic() - started,
                                     "peak_process_memory_bytes": peak_memory_bytes()})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["train", "score"])
    p.add_argument("--dataset", choices=["fit", "validation", "benchmark"])
    p.add_argument("--geo", choices=["geo_off", "geo_exact"])
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--full", action="store_true", help="Refit using held-out labels after selection")
    p.add_argument("--final", action="store_true", help="Score with the refitted final model")
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "train":
        train(root, full=args.full, geo=args.geo or "geo_off")
    else:
        if not args.dataset or not args.geo:
            p.error("score requires --dataset and --geo")
        score(root, args.dataset, args.geo, final=args.final)


if __name__ == "__main__":
    main()
