"""Build leakage-safe query splits and item tables from train.parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import ITEM_COLS, SEARCH_COLS, normalize_query, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--seed", type=int, default=20260928)
    args = p.parse_args()
    root = args.root.resolve()
    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(root / "train.parquet", columns=SEARCH_COLS + ITEM_COLS)
    train = train.drop_duplicates(SEARCH_COLS + ["item_id"])
    for c in ("search_query", "search_infm_params_text"):
        train[c] = train[c].fillna("").astype(str)
    for c in ("item_title_raw", "item_description_raw", "item_infm_params_text"):
        train[c] = train[c].fillna("").astype(str)
    for c in ("item_price", "item_latitude", "item_longitude"):
        train[c] = pd.to_numeric(train[c], errors="coerce")

    # One stable snapshot per item. Item metadata can be inspected at retrieval
    # time even when the item's click is held out for validation.
    items = train[ITEM_COLS].drop_duplicates("item_id").reset_index(drop=True)
    items.to_parquet(out / "train_items.parquet", index=False)

    # Group all positive item IDs by the *complete* query. Then hold out whole
    # normalized query texts, avoiding leakage through another city/filter.
    grouped = train.groupby(SEARCH_COLS, dropna=False, sort=False)["item_id"].agg(list).reset_index()
    grouped["split_text"] = grouped.search_query.map(normalize_query)
    grouped["word_count"] = grouped.search_query.map(lambda s: len(normalize_query(s).split()))
    grouped["has_filter"] = grouped.search_infm_params_text.ne("")
    rng = np.random.default_rng(args.seed)
    representatives = grouped.drop_duplicates("split_text")
    texts = representatives["split_text"].to_numpy()
    if len(texts) < 2000:
        raise RuntimeError("At least 2000 unique query texts are required")
    # Sampling query texts, rather than click rows, prevents popular queries
    # from dominating validation. The stored list makes runs reproducible.
    chosen_texts = rng.choice(texts, size=2000, replace=False)
    representative = representatives.set_index("split_text").loc[chosen_texts]
    strata = (representative.word_count.clip(upper=5).astype(str) + "_" +
              representative.has_filter.astype(int).astype(str)).to_numpy()
    retrieval_texts, full_eval_texts = train_test_split(
        chosen_texts, test_size=600, random_state=args.seed, stratify=strata)
    full_strata = (representatives.set_index("split_text").loc[full_eval_texts]
                   .word_count.clip(upper=5).astype(str) + "_" +
                   representatives.set_index("split_text").loc[full_eval_texts]
                   .has_filter.astype(int).astype(str)).to_numpy()
    tune_texts, holdout_texts = train_test_split(
        full_eval_texts, test_size=200, random_state=args.seed + 1, stratify=full_strata)
    split_map = {t: "retrieval" for t in retrieval_texts}
    split_map.update({t: "tune" for t in tune_texts})
    split_map.update({t: "holdout" for t in holdout_texts})
    grouped["split"] = grouped.split_text.map(split_map).fillna("train")

    # One full query record per selected text; keep every positive item for
    # that record. Random representative avoids preferring unusually popular
    # query/filter combinations with many observed positives.
    val = grouped[grouped.split.ne("train")].copy()
    val = val.iloc[rng.permutation(len(val))]
    val = val.drop_duplicates("split_text").reset_index(drop=True)
    val = val.rename(columns={"item_id": "relevant"})
    val.insert(0, "eval_id", [f"v{i:05d}" for i in range(len(val))])
    val.to_parquet(out / "validation_queries.parquet", index=False)

    # All rows for any held-out text are excluded from CatBoost training and
    # historical-click lookup. Keep only the columns needed for these tasks.
    fitting = train[~train.search_query.map(normalize_query).isin(set(chosen_texts))]
    fitting[SEARCH_COLS + ["item_id"]].to_parquet(out / "fit_pairs.parquet", index=False)

    bitems = pd.read_parquet(root / "benchmark_items.parquet", columns=ITEM_COLS)
    for c in ("item_title_raw", "item_description_raw", "item_infm_params_text"):
        bitems[c] = bitems[c].fillna("").astype(str)
    for c in ("item_price", "item_latitude", "item_longitude"):
        bitems[c] = pd.to_numeric(bitems[c], errors="coerce")
    bitems.to_parquet(out / "benchmark_items.parquet", index=False)

    save_json(out / "split.json", {
        "seed": args.seed,
        "train_rows": len(fitting),
        "train_items": len(items),
        "split_texts": {name: sorted([t for t, s in split_map.items() if s == name])
                        for name in ("retrieval", "tune", "holdout")},
        "evaluation_query_ids": {name: val.loc[val.split.eq(name), "eval_id"].tolist()
                                 for name in ("retrieval", "tune", "holdout")},
    })
    print(f"Prepared {len(items)} train items and {len(val)} validation queries")


if __name__ == "__main__":
    main()
