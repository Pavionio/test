"""Freeze training, validation and benchmark query tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import SEARCH_COLS, normalize_query


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--fit-count", type=int, default=2500)
    args = p.parse_args()
    root = args.root.resolve()
    out = root / "data" / "processed"
    pairs = pd.read_parquet(out / "fit_pairs.parquet")
    groups = pairs.groupby(SEARCH_COLS, sort=False, dropna=False).item_id.agg(list).reset_index()
    groups["split_text"] = groups.search_query.map(normalize_query)
    groups = groups.drop_duplicates("split_text").reset_index(drop=True)
    rng = np.random.default_rng(args.seed + 1)
    selected = groups.iloc[rng.choice(len(groups), min(args.fit_count, len(groups)), replace=False)].copy()
    selected.insert(0, "eval_id", [f"t{i:05d}" for i in range(len(selected))])
    selected = selected.rename(columns={"item_id": "relevant"})
    selected.to_parquet(out / "fit_queries.parquet", index=False)

    benchmark = pd.read_parquet(root / "benchmark_queries.parquet")
    benchmark = benchmark.rename(columns={"query_id": "eval_id"})
    benchmark.to_parquet(out / "benchmark_queries.parquet", index=False)
    print(f"Training queries: {len(selected)}; benchmark queries: {len(benchmark)}")


if __name__ == "__main__":
    main()

