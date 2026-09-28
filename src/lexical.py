"""BM25 and character TF-IDF indexes, built without relevance labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import bm25s
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import item_text, tokens


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["train", "benchmark"], required=True)
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    items = pd.read_parquet(root / "data" / "processed" / f"{args.dataset}_items.parquet",
                            columns=["item_title_raw", "item_description_raw", "item_infm_params_text"])
    base = root / "artifacts" / "indexes" / args.dataset
    base.mkdir(parents=True, exist_ok=True)

    for name, texts in (
        ("title", items.item_title_raw.fillna("").tolist()),
        ("body", [item_text(r) for r in items.to_dict("records")]),
    ):
        path = base / name
        if path.exists():
            print(f"Using existing {path}", flush=True)
            continue
        index = bm25s.BM25()
        index.index([tokens(x) for x in texts], show_progress=True)
        index.save(str(path))
        print(f"Saved {path}", flush=True)

    char_path = base / "title_char.joblib"
    if not char_path.exists():
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                     max_features=900000, dtype=np.float32)
        matrix = vectorizer.fit_transform(items.item_title_raw.fillna("").astype(str))
        joblib.dump((vectorizer, matrix), char_path, compress=0)
        print(f"Saved {char_path}, shape={matrix.shape}", flush=True)


if __name__ == "__main__":
    main()
