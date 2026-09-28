"""Encode query tables with the same frozen BGE-M3 weights as documents."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from .common import query_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fit", "validation", "benchmark"], required=True)
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    queries = pd.read_parquet(root / "data" / "processed" / f"{args.dataset}_queries.parquet")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(root / "models" / "bge-m3"), device=device)
    model.max_seq_length = 256
    if device == "cuda":
        model.half()
    vectors = model.encode([query_text(r) for r in queries.to_dict("records")], batch_size=64,
                           normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    target = root / "artifacts" / "embeddings" / f"{args.dataset}_queries.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, vectors.astype(np.float16))
    print(target, vectors.shape)


if __name__ == "__main__":
    main()

