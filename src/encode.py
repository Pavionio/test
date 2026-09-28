"""Create reusable, normalized BGE-M3 embeddings without an external API."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from .common import item_text, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["train", "benchmark"], required=True)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=2048)
    args = p.parse_args()
    root = args.root.resolve()
    path = root / "data" / "processed" / f"{args.dataset}_items.parquet"
    items = pd.read_parquet(path, columns=["item_title_raw", "item_infm_params_text", "item_description_raw"])
    target = root / "artifacts" / "embeddings" / f"{args.dataset}_bge_m3.npy"
    progress_path = target.with_suffix(".progress.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(root / "models" / "bge-m3"), device=device)
    model.max_seq_length = 256
    if device == "cuda":
        model.half()
    dim = model.get_sentence_embedding_dimension()
    if target.exists():
        arr = np.lib.format.open_memmap(target, mode="r+")
        if arr.shape != (len(items), dim):
            raise ValueError("Existing embedding shape does not match items")
    else:
        arr = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16, shape=(len(items), dim))
    import json
    start = json.loads(progress_path.read_text(encoding="utf-8"))["next_row"] if progress_path.exists() else 0
    for left in range(start, len(items), args.chunk_size):
        right = min(left + args.chunk_size, len(items))
        texts = [item_text(row, dense=True) for row in items.iloc[left:right].to_dict("records")]
        vectors = model.encode(texts, batch_size=args.batch_size,
                               normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False)
        arr[left:right] = vectors.astype(np.float16)
        arr.flush()
        save_json(progress_path, {"next_row": right, "total": len(items), "model": "BAAI/bge-m3",
                                  "max_seq_length": 256, "dimension": dim})
        print(f"{args.dataset}: encoded {right}/{len(items)}", flush=True)
    print(target)


if __name__ == "__main__":
    main()

