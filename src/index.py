"""Upload stored embeddings to a local Qdrant collection with location payload."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["train", "benchmark"], required=True)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    args = p.parse_args()
    root = args.root.resolve()
    items = pd.read_parquet(root / "data" / "processed" / f"{args.dataset}_items.parquet",
                            columns=["item_location_id"])
    arr = np.load(root / "artifacts" / "embeddings" / f"{args.dataset}_bge_m3.npy", mmap_mode="r")
    if len(items) != len(arr):
        raise ValueError("Item order and vector count differ")
    name = f"avito_{args.dataset}_bge_m3"
    # gRPC transmits binary float vectors and is substantially faster than
    # JSON HTTP for hundreds of thousands of 1024-dimensional points.
    client = QdrantClient(url=args.url, grpc_port=6334, prefer_grpc=True, timeout=120)
    collections = {c.name for c in client.get_collections().collections}
    if name not in collections:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=arr.shape[1], distance=models.Distance.COSINE,
                                               on_disk=False),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, on_disk=False),
        )
        client.create_payload_index(collection_name=name, field_name="item_location_id",
                                    field_schema=models.PayloadSchemaType.INTEGER)
    start = client.count(name, exact=True).count
    if start > len(items):
        raise ValueError("Collection contains more points than item table")
    for left in range(start, len(items), args.batch_size):
        right = min(left + args.batch_size, len(items))
        points = [models.PointStruct(id=i, vector=arr[i].astype(np.float32).tolist(),
                                     payload={"item_location_id": int(items.item_location_id.iat[i])})
                  for i in range(left, right)]
        client.upsert(collection_name=name, points=points, wait=True)
        if right % (args.batch_size * 25) == 0 or right == len(items):
            print(f"{name}: {right}/{len(items)}", flush=True)
    print(f"Indexed {client.count(name, exact=True).count} points in {name}")
    # Qdrant builds HNSW segments asynchronously after upserts. Wait before
    # measuring retrieval so every experiment sees the same completed index.
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        info = client.get_collection(name)
        if info.status == models.CollectionStatus.GREEN:
            print(f"{name}: ready (indexed_vectors={info.indexed_vectors_count})")
            break
        print(f"{name}: optimizer status {info.status}", flush=True)
        time.sleep(15)
    else:
        raise TimeoutError(f"Qdrant collection {name} did not become ready")


if __name__ == "__main__":
    main()
