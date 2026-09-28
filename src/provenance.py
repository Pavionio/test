"""Record inputs and runtime so every result can be audited later."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import requests
import torch

from .common import save_json, sha256

PACKAGES = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "catboost",
            "qdrant-client", "bm25s", "transformers", "sentence-transformers",
            "huggingface-hub", "torch", "joblib", "psutil", "python-dateutil"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    files = ["train.parquet", "benchmark_queries.parquet", "benchmark_items.parquet",
             "answer.csv",
             "models/bge-m3/pytorch_model.bin",
             "models/bge-reranker-v2-m3/model.safetensors",
             "artifacts/models/catboost_prefilter.cbm",
             "artifacts/models/catboost_prefilter_geo_exact.cbm",
             "artifacts/models/catboost_final.cbm",
             "artifacts/models/catboost_final_geo_exact.cbm"]
    try:
        qdrant = requests.get(os.environ.get("QDRANT_URL", "http://localhost:6333") + "/", timeout=5).json()
    except Exception:
        qdrant = None
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {x: importlib.metadata.version(x) for x in PACKAGES},
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "qdrant": qdrant,
        "sha256": {x: sha256(root / x) for x in files if (root / x).exists()},
    }
    save_json(root / "artifacts" / "experiments" / "provenance.json", report)
    print("Saved artifacts/experiments/provenance.json")


if __name__ == "__main__":
    main()
