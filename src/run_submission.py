"""One resumable Docker command from a supplied ZIP to the benchmark answer.

Only the already selected B pipeline runs here. Validation and CatBoost
experiments remain in scripts/run_all.ps1 for research reproduction, but are
not needed to generate the submitted answer.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import torch

from .common import sha256
from .download_data import INPUT_SHA256
from .download_models import MODELS


def inputs_ready(root: Path) -> bool:
    return all((root / name).is_file() and sha256(root / name) == digest
               for name, digest in INPUT_SHA256.items())


def models_ready(root: Path) -> bool:
    manifest_path = root / "models" / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for repo, revision, folder, weight, digest in MODELS:
        entry = manifest.get(repo, {})
        path = root / "models" / folder / weight
        if (entry.get("revision") != revision or entry.get("weight_sha256") != digest
                or not path.is_file() or sha256(path) != digest):
            return False
        # Both model loaders need metadata and a tokenizer beside the weight.
        if not (path.parent / "config.json").is_file() or not (path.parent / "tokenizer.json").is_file():
            return False
    return True


def wait_for_qdrant(url: str, timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = requests.get(url.rstrip("/") + "/", timeout=5)
            if response.ok and response.json().get("title") == "qdrant - vector search engine":
                return
        except (requests.RequestException, ValueError):
            pass
        time.sleep(3)
    raise TimeoutError(f"Qdrant did not become available at {url}")


def run(root: Path, module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset-zip", type=Path, default=Path("/input/dataset.zip"),
                        help="Path to the user-supplied dataset.zip")
    parser.add_argument("--skip-model-download", action="store_true",
                        help="Use previously verified model weights")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned commands without executing them")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.dry_run:
        for step in ("import supplied dataset.zip", "download_models", "prepare", "queries",
                     "lexical benchmark", "encode benchmark", "index benchmark",
                     "encode_queries benchmark", "retrieve benchmark geo_off",
                     "retrieve benchmark geo_exact", "rerank benchmark geo_off B",
                     "rerank benchmark geo_exact B", "finalize B geo_exact geo_bias"):
            print(step)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("The benchmark run requires a CUDA GPU; check Docker GPU access")
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    wait_for_qdrant(url)

    # The input archive is supplied through a read-only Docker bind mount.
    # No request to Yandex.Disk is made by the pipeline.
    run(root, "src.download_data", "--archive", str(args.dataset_zip))
    if not args.skip_model_download and not models_ready(root):
        run(root, "src.download_models")
    if not inputs_ready(root) or not models_ready(root):
        raise RuntimeError("Input data or pinned model weights are missing or corrupted")
    # Each stage runs in a separate process to release GPU and CPU memory.
    # Existing chunk/checkpoint files let this command resume after failure.
    run(root, "src.prepare")
    run(root, "src.queries")
    run(root, "src.lexical", "--dataset", "benchmark")
    run(root, "src.encode", "--dataset", "benchmark")
    run(root, "src.index", "--dataset", "benchmark")
    run(root, "src.encode_queries", "--dataset", "benchmark")
    for geo in ("geo_off", "geo_exact"):
        run(root, "src.retrieve", "--dataset", "benchmark", "--geo", geo)
    for geo in ("geo_off", "geo_exact"):
        run(root, "src.rerank", "--dataset", "benchmark", "--geo", geo, "--branch", "B")
    run(root, "src.finalize", "--branch", "B", "--geo", "geo_exact", "--geo-bias")
    print(f"\nReady: {root / 'answer.csv'}", flush=True)


if __name__ == "__main__":
    main()
