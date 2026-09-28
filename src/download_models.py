"""Download pinned open-source weights once; inference itself stays offline."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import HfApi, snapshot_download

from .common import save_json, sha256

MODELS = [
    ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181",
     "bge-m3", "pytorch_model.bin",
     "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"),
    ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
     "bge-reranker-v2-m3", "model.safetensors",
     "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286"),
]


def stream_weight(repo: str, revision: str, filename: str, target: Path, expected: int) -> None:
    if target.exists() and target.stat().st_size == expected:
        return
    part = target.with_suffix(target.suffix + ".part")
    state_path = target.with_suffix(target.suffix + ".chunks.json")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true"
    chunk_size = 16 * 1024 * 1024
    count = math.ceil(expected / chunk_size)
    done = set(json.loads(state_path.read_text(encoding="utf-8"))) if state_path.exists() else set()
    if not part.exists() and done:
        raise RuntimeError("Chunk state exists but the partial weight file is missing")
    with part.open("r+b" if part.exists() else "w+b") as f:
        f.truncate(expected)
    lock = threading.Lock()

    def fetch(i: int) -> int:
        start = i * chunk_size
        end = min(expected, start + chunk_size) - 1
        for attempt in range(6):
            try:
                response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                response.raise_for_status()
                block = response.content
                if response.status_code != 206 or len(block) != end - start + 1:
                    raise IOError(f"Bad range response {response.status_code}: {len(block)} bytes")
                with part.open("r+b") as out:
                    out.seek(start)
                    out.write(block)
                with lock:
                    done.add(i)
                    state_path.write_text(json.dumps(sorted(done)), encoding="utf-8")
                    if len(done) % 10 == 0 or len(done) == count:
                        print(f"{filename}: {len(done)}/{count} chunks", flush=True)
                return i
            except Exception as exc:
                if attempt == 5:
                    raise
                print(f"{filename} chunk {i}: retry {attempt+1}: {exc}", flush=True)
                time.sleep(min(2 ** attempt, 15))
        raise RuntimeError(f"Could not download chunk {i}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed(pool.submit(fetch, i) for i in range(count) if i not in done):
            future.result()
    if len(done) != count:
        raise RuntimeError(f"Incomplete weight: {len(done)}/{count} chunks")
    part.replace(target)
    state_path.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root.resolve()
    api = HfApi()
    manifest = {}
    for repo, revision, local, weight, expected_sha256 in MODELS:
        folder = root / "models" / local
        folder.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo, revision=revision, local_dir=folder,
                          allow_patterns=["*.json", "*.model"], max_workers=4)
        info = api.model_info(repo, revision=revision, files_metadata=True)
        expected = next(s.size for s in info.siblings if s.rfilename == weight)
        print(f"Downloading {repo}/{weight}: {expected/1e9:.2f} GB", flush=True)
        stream_weight(repo, revision, weight, folder / weight, expected)
        actual_sha256 = sha256(folder / weight)
        if actual_sha256 != expected_sha256:
            (folder / weight).unlink()
            raise RuntimeError(f"SHA256 mismatch for {repo}/{weight}: {actual_sha256}")
        manifest[repo] = {"revision": revision, "weight_file": str(Path("models") / local / weight),
                          "weight_sha256": actual_sha256, "weight_bytes": expected}
        save_json(root / "models" / "manifest.json", manifest)
        print(f"Ready: {repo}", flush=True)


if __name__ == "__main__":
    main()
