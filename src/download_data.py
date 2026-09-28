"""Import the user-supplied task archive without contacting Yandex.Disk.

The archive and each extracted Parquet file are checked against the hashes of
the public dataset used for the recorded solution. Existing valid files are
reused, so an interrupted pipeline can be restarted without extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

from .common import sha256

ARCHIVE_SHA256 = "8dd3cba59201bae333c11db89c5111198fa10cc52a70c70bdc57bd6a248fb777"
INPUT_SHA256 = {
    "train.parquet": "e150ab7a5c98769f643b95ee3a02dc0f664b647facd69a7d280ec6d48ca554a7",
    "benchmark_queries.parquet": "e49de4fb76f03979a4af96034817767d39ae5de9f94e254fed028243188dda06",
    "benchmark_items.parquet": "193b3a3961464620cbf8d8797152b7f90cae1826ca749fb7817a210984a09899",
}


def verified_inputs(root: Path) -> bool:
    """Return whether all three files match the expected public inputs."""
    return all((root / name).is_file() and sha256(root / name) == digest
               for name, digest in INPUT_SHA256.items())


def extract_inputs(archive: Path, root: Path) -> None:
    """Extract only the named inputs, rejecting duplicates or modified files."""
    with zipfile.ZipFile(archive) as zf:
        members: dict[str, zipfile.ZipInfo] = {}
        for member in zf.infolist():
            name = Path(member.filename).name
            if name in INPUT_SHA256:
                if name in members:
                    raise RuntimeError(f"Duplicate {name} in archive")
                members[name] = member
        if set(members) != set(INPUT_SHA256):
            raise RuntimeError(f"Archive members do not match expected inputs: {set(members)}")
        for name, expected in INPUT_SHA256.items():
            target = root / name
            if target.is_file() and sha256(target) == expected:
                continue
            part = target.with_suffix(".parquet.part")
            digest = hashlib.sha256()
            with zf.open(members[name]) as source, part.open("wb") as out:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(block)
                    out.write(block)
            if digest.hexdigest() != expected:
                part.unlink(missing_ok=True)
                raise RuntimeError(f"Extracted {name} has a different SHA256")
            part.replace(target)
            print(f"Verified {name}", flush=True)


def import_archive(archive: Path, root: Path) -> None:
    """Require the exact dataset ZIP even when extracted files are cached."""
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive}")
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError(f"Dataset archive SHA256 mismatch: {archive}")
    if not verified_inputs(root):
        extract_inputs(archive, root)
    if not verified_inputs(root):
        raise RuntimeError("Input verification failed after extraction")
    print("All three input Parquet files match recorded SHA256", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--archive", type=Path, default=Path("dataset.zip"),
                        help="Path to the already downloaded public dataset.zip")
    args = parser.parse_args()
    import_archive(args.archive.resolve(), args.root.resolve())


if __name__ == "__main__":
    main()
