"""Offline checks for importing the user-supplied dataset archive."""

import hashlib
import zipfile

from src import download_data


def test_extract_verifies_contents_and_nested_paths(tmp_path, monkeypatch):
    contents = {name: f"contents of {name}".encode()
                for name in download_data.INPUT_SHA256}
    monkeypatch.setattr(download_data, "INPUT_SHA256",
                        {name: hashlib.sha256(value).hexdigest()
                         for name, value in contents.items()})
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, value in contents.items():
            zf.writestr(f"archive/inputs/{name}", value)
    download_data.extract_inputs(archive, tmp_path)
    assert download_data.verified_inputs(tmp_path)
    for name, value in contents.items():
        assert (tmp_path / name).read_bytes() == value


def test_import_archive_requires_path_and_checks_zip_hash(tmp_path, monkeypatch):
    contents = {name: f"contents of {name}".encode()
                for name in download_data.INPUT_SHA256}
    monkeypatch.setattr(download_data, "INPUT_SHA256",
                        {name: hashlib.sha256(value).hexdigest()
                         for name, value in contents.items()})
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, value in contents.items():
            zf.writestr(name, value)
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(download_data, "ARCHIVE_SHA256", "incorrect")
    try:
        download_data.import_archive(archive, output)
    except ValueError as exc:
        assert "SHA256 mismatch" in str(exc)
    else:
        raise AssertionError("Modified archive was accepted")
    monkeypatch.setattr(download_data, "ARCHIVE_SHA256",
                        hashlib.sha256(archive.read_bytes()).hexdigest())
    download_data.import_archive(archive, output)
    assert download_data.verified_inputs(output)
