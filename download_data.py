"""Download only FD001 files from NASA's archive; never invent a fallback."""
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", "readme.txt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="data")
    args = parser.parse_args()
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in FILES):
        raise SystemExit("Data already exists; choose a new destination to avoid overwriting.")
    try:
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            contents = {Path(name).name: archive.read(name) for name in archive.namelist()
                        if Path(name).name in FILES}
        if set(contents) != set(FILES):
            raise ValueError("Archive does not contain expected FD001 files")
        for name, content in contents.items():
            (destination / name).write_bytes(content)
        manifest = {"source_url": URL, "archive_sha256": hashlib.sha256(payload).hexdigest(),
                    "files": {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}}
        (destination / "source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"NASA FD001 saved to {destination}")
    except Exception as exc:
        raise SystemExit(f"NASA download failed: {exc}. Obtain the three original FD001 text files manually.")


if __name__ == "__main__":
    main()
