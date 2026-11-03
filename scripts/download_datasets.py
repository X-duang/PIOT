#!/usr/bin/env python3
"""Download the exact DADER benchmark CSVs used in the paper and check hashes."""
import csv
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
REVISION = "80f3445"
BASE = f"https://raw.githubusercontent.com/ruc-datalab/DADER/{REVISION}/data"

def main():
    with (ROOT / "data" / "checksums.csv").open(newline="", encoding="utf-8") as f:
        entries = list(csv.DictReader(f))
    for entry in entries:
        name = entry["dataset"]
        data = urlopen(f"{BASE}/{name}/{name}.csv", timeout=60).read().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}: {digest}")
        target = ROOT / "data" / name / f"{name}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"{name}: verified")

if __name__ == "__main__":
    main()
