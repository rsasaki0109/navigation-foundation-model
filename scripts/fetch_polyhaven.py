"""Download the CC0 Poly Haven furniture models used by the RNE-rendered GIFs (1k textures).

Usage: python scripts/fetch_polyhaven.py   (writes assets/polyhaven/<asset>/...)
"""

import json
import sys
import urllib.request
from pathlib import Path

ASSETS = [
    "WoodenChair_01", "Sofa_01", "potted_plant_02", "GothicBed_01", "Television_01",
    "WoodenTable_01", "ClassicConsole_01", "coffee_table_round_01", "drawer_cabinet",
]
OUT = Path(__file__).resolve().parents[1] / "assets" / "polyhaven"
UA = {"User-Agent": "navfm-asset-fetch"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def main():
    for name in sys.argv[1:] or ASSETS:
        files = json.loads(get(f"https://api.polyhaven.com/files/{name}"))["gltf"]["1k"]["gltf"]
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.gltf").write_bytes(get(files["url"]))
        for rel, info in files["include"].items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(get(info["url"]))
        print(name, sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) // 1024, "KiB")


if __name__ == "__main__":
    main()
