"""Build an immutable supervised GSV-Cities index after archive validation."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path("/home/quyet/k210_lab")
GSV = ROOT / "datasets" / "gsv_cities"
OUT = ROOT / "datasets" / "success_first"
REPORT = ROOT / "reports" / "success_first"
DEV_CITIES = {"Boston", "Brussels", "Osaka"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def image_name(row) -> str:
    city = str(row.city_id)
    place = str(int(row.place_id)).zfill(7)
    year = str(int(row.year)).zfill(4)
    month = str(int(row.month)).zfill(2)
    north = str(int(row.northdeg)).zfill(3)
    return f"{city}_{place}_{year}_{month}_{north}_{row.lat}_{row.lon}_{row.panoid}.jpg"


def main() -> None:
    dataframe_dir = GSV / "Dataframes"
    image_root = GSV / "Images"
    csvs = sorted(dataframe_dir.glob("*.csv"))
    if not csvs or not image_root.is_dir():
        raise RuntimeError(f"GSV layout incomplete under {GSV}")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "gsv_cities.sqlite"
    temporary = destination.with_suffix(".sqlite.tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.executescript("""
      PRAGMA journal_mode=OFF;
      PRAGMA synchronous=OFF;
      CREATE TABLE images (
        image_id TEXT PRIMARY KEY, place_id TEXT NOT NULL, city TEXT NOT NULL,
        path TEXT NOT NULL, split TEXT NOT NULL, year INTEGER, month INTEGER,
        northdeg INTEGER, lat REAL, lon REAL
      );
      CREATE INDEX images_place ON images(place_id);
      CREATE INDEX images_split_place ON images(split, place_id);
    """)
    missing = []
    city_stats = {}
    for csv_path in csvs:
        city = csv_path.stem
        frame = pd.read_csv(csv_path)
        required = {"place_id", "city_id", "panoid", "year", "month", "northdeg", "lat", "lon"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"{csv_path} missing columns {sorted(required - set(frame.columns))}")
        split = "dev" if city in DEV_CITIES else "train"
        places = set()
        rows = []
        for row in frame.itertuples(index=False):
            name = image_name(row)
            path = image_root / str(row.city_id) / name
            if not path.is_file():
                if len(missing) < 20:
                    missing.append(str(path))
                continue
            place_id = f"gsv:{city}:{int(row.place_id):07d}"
            places.add(place_id)
            rows.append((f"gsv:{city}:{name}", place_id, city, str(path), split,
                         int(row.year), int(row.month), int(row.northdeg), float(row.lat), float(row.lon)))
        connection.executemany("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        city_stats[city] = {"split": split, "images": len(rows), "places": len(places),
                            "csv_sha256": sha256(csv_path)}
    if missing:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"GSV extracted files missing; examples: {missing[:3]}")
    # Only places with at least four real views are admitted.
    connection.execute("DELETE FROM images WHERE place_id IN (SELECT place_id FROM images GROUP BY place_id HAVING COUNT(*) < 4)")
    connection.commit()
    stats = {}
    for split in ("train", "dev"):
        images = connection.execute("SELECT COUNT(*) FROM images WHERE split=?", (split,)).fetchone()[0]
        places = connection.execute("SELECT COUNT(DISTINCT place_id) FROM images WHERE split=?", (split,)).fetchone()[0]
        stats[split] = {"images": images, "places": places}
    connection.close()
    temporary.replace(destination)
    manifest = {
        "schema_version": 1, "dataset": "GSV-Cities", "role": "large_scale_vpr_pretraining",
        "license": "CC BY-NC-ND 4.0", "root": str(GSV), "index": str(destination),
        "index_sha256": sha256(destination), "dev_cities": sorted(DEV_CITIES),
        "split_policy": "entire cities held out for DEV; no place crosses train/dev",
        "stats": stats, "cities": city_stats,
    }
    atomic_json(REPORT / "gsv_cities_manifest.json", manifest)
    print(json.dumps({"stats": stats, "index_sha256": manifest["index_sha256"]}))


if __name__ == "__main__":
    main()
