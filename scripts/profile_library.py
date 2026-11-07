"""Profile FaceGroup on a real photo folder, read-only, with a throwaway database.

    uv run python scripts/profile_library.py "C:\\Users\\me\\OneDrive\\Pictures"

Reports speed per stage, file formats and sizes, failures and why, faces per photo, EXIF coverage, and
writes contact sheets of detected faces (lowest-confidence first) so false detections are easy to spot.
Everything is written to data\\eval\\real; the photo folder is only read.

The contact sheets show the user's own photos: they are for the user to look at. AI assistants must not
open them (see CLAUDE.md, Privacy).
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from facegroup.config import default_data_dir, default_eval_dir



def run_scan(folder: Path, data: Path) -> tuple[str, float]:
    env = {**os.environ, "FACEGROUP_DATA_DIR": str(data), "PYTHONIOENCODING": "utf-8",
           "FACEGROUP_KEEP_WORKING_IMAGES": "true"}
    models = default_data_dir() / "models"
    if not (data / "models").exists() and models.exists():
        data.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(data / "models"), str(models)], capture_output=True)
    t0 = time.monotonic()
    out = subprocess.run([sys.executable, "-m", "facegroup.entrypoints.cli", "scan", str(folder)],
                         env=env, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    return out, time.monotonic() - t0


def contact_sheet(rows, data: Path, out: Path, title_px: int = 18, cell: int = 128, cols: int = 10) -> None:
    """rows: (item_id, x1, y1, x2, y2, det_score, quality) from the working images."""
    cache = data / "cache" / "work"
    tiles = []
    for item_id, x1, y1, x2, y2, score, quality in rows:
        p = cache / f"{item_id // 1000:05d}" / f"{item_id}.jpg"
        img = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR) if p.exists() else None
        if img is None:
            continue
        w, h = x2 - x1, y2 - y1
        pad = 0.25
        crop = img[max(0, int(y1 - h * pad)):int(y2 + h * pad), max(0, int(x1 - w * pad)):int(x2 + w * pad)]
        if crop.size == 0:
            continue
        tile = np.full((cell + title_px, cell, 3), 255, np.uint8)
        s = cell / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
        tile[title_px:title_px + crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(tile, f"{score:.2f} q{quality:.2f} {int(min(w, h))}px", (2, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    out.write_bytes(buf.tobytes())


def thumbnail_sheet(media_ids, data: Path, out: Path, cell: int = 160, cols: int = 8) -> None:
    """Whole-photo thumbnails, e.g. photos where no face was found, to spot missed faces."""
    tiles = []
    for mid in media_ids:
        p = data / "cache" / "thumbs" / f"{mid // 1000:05d}" / f"{mid}.jpg"
        img = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR) if p.exists() else None
        if img is None:
            continue
        tile = np.full((cell, cell, 3), 255, np.uint8)
        s = cell / max(img.shape[:2])
        img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
        tile[:img.shape[0], :img.shape[1]] = img
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out.write_bytes(cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=default_eval_dir() / "real")
    args = ap.parse_args()
    data = args.out / "data"
    if data.exists():
        for name in ("facegroup.db", "facegroup.db-wal", "facegroup.db-shm"):
            (data / name).unlink(missing_ok=True)
        shutil.rmtree(data / "cache", ignore_errors=True)

    out, seconds = run_scan(args.folder, data)
    print(out)
    stage_lines = [m.strip() for m in re.findall(r"^\s+(?:derive|faces): .*$", out, re.M)]

    db = sqlite3.connect(data / "facegroup.db")
    def q(sql, *a):
        return db.execute(sql, a).fetchall()
    media = q("SELECT m.rel_path, m.size, m.width, m.height, m.taken_at, m.gps_lat FROM media m")
    exts = collections.Counter(Path(r[0]).suffix.lower() for r in media)
    mp = [r[2] * r[3] / 1e6 for r in media if r[2]]
    per_photo = collections.Counter(n for (n,) in q(
        "SELECT COUNT(f.id) FROM item i LEFT JOIN face f ON f.item_id=i.id GROUP BY i.id"))
    faces = q("SELECT item_id, x1, y1, x2, y2, det_score, quality FROM face ORDER BY det_score")
    dead = q("SELECT m.rel_path, j.stage, j.error FROM job j JOIN media m ON m.id=j.media_id WHERE j.state='dead'")

    report = [
        f"# Real-library profile: {args.folder}",
        "",
        f"- Files: {len(media)} ({', '.join(f'{k} {v}' for k, v in exts.most_common())})",
        f"- Total time: {seconds:.1f}s ({len(media) / max(seconds, 1e-9):.1f} files/s end to end, incl. model load)",
        f"- Stages: {'; '.join(stage_lines) or 'n/a'}",
        f"- Image size: median {np.median(mp):.1f} MP, max {max(mp):.1f} MP" if mp else "- Image size: n/a",
        f"- EXIF date: {sum(1 for r in media if r[4])}/{len(media)}; GPS: {sum(1 for r in media if r[5] is not None)}/{len(media)}",
        f"- Faces: {len(faces)} total; photos by face count: "
        + ", ".join(f"{k} faces: {v}" for k, v in sorted(per_photo.items())),
        f"- Failed files: {len(dead)}",
    ]
    report += [f"  - {p} [{s}]: {e}" for p, s, e in dead[:30]]
    sheets = []
    if faces:
        contact_sheet(faces[:60], data, args.out / "faces_lowest_confidence.jpg")
        contact_sheet(faces[-30:], data, args.out / "faces_highest_confidence.jpg")
        small = sorted(faces, key=lambda r: min(r[3] - r[1], r[4] - r[2]))[:30]
        contact_sheet(small, data, args.out / "faces_smallest.jpg")
        sheets = ["faces_lowest_confidence.jpg", "faces_highest_confidence.jpg", "faces_smallest.jpg"]
    no_face = [r[0] for r in q(
        "SELECT i.media_id FROM item i LEFT JOIN face f ON f.item_id=i.id GROUP BY i.id HAVING COUNT(f.id)=0 LIMIT 48")]
    if no_face:
        thumbnail_sheet(no_face, data, args.out / "photos_without_faces.jpg")
        sheets.append("photos_without_faces.jpg")
    report.append(f"- Contact sheets in {args.out}: {', '.join(sheets)}")
    text = "\n".join(report)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
