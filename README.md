# FaceGroup

Local-first face grouping and search for your own photo library. Everything runs on your computer and the library is never modified. Personal, non-commercial use: the default face model (InsightFace `buffalo_l`) is licensed for non-commercial use only.

- Plan: [PLAN.md](PLAN.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Design review: [docs/PLAN_REVIEW.md](docs/PLAN_REVIEW.md)

**Status:** M1 (indexing), M2 (People) and M3 (search) done. Text in images (OCR) comes in M4.

## Quick start

Double-click **`FaceGroup.cmd`**. The first time, it installs what it needs (a few minutes) and downloads the face model (~275 MB). Your browser then opens the People screen: add your photo folder, wait for the scan, then name the suggested groups.

## Setup (Windows)

```powershell
python -m pip install --user uv
# Keep the Python environment outside OneDrive (thousands of files should not sync):
setx UV_PROJECT_ENVIRONMENT "%LOCALAPPDATA%\FaceGroup\venv"
# open a new terminal, then in the project folder:
python -m uv sync
python -m uv run facegroup download-models   # ~275 MB, once
```

`uv` installs its own Python 3.12, so the system Python version does not matter. On Windows the DirectML build of ONNX Runtime is used, so any GPU (including Intel/AMD integrated graphics) speeds up face analysis. The CPU is used when no GPU is available.

## Use

**People screen** (`FaceGroup.cmd`, or `python -m uv run facegroup ui`):

- **Who is this?** shows groups of faces that look like one person. Type a name and Save, or open **Review** to untick faces that belong to someone else. Cartoons, statues and animals: **Not a person**.
- Once a person has a name, FaceGroup finds them in other photos by itself. Those faces get a dashed outline on the person's page until you confirm them (**Yes, it's them**) or remove them (**Not them**). A face you remove from a person is never put back automatically.
- **Merge** joins two people. **New person** splits selected faces off. **Undo** reverses the last change.
- **Forget this person** deletes their faces from FaceGroup for good, and later scans skip them. Photos are never touched.
- **Photos** (home): your timeline by month with a year/month jump list, **On this day** memories from earlier years, and screenshots/clutter hidden unless you ask. **Events** groups days and trips automatically. Any search can be **saved** and reopened from the search page.
- **Tidy** (top bar): screenshots, documents, receipts, memes and wallpapers are sorted out, so you can search "Photos only". It also finds copies of the same photo (suggesting which to keep) and folds bursts of similar shots together. Hide, favourite and rate photos. All of this stays inside FaceGroup; files are never changed.
- **Text in images:** words in screenshots, receipts and documents are searchable ("order shipped", an invoice number). Type people's names in search ("Mom and Dad beach"); if a name is also a normal word, use "search the words instead".
- **Check** (top bar): a few quick yes/no questions FaceGroup is least sure about ("Is this group Mom?", "Are Face 7 and Face 12 the same person?"). It measures how often its automatic matches are right, from your own answers and confirmed faces, and tunes its strictness to stay at 98% or better. It becomes more careful quickly, and looser only with plenty of evidence. Two faces in one photo are always treated as two different people.
- **Search** (box at the top): describe the photo ("beach at sunset", "birthday cake"), and narrow by person, date or folder. Words in file and folder names match too. On a photo, **Find similar photos**.
- The link the app opens only works on this computer, for this session.

**Command line:**

```powershell
python -m uv run facegroup scan "D:\Photos"   # add a folder and index it
python -m uv run facegroup scan               # rescan every folder: only new/changed files are processed
python -m uv run facegroup status             # counts, plus files that failed and why
python -m uv run facegroup people             # people and suggested groups, as text
python -m uv run facegroup retry              # re-queue failed files
```

- **Ctrl+C** stops after the current batch. Indexing also resumes safely after a crash or power loss.
- **OneDrive "online-only" files are skipped**, so a scan never downloads your cloud library. Use `--fetch-online-only` to include them.
- **Moved or renamed files** keep their faces and labels. **Deleted files** are kept as "missing" for 30 days, so a disconnected drive loses nothing.

## Data and settings

Everything lives in `%LOCALAPPDATA%\FaceGroup` (never in a synced folder):

| Path | Contents |
|---|---|
| `facegroup.db` | Library index, jobs, faces, embeddings (SQLite) |
| `cache\thumbs` | Thumbnails |
| `models\buffalo_l` | Face model files |
| `logs\facegroup.log` | Rotating log |
| `config.toml` | Optional settings (see `facegroup/config.py`) |

Example `config.toml`:

```toml
device = "cpu"          # auto | directml | cpu
min_face_px = 48        # ignore smaller faces
decode_workers = 4
```

Any setting can also be set as an environment variable, e.g. `FACEGROUP_DEVICE=cpu`.

## Development

```powershell
python -m uv run pytest                  # unit + pipeline tests (fake face model, ~10 s)
python -m uv run pytest -m integration   # real model on a real group photo (CPU and GPU)
python -m uv run python scripts/bench.py --count 10000 --kill-after 240   # acceptance benchmark
```

Code layout follows the architecture: `facegroup/domain` (pure rules), `ports` (interfaces), `services` (use cases), `adapters` (SQLite, decoders, models), `entrypoints` (CLI), wired together in `bootstrap.py`.
