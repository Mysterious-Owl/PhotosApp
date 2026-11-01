# FaceGroup

Local-first face grouping and search for your own photo library, plus file tools (duplicate files, HEIC conversion, metadata removal). Everything runs on your computer. Personal, non-commercial use: the default face model (InsightFace `buffalo_l`) is licensed for non-commercial use only.

- Plan: [PLAN.md](PLAN.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Design review: [docs/PLAN_REVIEW.md](docs/PLAN_REVIEW.md)
- Accuracy: [docs/EVALUATION.md](docs/EVALUATION.md)
- Similar apps and feature ideas: [docs/FEATURE_RESEARCH.md](docs/FEATURE_RESEARCH.md)

**Status:** indexing, People, search, text in images, Tidy (categories, duplicates, bursts), browsing (timeline, On this day, Events) and the file tools (Duplicate files, Convert HEIC, Remove metadata) are done. Video and captions are not built yet.

Your photos are never changed, with one exception that you start yourself: **Duplicate files** moves extra copies into a holding folder when you confirm, and can put them back.

## Quick start

Double-click **`FaceGroup.cmd`**. The first time, it installs what it needs (a few minutes) and downloads the face model (~275 MB). Your browser then opens the People screen: add your photo folder, wait for the scan, then name the suggested groups.

## Setup (Windows)

```powershell
python -m pip install --user uv
# in the project folder (keep it outside OneDrive; the environment goes in .venv):
python -m uv sync
python -m uv run facegroup download-models   # ~275 MB, once
```

`uv` installs its own Python 3.12, so the system Python version does not matter. On Windows the DirectML build of ONNX Runtime is used, so any GPU (including Intel/AMD integrated graphics) speeds up face analysis. The CPU is used when no GPU is available.

## Use

**People screen** (`FaceGroup.cmd`, or `python -m uv run facegroup ui`):

- **Who is this?** shows groups of faces that look like one person. Type a name and Save, or open **Review** to untick faces that belong to someone else. Cartoons, statues and animals: **Not a person**.
- Once a person has a name, FaceGroup finds them in other photos by itself. Those faces get a dashed outline on the person's page until you confirm them (**Yes, it's them**) or remove them (**Not them**). A face you remove from a person is never put back automatically.
- **Merge** joins two people. **New person** splits selected faces off. **Undo** reverses the last change.
- **Forget this person** deletes the faces you confirmed as them from FaceGroup for good, and later scans skip those. Faces FaceGroup matched to them automatically are not deleted (they might be someone else); they go back to unsorted. Photos are never touched.
- **Photos** (home): your timeline by month with a year/month jump list, **On this day** memories from earlier years, and screenshots/clutter hidden unless you ask. **Events** groups days and trips automatically. Any search can be **saved** and reopened from the search page.
- **Tidy** (top bar): screenshots, documents, receipts, memes and wallpapers are sorted out, so you can search "Photos only". It also finds copies of the same photo (suggesting which to keep) and folds bursts of similar shots together. Hide, favourite and rate photos. All of this stays inside FaceGroup; files are never changed.
- **Text in images:** words in screenshots, receipts and documents are searchable ("order shipped", an invoice number). Type people's names in search ("Mom and Dad beach"); if a name is also a normal word, use "search the words instead".
- **Check** (top bar): a few quick yes/no questions FaceGroup is least sure about ("Is this group Mom?", "Are Face 7 and Face 12 the same person?"). It measures how often its automatic matches are right, from your own answers and confirmed faces, and tunes its strictness to stay at 98% or better. It becomes more careful quickly, and looser only with plenty of evidence. Two faces in one photo are always treated as two different people.
- **Search** (box at the top): describe the photo ("beach at sunset", "birthday cake"), and narrow by person, date or folder. Words in file and folder names match too. On a photo, **Find similar photos**.
- The link the app opens only works on this computer, for this session. FaceGroup opens in its own Edge window when Edge is installed. Theme (System, Light, Dark) is at the bottom of the sidebar.

**File tools** (any folder or drive, not only the library). Every folder box has **Browse…**, which opens the Windows folder picker; you can also type or paste a path.

- **Duplicate files** (Clean up): finds byte-identical files (photos and videos, or all files) across folders and drives. Under **Folder pairs**, choose which folder's copies to keep; that choice applies to every duplicate between the two folders. Copies inside one folder keep the plain name over "IMG_1 (1).jpg", and PNG over HEIC. A HEIC photo that also has a PNG made from it (same name, checked to be the same picture) shows as **HEIC + PNG**: the PNG is kept, and the HEIC moves only if you tick that option. Anything else is only a suggestion until you approve it. **Move** puts extra copies in `FaceGroup removed` at the top of the same drive, keeping their folder structure; **Moved** has **Restore**. Nothing is deleted, and you empty `FaceGroup removed` yourself. It also lists empty folders. OneDrive cloud-only files are never downloaded.
- **Convert HEIC** (Tools): iPhone HEIC/HEIF photos to PNG or JPEG copies in "`<folder> converted`", keeping the date, camera and location. It can also copy everything else, so the new folder is complete.
- **Remove metadata** (Tools): **Check only** counts what the pictures reveal (location, camera, date). **Make clean copies** writes them to "`<folder> clean`" without EXIF, GPS, XMP, IPTC or comments, turned upright first. It keeps the colour profile, and the date taken if you tick it.
- **Right-click menu:** `python -m uv run facegroup context-menu install` adds a FaceGroup submenu to folders and drives in Explorer (current user only; on Windows 11 it is under **Show more options**). It opens the tool with that folder filled in, inside the running FaceGroup if it is already open. Remove it with `facegroup context-menu remove`.

Convert HEIC and Remove metadata replace the old `Apple image convertor` and `photo metadata clean` scripts, and Duplicate files replaces `photo dedup`.

**Command line:**

```powershell
python -m uv run facegroup scan "D:\Photos"   # add a folder and index it
python -m uv run facegroup scan               # rescan every folder: only new/changed files are processed
python -m uv run facegroup status             # counts, plus files that failed and why
python -m uv run facegroup people             # people and suggested groups, as text
python -m uv run facegroup retry              # re-queue failed files
python -m uv run facegroup convert "D:\iPhone" --jpg --all   # HEIC to JPEG copies (works while the app is open)
python -m uv run facegroup metadata "D:\To share"            # report; add --clean for clean copies
python -m uv run facegroup ui --page duplicates --folder "D:\"   # open a tool with a folder filled in
```

- **Ctrl+C** stops after the current batch. Indexing also resumes safely after a crash or power loss.
- **OneDrive "online-only" files are skipped**, so a scan never downloads your cloud library. Use `--fetch-online-only` to include them.
- **Moved or renamed files** keep their faces and labels. **Deleted files** are kept as "missing" for 30 days, so a disconnected drive loses nothing.

## Data and settings

Everything lives in `data\app` in the project folder (git-ignored, and refused if inside OneDrive). Test data goes in `data\eval` and `data\bench`. Set `FACEGROUP_DATA_DIR` to use another folder.

| Path | Contents |
|---|---|
| `facegroup.db` | Library index, jobs, faces, embeddings (SQLite) |
| `cache\thumbs`, `cache\faces` | Thumbnails and face crops |
| `models\buffalo_l`, `models\siglip2-base` | Face model and search model files |
| `logs\facegroup.log` | Rotating log |
| `config.toml` | Optional settings (see `facegroup/config.py`) |
| `session.json` | Port and login token of the running app, so a second launch (e.g. the right-click menu) opens there; removed on exit |

Example `config.toml`:

```toml
device = "cpu"          # auto | directml | cpu
min_face_px = 48        # ignore smaller faces
decode_workers = 4
```

Any setting can also be set as an environment variable, e.g. `FACEGROUP_DEVICE=cpu`.

**Back up `data\app` before cleaning the repo.** It holds your names, groups and labels, and `git clean -x` (or deleting the project folder) removes it. Your photos themselves are never stored there.

## Development

```powershell
python -m uv run pytest                  # unit + pipeline tests (fake face model, ~1 min)
python -m uv run pytest -m integration   # real model on a real group photo (CPU and GPU)
python -m uv run python scripts/bench.py --count 10000 --kill-after 240   # acceptance benchmark
```

Code layout follows the architecture: `facegroup/domain` (pure rules), `ports` (interfaces), `services` (use cases), `adapters` (SQLite, decoders, models), `entrypoints` (CLI and web UI), wired together in `bootstrap.py`. See [docs/ARCHITECTURE.md §7](docs/ARCHITECTURE.md#7-code-layout).
