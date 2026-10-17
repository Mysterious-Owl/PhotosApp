# FaceGroup

Local-first face grouping and search for your own photo library. Everything runs on your computer and the library is never modified. Personal, non-commercial use: the default face model (InsightFace `buffalo_l`) is licensed for non-commercial use only.

- Plan: [PLAN.md](PLAN.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Design review: [docs/PLAN_REVIEW.md](docs/PLAN_REVIEW.md)

**Status:** milestone M1 (core indexing, command line). Grouping and the web UI come in M2.

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

```powershell
python -m uv run facegroup scan "D:\Photos"   # add a folder and index it
python -m uv run facegroup scan               # rescan every folder: only new/changed files are processed
python -m uv run facegroup status             # counts, plus files that failed and why
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
