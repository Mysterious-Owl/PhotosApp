# FaceGroup — Application Plan

As of 2026-10-17 · Author: Tanishq Gupta · Status: draft v4 (design review applied; see [docs/PLAN_REVIEW.md](docs/PLAN_REVIEW.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))

FaceGroup is a local-first app for **personal, non-commercial use** on the user's own photo library. It scans folders of photos and videos, finds faces and groups them into people, and builds a search index of people, objects, scenes, visible text, dates and places. By default everything runs on the user's own computer. Cloud models can be added as optional plug-ins but are never required.

---

## 1. Overview

**Goals**

- Point the app at one or more folders. Indexing is incremental: only new or changed files are processed again.
- Find faces and group them into people. The user can name, merge and split groups.
- Search by person, plain-language description, visible text, date, place, folder, or a reference image.
- Run on an ordinary laptop without a dedicated graphics card. The reference machine is Windows 11 with 16 GB RAM, a regular processor and built-in graphics. Use a graphics card when one is present.
- Prefer open models. Allow free-to-download models with non-commercial licences (e.g. InsightFace), because use is personal. If the code is ever published, users download such models themselves; they are never bundled.

**Not in v1**

- Cloud sync, multiple user accounts, mobile apps.
- Identifying people the user has not labelled: no public face databases, no web lookup.
- Changing media files. The app only reads them.

---

## 2. Review of the existing prototype

The prototype in `archive/Local_Face_Finder_Windows11/local_face_finder` is a single 400-line `app.py`. It groups faces from one folder and it works. It uses a Tkinter desktop window, InsightFace `buffalo_l` running on the processor through ONNX Runtime, and a SQLite database in `%LOCALAPPDATA%\LocalFaceFinder`. It is a sound first version to learn from, but several bugs would show up on a real photo library.

### What it gets right (keep these ideas)

- It never changes originals: no moving, renaming or editing.
- Scans are incremental: a file is skipped when its modification time and size are unchanged.
- Manual work survives a re-scan: only faces not yet assigned to a person are re-grouped.
- The interface is honest about uncertainty. It calls groups "suggestions" and uses a cautious match threshold (cosine similarity of 0.48).
- It has no network code, so privacy is the default. The README states the intended use: your own library, not identifying strangers.

### Bugs and risks, most serious first

| # | Issue | Where | Effect | Fix in FaceGroup |
|---|---|---|---|---|
| 1 | Hard dependency on the `insightface` package, which needs C++ build tools on Windows (see #8). The `buffalo_l` weights are non-commercial, which is fine for personal use | `load_engine()` | Hard install; model hard-wired | Load the buffalo_l ONNX files directly with `onnxruntime` behind a `FaceAnalyzer` port, so the model is swappable |
| 2 | `cv2.imread` cannot open files whose path has non-English characters on Windows | `_scan_worker` | Files in folders with such names (common with OneDrive, or Hindi or accented names) are silently saved as having 0 faces | Read the bytes with `np.fromfile` and decode with `cv2.imdecode`, or decode with Pillow |
| 3 | Grouping is greedy, using a running average for each group | `group_unassigned()` | Results depend on the order files are read. Group averages drift until look-alikes merge. Every face seen once becomes its own "person", flooding the list | Use a clustering method that needs no group count (HDBSCAN or Chinese Whispers) with a minimum group size. Faces seen only once go into an "Unsorted" bin |
| 4 | No check on face quality | scan loop | Tiny, blurry and background faces pollute the groups | Skip faces below a minimum size and detection confidence, and store a quality score |
| 5 | The gallery decodes full-size images on the interface thread | `show_paths()` | The window freezes for seconds to minutes with thousands of photos | Cache thumbnails on disk and load the grid lazily as the user scrolls |
| 6 | No way to fix mistakes | UI | You cannot merge two groups, split one, or move a single face. The gallery shows whole photos, not face close-ups | A face close-up review screen with merge, split, move and "not this person" |
| 7 | Face fingerprints (embeddings) are stored as comma-separated text | `save_photo()` | About 4x larger than binary and slow to read back. Nothing indexes them for fast search | Store them as binary `float32` and add a vector index |
| 8 | `install.bat` requires Python 3.11 but this machine has 3.10. `insightface` has no ready-made Windows packages and needs Microsoft's C++ build tools | install scripts | Installation fails for typical users | Ship a pinned set of ready-made packages, or a bundled Python |
| 9 | Photos only, one folder, no capture dates or GPS from photo metadata, no video | scope | Covers only the face half of the goal | Multiple library folders, photo metadata (EXIF), sampled video frames |
| 10 | No LICENSE file, no tests, and no record of which model produced each fingerprint | repo | Fingerprints from different model versions could be silently mixed | Store a `model_id` with every fingerprint, and add automated tests (CI) |

### What it changes in the plan

The prototype shows a desktop-native app is practical, which reopens the interface question (§9). Its single file mixes UI, storage, model calls and clustering. FaceGroup separates these into layers (§3).

---

## 3. Architecture

The full architecture, with 11 reviewed diagrams, is in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**. It covers context, layers, the pipeline, job and file lifecycles, the data model, scan, grouping, search and runtime. In summary:

- **One local process, four layers:** interfaces (UI, API, CLI) → application services (Library, Indexing, People, Search) → pure domain logic → ports with swappable adapters (models, SQLite, sqlite-vec, decoders). Dependencies point inward only.
- **One SQLite file in `%LOCALAPPDATA%\FaceGroup`**, never inside OneDrive. It holds metadata, jobs, FTS5 and sqlite-vec indexes. Embedding BLOBs are the canonical copy, and the vec0 tables are a derived index that can be rebuilt.
- **Stage-by-stage pipeline:** discover → probe → derive (decode once, EXIF-rotate, sample video frames) → faces / image embedding / OCR. One heavy model is loaded at a time. Jobs are idempotent, leased and retried.
- **Every image is an "item":** a photo is one item, a video gives one item per sampled frame. Faces, vectors and text all point at items.
- **Labels sit on face detections, not embeddings**, so changing the face model never loses names or merges.
- **One DB writer thread**, WAL mode, read-only connections for the API.
- **Grouping:** k-NN vote over faces *you* confirmed, so auto-assigned faces never vote. Unassigned faces are clustered into *suggestions*, and a person is created only when you accept. Your decisions are never overwritten.
- **Search filters first, then ranks:** person, date and folder chips narrow the candidates, then vector + FTS5 results are fused with reciprocal rank fusion.
- **Local API is locked down:** 127.0.0.1 only, Host-header allowlist, per-launch token, media served by id only.

---

## How models are compared (sections 4–6)

Every option is judged against the same criteria, in this order:

1. **Licence allows personal use** (all models listed here do, except where marked).
2. **Measured accuracy** on public benchmarks, and later on our own varied evaluation set (§8).
3. **Speed and RAM on a processor without a graphics card,** plus whether an ONNX version exists.
4. **Ecosystem:** maintenance, documentation, other projects using it.

The "OK to bundle?" columns only matter if the code is ever published (§7).

Who makes a model plays no part in choosing it. Licence facts were checked against the official model cards and repositories on 2026-10-17 (links in §10). "Unverified" means the source page did not say.

---

## 4. Face detection and recognition models

### Face detectors

| Model | Weights licence | OK to bundle in OSI project? | Size / speed | Strengths | Weaknesses |
|---|---|---|---|---|---|
| **YuNet** (OpenCV Zoo) | MIT | Yes | Very small; real-time on a processor. New `2026may` export accepts any image size | Tiny and fast; built into OpenCV (`cv2.FaceDetectorYN`); has an int8 version | Less accurate than SCRFD on small faces and side profiles |
| **SCRFD** (InsightFace packs) | Non-commercial research only | **No** (user must download and accept terms) | 10GF version is part of the 326 MB buffalo_l pack | Very accurate, including small faces | Licence; packaged together with a non-commercial recogniser |
| dlib HOG / CNN | CC0 (models), Boost (code) | Yes | HOG is fast but weak; CNN is slow without a graphics card | Mature, public domain | Lower accuracy; build trouble on Windows; the 68-point landmark model is **non-commercial** (use the 5-point one) |

### Face recognisers (produce the face fingerprints)

| Model | Weights licence | OK to bundle? | Architecture / training data | Reported accuracy | Notes |
|---|---|---|---|---|---|
| **SFace** (OpenCV Zoo) | Apache-2.0 | Yes | MobileFaceNet; training data not stated | about 99.4% (dataset not named on card) | Very light; fp32 and int8 ONNX. **PhotoPrism's default for new libraries since 2026** |
| **AuraFace v1** (fal) | Apache-2.0 | Yes | ResNet100 trained with ArcFace loss on "commercially available data" | LFW 99.65%, CPLFW 90.9% | Same file format as InsightFace models; needs a separate detector. Heavier than SFace |
| dlib ResNet v1 | CC0 | Yes | about 3M faces, 7,485 identities (FaceScrub, VGG, web) | LFW 99.38% | The `face_recognition` README itself warns of weaker results on children and differences across ethnicities |
| InsightFace buffalo_l / antelopev2 | **Non-commercial research only** (commercial licence available on request since Nov 2025) | **No** | ResNet50 on WebFace600K / ResNet100 on Glint360K | Among the most accurate open models | Used by Immich (AGPL). Fine as an **opt-in plug-in the user downloads**, never bundled |
| EdgeFace (Idiap) | Code BSD-3, **weights CC BY-NC-SA** | No | 1.2M–18M parameters | Strong for its size | Non-commercial weights; PyTorch only |
| facenet-pytorch (VGGFace2) | Code MIT, **weights licence unstated** | Unclear, so avoid | Inception-ResNet v1 | — | Legally unclear |

**Suggested default (personal use):** **InsightFace `buffalo_l`** (SCRFD detection + ArcFace ResNet50 recognition). Its non-commercial licence allows personal use, and it is the most accurate option here. **AuraFace** (Apache-2.0) is the fully open alternative behind the same interface, and **YuNet + SFace** is the low-power option. Before fixing the default, run buffalo_l against AuraFace on our evaluation set (§8). Don't rely only on LFW scores, which are near their ceiling and say little about children, old photos or side profiles.

---

## 5. Vision models

These are two separate jobs. **Image fingerprints** power search by description and search by example photo, and run on every image. **Vision-language models (VLMs)** turn images into words (captions, tags), are optional and are slow.

### 5a. Image fingerprint models (text and images in one shared space)

| Model | Licence | OK to bundle? | Size | What it handles | Strengths | Weaknesses |
|---|---|---|---|---|---|---|
| **EmbeddingGemma 2** (Google, released Oct 2026) | Apache-2.0 | Yes | 740M (270M text + 170M vision + 300M audio) | Text, image, **video, audio** in one 768-dimension space; can be cut to 512/256/128; 8K context | **One model for images, videos, captions and text search.** Matches the "Gemma embedding 2" request. Permissive licence | New, so little independent benchmarking yet; no official ONNX (community quantised versions exist); must use float32 on processors (no float16); heaviest option here |
| **SigLIP 2** (Google) | Apache-2.0 | Yes | base 0.4B, so400m 1B | Image and text | Well benchmarked; offered in Immich; flexible-resolution versions | Image and text only; text side weak on long queries |
| **OpenCLIP** (LAION / DataComp) | MIT-tagged weights | Yes (licence) | ViT-B/32 is about 0.15B | Image and text | Smallest and fastest; many sizes; mature ONNX support | LAION model cards advise against "deployed use" (a recommendation, not a licence term); training data uncurated |
| **Nomic Embed Vision v1.5** | Apache-2.0 | Yes | 93M | Image, in the same space as nomic-embed-text v1.5 | Very small; text side handles long text well | Needs custom code to load; less tested for searching photos |
| MobileCLIP / MobileCLIP2 (Apple) | **Research only** | No | 11M–320M (image side) | Image and text | Very fast | Licence excludes product use |
| Jina CLIP v2 | **CC BY-NC 4.0** | No | about 0.9B | Image and text, 89 languages | Multilingual | Non-commercial |
| Gemini Embedding 2 (Google API) | Paid API, no files to download | n/a | — | Text, image, audio, video, PDF | Strong and multimodal, nothing to install | **Sends media to Google**; about $0.00012 per image; not local-first. Optional cloud plug-in only |

> Correction to the earlier chat plan: "Gemma embedding 2" turns out to be **EmbeddingGemma 2**, an open-weights, Apache-2.0, multimodal model that runs locally. It is a different thing from the Gemini Embedding 2 API. The original EmbeddingGemma (300M, text only) is under the non-OSI Gemma Terms.

**Suggested approach:** test **EmbeddingGemma 2** against **SigLIP 2 (base)** on our own photos in M3, measuring search quality (recall@10 on about 200 labelled queries) and images per second on the reference laptop. If EmbeddingGemma 2 is close in quality and fast enough on the processor, it replaces both the image model *and* the separate text model, because captions, OCR text and queries all share one space. That makes the design simpler. Keep **OpenCLIP ViT-B/32** as a "low-power" option for weak machines.

### 5b. Vision-language models (captions, tags, describing scenes)

| Model | Licence | OK to bundle? | Size | Feasible on a processor? | Strengths | Weaknesses |
|---|---|---|---|---|---|---|
| **Florence-2** base / large | MIT | Yes | 0.23B / 0.77B | Yes, about 1 s per image (estimate) | Captions + OCR + object detection in one small model | Needs custom loading code; no official ONNX; no updates since 2024 |
| **SmolVLM2** | Apache-2.0 | Yes | 256M / 500M / 2.2B | Yes (256M–500M) | Very small; actively maintained | Weaker detail at small sizes |
| **Qwen3-VL** 2B / 4B | Apache-2.0 | Yes | 2B / 4B | Slow but usable when quantised (llama.cpp / Ollama) | Strong captions, OCR, multilingual | Several seconds per image on a processor |
| **Qwen3.5** small (0.8B / 2B) | Apache-2.0 | Yes | 0.8B / 2B | Yes when quantised | Newer (2026), multimodal from the start | Little independent captioning data yet |
| **Gemma 4** E2B | Apache-2.0 | Yes | 2.3B effective (5.1B total) | Yes when quantised | Text + image + audio; permissive (unlike Gemma 3) | New |
| **InternVL3.5** 1B / 2B | Apache-2.0 | Yes | 1.1B / 2.3B | Yes when quantised | Strong scores for its size | Licence was inconsistent on InternVL3 pages; 3.5 is clear |
| Moondream 2 | Apache-2.0 | Yes | 2B | Yes | Tuned for describing and asking about images | Last update Jun 2025 |
| Moondream 3.x | Moondream License (bans hosted services) | No (not OSI) | 9B mixture-of-experts | Targets graphics cards | Strong | Licence; hardware |
| Phi-4-multimodal | MIT | Yes | 5.6B | Heavy | Official ONNX | English-only vision; too large for most laptops |
| Gemma 3 / 3n | Gemma Terms (not OSI) | Avoid | — | — | — | Use Gemma 4 instead |
| Qwen2.5-VL-3B | **Research only** | No | 3B | — | — | Non-commercial (the 7B is Apache, the 3B is not) |

**Suggested approach:** captions are **deferred from v1** and added later as an optional plug-in (one port, one pipeline stage). Default to **Florence-2 base** (smallest, does captioning and OCR in one model) or **SmolVLM2-500M** for laptops. Offer **Qwen3-VL / Qwen3.5 / Gemma 4 / InternVL3.5** through Ollama or llama.cpp for users with more RAM or a graphics card. Measure caption quality and speed before fixing the default. The plug-in design means changing it later is cheap.

### 5c. OCR (text inside images)

| Tool | Licence | Notes |
|---|---|---|
| **RapidOCR** | Apache-2.0 | PaddleOCR models converted to ONNX; easy to install with pip; fast on a processor |
| **PaddleOCR** (PP-OCRv6) | Apache-2.0 | Newest models (tiny 1.5M to medium 34.5M); larger dependency |
| Tesseract 5 | Apache-2.0 | Mature; weaker on photos of real-world scenes than on scanned documents |

Suggested: RapidOCR, or skip it if the chosen VLM already does OCR well.

---

## 6. Text embeddings, vector store and search

### Text fingerprints (needed only if the image model has no strong text side)

| Model | Licence | Size | Notes |
|---|---|---|---|
| **EmbeddingGemma 2** | Apache-2.0 | 740M | Same space as images (§5a), so no separate text model needed |
| **Qwen3-Embedding-0.6B** | Apache-2.0 | 0.6B | Strong multilingual model; 32K context; no official ONNX |
| **nomic-embed-text v1.5** | Apache-2.0 | about 0.1B | Small, ONNX available; same space as Nomic Embed Vision |
| **BGE-M3** | MIT | about 0.6B (unverified) | Dense + sparse + ColBERT retrieval; multilingual |
| **multilingual-e5-small** | MIT | small | Very light; 512-token limit |
| EmbeddingGemma (v1, 300M) | Gemma Terms (not OSI) | 308M | Superseded by v2 for this project |

### Vector store

| Option | Licence | Mode | Fits when | Trade-off |
|---|---|---|---|---|
| **sqlite-vec** | MIT / Apache-2.0 | Extension inside SQLite | Up to around 100k–200k items (exact search) | Everything in one file; **still pre-v1**, so breaking changes are possible |
| **LanceDB** | Apache-2.0 | Embedded library | Large libraries (1M+ fingerprints) | Separate data files next to SQLite |
| **FAISS** | MIT | Library in memory | Fastest raw search | You manage saving to disk and syncing with SQLite yourself |
| Qdrant (local mode) | Apache-2.0 | Embedded local mode is meant for development | Shipping a server later | Heavy for a desktop app |

Suggested: hide the store behind a `VectorStore` interface. Start on **sqlite-vec**, and move to LanceDB if the library-size decision (§9) calls for it.

### How a search runs

1. **Read the query:** person, date range and folder come from explicit filter chips in the UI. Free-text date parsing ("last summer") is left out of v1 as fragile. The text box is the description.
2. **Gather candidates:** the fingerprint search (description vs. image fingerprints, plus caption/OCR fingerprints) and the FTS5 keyword search (captions, OCR, file paths) run side by side.
3. **Merge:** filters are applied *before* retrieval (filter first, then rank). The vector and keyword result lists are then combined with reciprocal rank fusion, and video frames are collapsed to one score per video.
4. **Search by example:** an uploaded photo is turned into an image fingerprint for the image search, and into a face fingerprint ("photos of this person") when it contains a face.

---

## 7. Licensing (personal use)

**Scope is personal, non-commercial use, so non-commercial model licences (InsightFace, EdgeFace, MobileCLIP, Jina CLIP v2, Qwen2.5-VL-3B) are acceptable.** The rules below only matter if the code is ever published:

1. **If published, bundle only models whose licence is OSI-compatible** (Apache-2.0, MIT, BSD, CC0). As of 2026-10-17 that includes: YuNet, SFace, AuraFace, dlib ResNet (CC0, 5-point landmarks only), EmbeddingGemma 2, SigLIP 2, OpenCLIP, Nomic, Florence-2, SmolVLM2, Qwen3-VL, Qwen3.5, Gemma 4, InternVL3.5, Moondream 2, RapidOCR/PaddleOCR, Tesseract.
2. **If published, non-commercial models become downloads the user agrees to.** The user downloads them and the app shows their licence first. Examples: InsightFace packs, EdgeFace, MobileCLIP, Jina CLIP v2, Qwen2.5-VL-3B, Moondream 3.x, any Gemma-Terms model.
3. **Keep a `MODELS.md` register** listing each model's licence, source URL and SHA-256 checksum. The app verifies the checksum after download.
4. **Record `model_id` with every stored fingerprint** so changing model is a clean re-index.
5. **Prototype code:** the existing `app.py` has no licence file. Confirm who wrote it and that it can be released before reusing any of it.
6. **Project licence options:** MIT (simplest), Apache-2.0 (adds a patent grant; matches most of the model licences above), AGPL-3.0 (keeps hosted forks open; used by Immich). Decision in §9.

---

## 8. Fairness, bias and privacy

**Bias in face recognition is measurable and matters here.** NIST's FRVT/FRTE tests show that error rates of face recognition models differ by age, sex and skin tone. The size of the gap depends on the model and its training data. For a photo organiser, the effect is incorrect merges, such as relatives or children merged together, and incorrect splits, such as one person split across age or lighting. These errors are not spread evenly across groups of people.

How to reduce it:

- **No single threshold assumed fair.** Start from the model's recommended value. Give a slider with a live preview of how many faces it would assign. Add per-library automatic tuning only if the evaluation shows it is needed.
- **Children and ageing.** Fingerprints of children change quickly. Assignment uses a k-NN vote over all faces you confirmed, so a person has many reference faces across ages, not one average.
- **The user has the final say.** Automatic results are labelled as suggestions. Merge, split and "not this person" are always one click away and permanent.
- **Measure, don't assume.** Keep a small labelled evaluation set from your own library, varied in age, skin tone, lighting and pose. Record each model's merge and split error rates per group in `tests/eval/`. Don't claim "unbiased". Report numbers.
- **Neutral model choice.** Every model comparison lists strengths, weaknesses and licence side by side. No vendor is the default because of who makes it. Defaults follow the criteria above section 4 and are confirmed by our own evaluation.

**Privacy**

- No network calls unless the user turns on a cloud plug-in. Each such plug-in shows exactly what leaves the machine.
- No telemetry by default.
- Face fingerprints are sensitive biometric data. They stay local. "Forget this person" deletes their faces, embeddings, crops and thumbnails, and records a suppression entry so the next scan does not re-create them.
- Matching against outside photo collections is out of scope.

---

## 9. Milestones and open decisions

### Milestones

| Milestone | Scope | Done when |
|---|---|---|
| M1 Core ✅ | Layered skeleton, config, SQLite schema + migrations, writer thread, scanner with OneDrive placeholder check, derive stage, job queue, buffalo_l ONNX adapter, CLI (`facegroup scan <dir>`), unit tests with fake adapters | 10k photos index without crashing, resume after being killed, and peak RAM is measured |
| M2 People (steps 1–2 ✅, evaluation open) | k-NN assignment, cluster suggestions, people screen with accept/merge/split/move/reject and undo, local API security | A 2k-photo library can be cleaned up in under 15 minutes, and property tests show no user decision is ever overwritten |
| M3 Visual search ✅ (SigLIP 2) | Image-text fingerprints, search bar, filters (person, date, folder) | "beach sunset" returns relevant images on the test set |
| M4 Text | OCR, optional captions, text fingerprints, combined search | Search for words that appear in images (receipts, signs) |
| M5 Video and polish | Video frame sampling + face tracks, folder watching, installer, evaluation report | One-click Windows install on a clean machine |

### M1 results (2026-10-17): done criteria met

Benchmark: `scripts/bench.py --count 10000 --kill-after 240` on the reference laptop (Intel Core Ultra, 14 cores, integrated GPU via DirectML, 16 GB). The library is 10,000 synthetic variants of a public 6-person group photo (InsightFace `t1.jpg`): flips, crops, 0.8–2.5x scale, brightness, 10% EXIF-rotated, so 4–6 faces per photo.

| Criterion | Result |
|---|---|
| 10k photos index without crashing | 10,000 / 10,000 derived and face-analysed, **0 failed**; 53,365 faces stored with embeddings |
| Resume after being killed | Hard-killed at 240 s (6,450 derived, 6 mid-job). The next run re-queued the 6 and finished everything; nothing lost or duplicated |
| Peak RAM measured | **0.92 GB** peak working set (process total, model loaded) |
| Speed | Derive about 27 photos/s. Faces about **5.1 photos/s (about 27 faces/s)** on the integrated GPU. Total 37 min for 10k photos at 5.3 faces each; real libraries average fewer faces per photo, so they will be faster |
| Disk | Database 251 MB, thumbnails 250 MB; working images cleaned up as designed |

### Accuracy check (2026-10-24)

The scale benchmark above measures robustness, not accuracy. Accuracy is in [docs/EVALUATION.md](docs/EVALUATION.md): LFW 99.83% (matches the published model, so our detection and alignment port is correct), 0 failures on 254 real photos, and the published per-group gap for buffalo_l (East Asian 75.0 vs Caucasian 94.7 on InsightFace's MR benchmark), to be measured on our own data in M2.

### Decided (2026-10-17)

- **Interface:** FastAPI + server-rendered pages with HTMX, opened in the browser. No Node.js toolchain.
- **Library size:** under 50k items, so sqlite-vec and HDBSCAN are enough; LanceDB and Chinese Whispers stay as fallbacks only.
- **Prototype:** archived to `archive/Local_Face_Finder_Windows11/` as reference. FaceGroup is built fresh.

### Open decisions

1. **Interface (decide before M1, because it shapes the API).** Options:
   - Keep Tkinter: simplest, nothing new to install, dated look; the UI would call services directly.
   - **FastAPI + server-rendered pages with HTMX** (review recommendation): opens in a browser, no Node toolchain.
   - FastAPI + React: richest UI, adds a Node build.
   - Tauri wrapper around a web UI: a desktop app; adds a Rust toolchain.
2. **Default face model.** InsightFace buffalo_l (proposed; allowed for personal use) or AuraFace (fully open). Compare in M2 and keep the better one.
3. **Image search model.** ✅ Decided 2026-10-24: SigLIP 2 now (verified ONNX, ~1 h per 50k photos on the GPU); EmbeddingGemma 2 later as an optional higher-accuracy mode (86% vs 78.5% R@1, but ~35x slower here). See docs/EVALUATION.md §6.
4. **Project licence (only if published later).** Options: MIT, Apache-2.0 (adds a patent grant), or AGPL-3.0, which keeps hosted forks open (Immich uses AGPL).
5. **Library size to aim for.** Thousands vs. hundreds of thousands of items. This decides between sqlite-vec and LanceDB, and between HDBSCAN and Chinese Whispers for suggestions.
6. **Captions in v1?** They are 1–3 seconds per image on the processor. They could ship off by default.

---

## 10. Sources (licences checked 2026-10-17)

- Face: [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) · [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) · [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) · [AuraFace v1](https://huggingface.co/fal/AuraFace-v1) · [dlib-models](https://github.com/davisking/dlib-models) · [face_recognition](https://github.com/ageitgey/face_recognition) · [EdgeFace](https://huggingface.co/Idiap/EdgeFace-Base) · [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- Context: [Immich facial recognition](https://docs.immich.app/features/facial-recognition) · [PhotoPrism face recognition](https://docs.photoprism.app/developer-guide/vision/face-recognition/)
- Image fingerprints: [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) · [EmbeddingGemma 300M](https://huggingface.co/google/embeddinggemma-300m) · [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) · [OpenCLIP LAION ViT-B/32](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) · [MobileCLIP](https://github.com/apple/ml-mobileclip) · [Nomic Embed Vision v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) · [Jina CLIP v2](https://huggingface.co/jinaai/jina-clip-v2) · [Gemini Embedding API](https://ai.google.dev/gemini-api/docs/embeddings) · [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- Vision-language models: [Florence-2 base](https://huggingface.co/microsoft/Florence-2-base) · [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) · [Qwen3-VL 2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) · [Qwen3.5 2B](https://huggingface.co/Qwen/Qwen3.5-2B) · [Qwen2.5-VL-3B licence](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) · [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) · [Gemma Terms](https://ai.google.dev/gemma/terms) · [InternVL3.5 2B](https://huggingface.co/OpenGVLab/InternVL3_5-2B) · [Moondream 2](https://huggingface.co/vikhyatk/moondream2) · [Moondream licence](https://moondream.ai/license) · [Phi-4-multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- Text fingerprints: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) · [nomic-embed-text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) · [BGE-M3](https://huggingface.co/BAAI/bge-m3) · [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- OCR: [RapidOCR](https://github.com/RapidAI/RapidOCR) · [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) · [Tesseract](https://github.com/tesseract-ocr/tesseract)
- Vector stores: [sqlite-vec](https://github.com/asg017/sqlite-vec) · [LanceDB](https://github.com/lancedb/lancedb) · [FAISS](https://github.com/facebookresearch/faiss) · [Qdrant](https://github.com/qdrant/qdrant)
- OSI licence list: [opensource.org/licenses](https://opensource.org/licenses)
