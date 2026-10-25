# FaceGroup — Evaluation

As of 2026-10-24 · Covers M1 (faces), M2 (grouping, §3b) and M3 (search, §6).

**Summary:** our re-implementation of InsightFace's pipeline reproduces the model's published accuracy on the standard LFW benchmark (99.83%, identical to InsightFace's published 99.83%), so the detection, alignment and fingerprint code is correct. On 254 real photos it indexed every file without a failure, and the faces it found were real faces. It also finds drawn, cartoon and animal faces, which stage 2 has to handle. Accuracy on different groups of people has **not** been measured on our own data yet; InsightFace's published numbers show a large gap for East Asian faces.

## 1. What each test can and cannot show

| Test | Data | Shows | Does not show |
|---|---|---|---|
| Scale benchmark (`scripts/bench.py`) | 10,000 synthetic variants of one 6-person photo | Robustness, crash recovery, memory, throughput | Any kind of accuracy: there are only 6 identities |
| LFW verification (`tests/eval/eval_lfw.py`) | 6,000 standard pairs, 7,701 news photos of 5,749 people | Whether same-person / different-person decisions match the published model; catches silent alignment bugs | Performance on children, phone photos or non-celebrity photos; LFW is mostly adult, front-facing, and skewed toward white men |
| Real-library profile (`scripts/profile_library.py`) | 254 files in `OneDrive\Pictures`, read-only | Real formats and sizes, failure handling, false detections, speed on mixed content | Recognition accuracy (no labels) |

## 2. LFW verification

**Method.** Each image goes through the same code as the app: Pillow decoder → SCRFD detection (our port) → 5-point alignment (our port) → ArcFace R50 embedding. The face nearest the image centre is used. Standard 10-fold protocol: the threshold is chosen on 9 folds and accuracy measured on the held-out fold.

| Run | Accuracy | Images with no face detected |
|---|---|---|
| Published, InsightFace model zoo (buffalo_l) | **99.83%** | — |
| FaceGroup v1 (DirectML) | **99.82% ± 0.17** | 11 of 7,701 (fallback centre crop used) |
| FaceGroup v1.1, with padded retry (DirectML) | **99.83% ± 0.17** | **0** of 7,701 |

**The 11 misses** were faces that fill the frame and touch its edges, often partly covered (a hand, a cap brim). SCRFD is known to miss these. This matters for close-up selfies. Adding a 25% black border and detecting again recovered **all 11**, so the app now retries this way whenever a photo has no detections (cost: one extra detection, only on face-less photos). A close-up integration test guards it.

**Similarity scores** (cosine, 1 = identical), which inform the M2 grouping threshold:

| | 1st pct | 5th pct | median | 95th pct | 99th pct |
|---|---|---|---|---|---|
| Same person | 0.39 | 0.50 | 0.68 | | |
| Different people | | | 0.00 | 0.10 | 0.14 |

The best LFW threshold is about **0.24**. The prototype's 0.48 would have rejected about 5% of genuine same-person pairs. This does *not* mean 0.24 is right for grouping: LFW is easy and skewed (see §4). M2 will pick the threshold on a labelled sample of the user's own photos.

## 3. Real library: `OneDrive\Pictures`

| Measure | Result |
|---|---|
| Files | 254 (242 PNG, 12 JPEG), median 1.2 MP, largest 8.3 MP |
| Failures | **0** |
| Time | 32.6 s including model load; peak memory 0.89 GB |
| Photos with faces | 61 of 254; 153 faces in total (up to 9 per photo) |
| EXIF dates / GPS | 11 / 0 (mostly screenshots and saved images, which have no camera metadata) |

**Manual review of contact sheets** (lowest-confidence faces, smallest faces, and 48 photos where nothing was found):

- The lowest-confidence detections are genuine faces, mostly small or blurry faces in video-call screenshots. No background objects were detected as faces.
- **Drawn, anime and animal faces are detected** (cartoon characters, an illustrated skull, monkeys). That is correct for a face detector, but these are not people. M2 needs a "not a person" action, and they will mostly fall into "Unsorted" because their fingerprints are not consistent.
- The sampled photos without detections are wallpapers, anime scenes and web-page screenshots. No real person's face was missed in that sample.
- Low quality scores (down to 0.04) correspond to visibly blurry or tiny faces, so the score is usable for picking reference faces in M2.

This library is small and mostly screenshots. It does not include phone camera photos (HEIC, 12–50 MP), so speed and decoding for those remain untested on real data.

## 3b. Suggested groups on the real library (M2 tuning, 2026-10-24)

The 153 faces from §3 were clustered (HDBSCAN, leaf selection) and reviewed as contact sheets.

| Version | Groups | Mixed groups (wrong) | Same person split | Faces grouped |
|---|---|---|---|---|
| v1: drop members far from the group *centre* (< 0.35) | 23 | **3 people-groups mixed** (+2 mixed cartoon/animal groups) | 3 people split across 7 groups | 104 |
| v2: drop members whose mean similarity to the *other* members is < 0.40, then merge groups whose faces average ≥ 0.55 | 15 | **0** | 0 seen | 87 |

Why v1 failed: in a group of three, two faces of one person pull the centre towards themselves, so a stranger still looked close to it (centre 0.53, but 0.11 to the other members). Measured over all groups:

- Every pure group had all members at ≥ 0.52 mean similarity to the rest; every mixed group had a member at ≤ 0.35. The cut-off is 0.40.
- Split groups of one person averaged 0.67–0.87 face-to-face; the closest pair of different people averaged 0.35. The merge level is 0.55.

v2 trades coverage for purity on purpose: a face left unsorted costs one click later, while a stranger inside a person is easy to miss. Both thresholds are settings (`suggest_member_similarity`, `suggest_merge_similarity`). They come from one small library (about 25 people) and should be re-checked on a larger one.

## 4. Fairness: what is known and what is not

InsightFace publishes buffalo_l results on its multi-racial benchmark (MR, a strict true-accept rate at a very low false-accept rate, so the numbers are lower than LFW and not comparable to it):

| MR-ALL | African | Caucasian | South Asian | East Asian |
|---|---|---|---|---|
| 91.25 | 90.29 | 94.70 | 93.16 | **74.96** |

Source: [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo), checked 2026-10-24.

The gap for East Asian faces is about 20 points. In this app that would appear as more *split* groups (one person spread over several suggestions) and more *merges* between different people, for East Asian faces than for others. We have not measured it ourselves. LFW cannot measure it: it is dominated by white adult men.

**Plan (M2):**
1. Label a small set of faces from your own library (around 20 people, several photos each, a mix of ages), and measure merge and split rates per group.
2. Compare buffalo_l with AuraFace on that set. AuraFace is the open alternative in the plan. Keep whichever has the smaller gap at similar accuracy.
3. Keep "you have the final say" as the safety net: suggestions only, and one-click merge, split and reject.

## 5. Not yet tested

- Real phone camera photos: HEIC, 12–50 MP, EXIF rotation from actual phones.
- Real OneDrive online-only files (only simulated in tests; this library had none).
- Children and age progression (same person across 10+ years).
- Grouping quality: comes with M2.

## 6. Search model (M3)

**Decision: SigLIP 2 base, run with ONNX at fp16 on the GPU (fp32 on CPU). EmbeddingGemma 2 is more accurate and is planned as an optional mode.**

### 6a. Model comparison: official reference implementations, same data, same machine

Flickr30k 1K test set: 1,000 photos, 5 human captions each. "Photo from caption" = for each caption, is the right photo ranked first (R@1) or in the top 5? `tests/eval/eval_retrieval.py`, run in a separate PyTorch environment.

| Model | Photo from caption R@1 | R@5 | R@10 | Caption from photo R@1 | Dim | Reference CPU time / photo* |
|---|---|---|---|---|---|---|
| OpenCLIP ViT-B/32 (LAION-2B) | 66.5% | 88.2% | 93.0% | 84.2% | 512 | 0.05 s |
| SigLIP 2 base/16 224 | 78.5% | 93.9% | 96.7% | 91.8% | 768 | 0.18 s |
| **EmbeddingGemma 2** | **86.4%** | **97.5%** | **98.7%** | **96.2%** | 768 | 3–10 s |

\*PyTorch on CPU, partly measured while other jobs ran, so indicative only. OpenCLIP's 66.5% matches its published Flickr30k figure (~66%), which confirms the harness is set up correctly.

### 6b. What the app can actually run

| Runtime on this laptop | Time / photo | 50k photos |
|---|---|---|
| SigLIP 2, ONNX fp16, DirectML (integrated GPU) | **0.07 s** | ~1 h |
| SigLIP 2, ONNX fp32, CPU | 0.23 s | ~3 h |
| EmbeddingGemma 2 image encoder, ONNX fp32, CPU (full detail, 2,520 patches) | 2.9 s + text pass | ~40 h+ |
| EmbeddingGemma 2 image encoder, ONNX, DirectML | fails (unsupported Reshape in onnxruntime-directml 1.24) | — |

EmbeddingGemma 2 is about 8 points better at R@1 but about 35-40x slower here, and its ONNX pipeline (image patching plus splicing image tokens into the text model) would need its own parity check. Decision (2026-10-24): ship SigLIP 2 now and add EmbeddingGemma 2 later as an optional, slower, higher-accuracy mode.

### 6c. Our ONNX port vs the reference (catches silent bugs)

`tests/eval/eval_retrieval_onnx.py` runs the app's adapter on the same data and compares each vector with the reference's.

| SigLIP 2 ONNX files | Device | Photo from caption R@1 | Agreement with reference (cosine, min) |
|---|---|---|---|
| Reference (PyTorch) | CPU | 78.5% | — |
| **fp32** | CPU / DirectML | **78.5%** | **0.9995** |
| **fp16** (shipped on GPU) | DirectML | **78.3%** | **0.9988** |
| int8 "quantized" (published) | CPU | 35.9% | 0.56 |
| int8 "quantized" (published) | DirectML | 28.1% | 0.40 (text side) |

**The published int8 files are badly degraded** and are not used. The preprocessing was ruled out as the cause: pixel values match the reference processor to 1/127 and token ids match exactly. fp16 and fp32 vectors are interchangeable (cosine 0.999), so the model id does not include precision and switching device never forces a re-index.

### 6d. Real library

All 254 files of `OneDrive\Pictures` were embedded in about 17 s, including model load, with peak memory 1.03 GB. Eight queries were checked by eye against the actual content, and the top result was right for all eight: video-call screenshots, anime scenes, space wallpapers, checkout pages, the two monkey pictures, the cherry-blossom wallpaper, a specific comic, and a physics textbook page. A query takes about 50 ms after the text model has loaded; that load takes about 5 s and now happens in the background at start-up.

## 7. Declutter, duplicates, bursts (batch A, 2026-10-25)

All measured on public photos (Flickr30k 1K) and synthetic images, never the user's library. Script: `tests/eval/eval_categories.py`.

**Categories** compare each photo's existing SigLIP 2 vector with short descriptions per category, so no new model is needed. A photo only counts as clutter when a clutter category beats "photo" by the margin.

| Margin | Real photos called clutter (of 1,000) | Synthetic screenshots / documents / receipts / memes / illustrations caught |
|---|---|---|
| 0.03 | 3 (0.3%) | 100 / 100 / 100 / 98 / 100% |
| **0.04 (chosen)** | **0** | 100 / 100 / 100 / 80 / 100% |
| 0.05 | 0 | 100 / 100 / 100 / 42 / 78% |

Screenshots are also caught by a rule: a "Screenshot" file name, or a PNG with no camera date at an exact screen size. Synthetic clutter is crude, so its recall is a sanity check only. Real accuracy depends on the library, and wrong guesses are fixed per photo and never overwritten.

**Duplicates.** Image embeddings could not separate copies from different photos: resized and re-saved copies scored as low as 0.85, while different public photos scored as high as 0.94. A 64-bit difference hash, taken from the 320-px thumbnail, separates them cleanly:

| Pair type (public photos) | Hash difference (bits) |
|---|---|
| Resized + recompressed copies | ≤ 7 (99% ≤ 3) |
| Different photos | ≥ 10 |
| 15% crops of the same scene | ≥ 15 |

So a near copy is 6 bits or fewer. Flat images (blank or black) have degenerate hashes, with fewer than 8 or more than 56 bits set, so they are excluded; this was found by the tests. Exact copies come from the file hash.

**Bursts:** same folder, within 15 s, and embedding similarity ≥ 0.88 (crops of one scene stayed ≥ 0.90; 99.9% of different photos are < 0.81).

## 8. Text in images (batch B, 2026-10-25)

RapidOCR 3.10 with PaddleOCR PP-OCRv6 small detection and recognition (Apache-2.0; models bundled in the package, so nothing is downloaded). Measured on 24 synthetic images with known text (`tests/eval/eval_ocr.py`), never the user's library:

| Setting | Words read back | Time per text-heavy image |
|---|---|---|
| Defaults, CPU | 100% | 3.3–3.7 s |
| Defaults, DirectML | 100% | 3.9 s (slower than CPU, so not used) |
| **Detector input capped at 960 px, no rotation classifier, CPU (shipped)** | **100%** | **2.2 s** |

Synthetic text is clean, so 100% is an upper bound; real photos of signs or crumpled receipts will be lower. Because of the time, text is read by default only in pictures categorised as screenshot, document, receipt or meme; `ocr_all_photos = true` reads every photo. Text is searched with FTS5 alongside meaning and file names (reciprocal rank fusion), and is never written to logs.

## Reproduce

```powershell
python -m uv run python tests/eval/eval_lfw.py                         # ~13 min, downloads LFW once
%LOCALAPPDATA%\FaceGroup-eval\refenv\Scripts\python tests/eval/eval_retrieval.py   # reference models (PyTorch env)
python -m uv run python tests/eval/eval_retrieval_onnx.py --variant fp16   # app's ONNX port vs reference
python -m uv run python scripts/profile_library.py "C:\path\to\photos"  # read-only
python -m uv run python scripts/bench.py --count 10000 --kill-after 240
```

All data and outputs go to `%LOCALAPPDATA%\FaceGroup-eval` and `%LOCALAPPDATA%\FaceGroup-bench`, never into the repo.
