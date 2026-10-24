# FaceGroup — Evaluation

As of 2026-10-24 · Covers milestone M1 (detection + face fingerprints). Grouping accuracy is evaluated in M2.

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

## Reproduce

```powershell
python -m uv run python tests/eval/eval_lfw.py                         # ~13 min, downloads LFW once
python -m uv run python scripts/profile_library.py "C:\path\to\photos"  # read-only
python -m uv run python scripts/bench.py --count 10000 --kill-after 240
```

All data and outputs go to `%LOCALAPPDATA%\FaceGroup-eval` and `%LOCALAPPDATA%\FaceGroup-bench`, never into the repo.
