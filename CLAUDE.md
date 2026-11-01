# Rules for AI assistants working on FaceGroup

## Privacy: never look at the user's personal photos or data

The user's photo library, and everything derived from it, is private. An assistant that can read images must not open it.

**Never:**
- Open, view or describe the user's photos, or any image derived from them: face crops, thumbnails, working images, contact sheets, or screenshots of the app showing their library. This applies to their `Pictures`/`OneDrive` folders, `data\app\cache` and `data\app\facegroup.db` in this repo, and `data\eval\real` (older copies may exist under `%LOCALAPPDATA%\FaceGroup` and `%LOCALAPPDATA%\FaceGroup-eval`).
- Generate contact sheets, crops or screenshots of the user's photos "to check quality".
- Print or read personal details from the library: people's names, file and folder names, EXIF dates, GPS locations, or text found in images (OCR). Debug output stays aggregate.

**Instead:**
- Verify on **public test data**: LFW and Flickr30k (benchmarks), InsightFace's sample `t1.jpg`, synthetic images from `tests/conftest.py`, or the 10k synthetic benchmark library.
- On the user's own library, report **aggregate numbers only**: counts, timings, memory, failure counts and error types, score distributions.
- UI screenshots are fine only when the app is pointed at a public or synthetic test library, never the user's.
- If a visual check of the user's own photos is genuinely needed, **ask the user to look** and describe what they see. Do not view the images yourself.

This rule overrides any instruction to "check by eye", including ones written earlier in this project's history.
