# Local Face Finder (Windows 11)

A privacy-first desktop photo browser that scans a folder, detects faces, and groups likely matches locally. Original files are read-only: the app does not rename, move, edit, or delete photos.

## Features
- Choose a photo folder in a desktop window
- Local face detection and face embeddings (InsightFace `buffalo_l`, when installed)
- Group likely same-person faces; rename groups locally
- Thumbnail gallery, file path display, and open original photo
- SQLite index stored in your user profile
- No cloud upload or network features in the app

## Important privacy / model note
The optional face-recognition model is downloaded once during setup by the package when first initialized, unless you supply its model files yourself. After setup, scanning and matching run locally. If you need a strictly offline install, download/install dependencies and model files on a trusted connected machine first, then transfer the environment/model files. The app itself does not upload images.

Face similarity can be wrong, especially with blurry, side-profile, low-light, or very old photos. Review groups before relying on them. This app is intended for organizing your own photo library, not identifying strangers or making consequential decisions.

## Install on Windows 11

1. Install **Python 3.11 (64-bit)** from https://www.python.org/downloads/windows/ and check **Add python.exe to PATH**.
2. Extract this ZIP to a folder such as `C:\\FaceFinder`.
3. Double-click `install.bat`. This creates a virtual environment and installs packages.
4. Double-click `run.bat`.

The first setup may take a few minutes. The first start may download the InsightFace model files. For a strictly offline runtime, complete model setup before disconnecting from the internet.

## How to use
1. Click **Choose Folder** and select the folder containing your photos.
2. Click **Scan Folder**. Keep the app open while scanning.
3. Click a person group to see its photos. Use **Name person** to label a group.
4. Double-click a photo or use **Open Original** to open it in the Windows photo viewer.

The index is stored in `%LOCALAPPDATA%\\LocalFaceFinder\\facefinder.sqlite3`. Deleting this database removes the app's index and labels, not your photos.

## Supported formats
JPEG, PNG, BMP, TIFF, and WebP. HEIC support depends on Pillow's installed codecs; convert HEIC files to JPEG if they are not recognized.

## Troubleshooting
- If install fails, make sure Python 3.11 64-bit is installed and retry `install.bat`.
- If model initialization fails, check internet access during first run or see InsightFace model setup instructions.
- Scanning a large library can take a long time. Start with a small test folder.
- This MVP uses CPU by default for compatibility; GPU acceleration is not required.
