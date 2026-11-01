"""Opening the UI: in an Edge app window when possible, and handing a page over to a FaceGroup that is already running.

Only one FaceGroup can use a data folder at a time. The running one writes `session.json` (port and login token)
into the data folder; a second launch, for example from the right-click menu, reads it and opens the requested
page in the running app instead of failing. The token is only as secret as the data folder itself, which already
holds everything FaceGroup knows.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import webbrowser
from pathlib import Path
from urllib.parse import quote, urlencode

SESSION_FILE = "session.json"
PAGES = {"photos": "/photos", "people": "/people", "duplicates": "/files/duplicates", "convert": "/tools/convert",
         "metadata": "/tools/metadata"}


def page_url(page: str, folder: str | None) -> str:
    """The path to open: a named page, with a folder filled in for the tools. Drive roots arrive from Explorer
    as `C:"` (the closing quote escaped by the backslash), so stray quotes are removed."""
    path = PAGES.get(page, "/photos")
    if folder:
        folder = folder.strip().strip('"')
        if len(folder) == 2 and folder[1] == ":":
            folder += "\\"
        path += "?" + urlencode({"folder": folder})
    return path


def login_url(port: int, token: str, next_path: str) -> str:
    return f"http://127.0.0.1:{port}/login?t={token}&next={quote(next_path, safe='')}"


def write_session(data_dir: Path, port: int, token: str) -> None:
    path = data_dir / SESSION_FILE
    path.write_text(json.dumps({"port": port, "token": token, "pid": os.getpid()}), encoding="utf-8")


def read_session(data_dir: Path) -> dict | None:
    try:
        return json.loads((data_dir / SESSION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def clear_session(data_dir: Path) -> None:
    (data_dir / SESSION_FILE).unlink(missing_ok=True)


def _edge() -> str | None:
    found = shutil.which("msedge")
    if found:
        return found
    for base in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES")):
        if base and (p := Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe").exists():
            return str(p)
    return None


def open_window(url: str) -> None:
    """An Edge app window (no tabs or address bar), like a desktop app; the default browser otherwise."""
    edge = _edge() if os.name == "nt" else None
    if edge:
        subprocess.Popen([edge, f"--app={url}"])
    else:
        webbrowser.open(url)
