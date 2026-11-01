"""The Windows "choose a folder" dialog, for the Browse buttons.

The browser cannot show a native folder picker that returns a full path, but the server runs on the same PC, so it
shows the dialog itself. It runs in a short-lived child process: Tk must own its main thread, and a dialog left open
cannot hang the web server.
"""

from __future__ import annotations

import os
import subprocess
import sys

_DIALOG = r"""
import sys, tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)  # in front of the FaceGroup window
path = filedialog.askdirectory(parent=root, initialdir=sys.argv[1] or None, mustexist=True,
                               title=sys.argv[2])
sys.stdout.write(path or "")
"""


def pick_folder(initial: str = "", title: str = "Choose a folder", timeout: float = 600) -> str | None:
    """The folder the user chose, or None if they cancelled (or the dialog could not be shown)."""
    initial = initial.strip().strip('"')
    if initial and not os.path.isdir(initial):
        initial = os.path.dirname(initial) if os.path.isdir(os.path.dirname(initial)) else ""
    try:
        out = subprocess.run([sys.executable, "-c", _DIALOG, initial, title], capture_output=True, text=True,
                             encoding="utf-8", timeout=timeout, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return os.path.normpath(out) if out else None
