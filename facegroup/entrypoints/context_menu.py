"""Explorer right-click menu: a "FaceGroup" submenu on folders, folder backgrounds and drives (current user only).

Each item runs FaceGroup.cmd with a page and the folder, which opens that tool with the folder filled in (or hands
it to the FaceGroup that is already running). On Windows 11 it is under "Show more options".
"""

from __future__ import annotations

from pathlib import Path

ROOT_KEY = r"Software\Classes"
MENU = "FaceGroup"
ITEMS = [  # (key, label, page)
    ("1dupes", "Find duplicate files here", "duplicates"),
    ("2convert", "Convert HEIC photos", "convert"),
    ("3metadata", "Check or remove metadata", "metadata"),
    ("4open", "Open FaceGroup", "photos"),
]
# Where the menu appears, and how that place passes the folder to the command.
PLACES = [(r"Directory\shell", "%1"), (r"Directory\Background\shell", "%V"), (r"Drive\shell", "%1")]


def commands(launcher: Path) -> dict[str, str]:
    """Registry values: key path -> command line (or label). Pure, so it can be tested anywhere."""
    out: dict[str, str] = {}
    for place, arg in PLACES:
        base = rf"{ROOT_KEY}\{place}\{MENU}"
        for key, label, page in ITEMS:
            out[rf"{base}\shell\{key}"] = label
            folder = "" if page == "photos" else f' --folder "{arg}"'
            out[rf"{base}\shell\{key}\command"] = f'"{launcher}" --page {page}{folder}'
    return out


def install(launcher: Path) -> int:
    import winreg

    for place, _ in PLACES:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{ROOT_KEY}\{place}\{MENU}") as k:
            winreg.SetValueEx(k, "MUIVerb", 0, winreg.REG_SZ, "FaceGroup")
            winreg.SetValueEx(k, "SubCommands", 0, winreg.REG_SZ, "")
    for key, value in commands(launcher).items():
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key) as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, value)
    return len(PLACES) * len(ITEMS)


def remove() -> None:
    import winreg

    def delete_tree(parent, name: str) -> None:
        try:
            with winreg.OpenKey(parent, name, 0, winreg.KEY_ALL_ACCESS) as k:
                while True:
                    try:
                        delete_tree(k, winreg.EnumKey(k, 0))
                    except OSError:
                        break
            winreg.DeleteKey(parent, name)
        except FileNotFoundError:
            pass

    for place, _ in PLACES:
        delete_tree(winreg.HKEY_CURRENT_USER, rf"{ROOT_KEY}\{place}\{MENU}")
