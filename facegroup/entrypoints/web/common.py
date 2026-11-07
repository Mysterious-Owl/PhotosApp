"""Small response helpers shared by the web routes."""

from __future__ import annotations

from fastapi.responses import RedirectResponse


def safe_next(url: str | None, default: str = "/people") -> str:
    """Only same-site relative paths, so a crafted form cannot redirect elsewhere."""
    if not url or not url.startswith("/") or url[1:2] in ("/", "\\"):  # "//x" and "/\x" leave the site
        return default
    return url


def redirect(url: str) -> RedirectResponse:
    """303: after a POST the browser follows with a GET."""
    return RedirectResponse(url, status_code=303)
