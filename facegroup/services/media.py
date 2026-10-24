"""MediaService: images for the UI, always looked up by id (never by a path the client supplies)."""

from __future__ import annotations

from pathlib import Path

import cv2

from facegroup.adapters.media.cache import DiskImageCache
from facegroup.ports.media import MediaDecoder, PermanentMediaError


class MediaService:
    def __init__(self, store, people_store, decoder: MediaDecoder, cache: DiskImageCache) -> None:
        self._store = store
        self._people = people_store
        self._decoder = decoder
        self._cache = cache

    def face_crop(self, face_id: int) -> Path | None:
        path = self._cache.face_crop_path(face_id)
        if path.exists():
            return path
        found = self._people.face_media(face_id)
        if found is None:
            return None
        media_id, bbox = found
        try:  # faces indexed before close-ups existed: rebuild from the original, once
            image = self._decoder.decode(self._store.media_path(media_id)).rgb
        except (PermanentMediaError, OSError, KeyError):
            return None
        self._cache.write_face_crop(face_id, image, bbox)
        return path if path.exists() else None

    def thumbnail(self, media_id: int) -> Path | None:
        path = self._cache.thumbnail_path(media_id)
        if path.exists():
            return path
        try:
            image = self._decoder.decode(self._store.media_path(media_id)).rgb
        except (PermanentMediaError, OSError, KeyError):
            return None
        self._cache.write_thumbnail(media_id, image)
        return path

    def photo_jpeg(self, media_id: int) -> bytes | None:
        """The photo as a browser-viewable JPEG (HEIC/TIFF included), EXIF-rotated, at working size."""
        try:
            rgb = self._decoder.decode(self._store.media_path(media_id)).rgb
        except (PermanentMediaError, OSError, KeyError):
            return None
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else None

    def media_label(self, media_id: int) -> str:
        try:
            return str(self._store.media_path(media_id))
        except KeyError:
            return ""
