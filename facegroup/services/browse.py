"""BrowseService: timeline, "on this day", events and saved searches. Read-only apart from saved searches."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qsl, urlencode

from facegroup.domain.browse import by_month, events, month_label, on_this_day
from facegroup.ports.search import SearchFilters

SAVED_KEYS = ("q", "person", "date_from", "date_to", "folder", "kind", "fav")


@dataclass
class Tile:
    media_id: int
    stack_size: int = 1


class BrowseService:
    def __init__(self, store, event_gap_hours: float = 8.0, event_min_photos: int = 6) -> None:
        self._store = store
        self._gap = event_gap_hours
        self._min = event_min_photos

    def _fold(self, ids: list[int]) -> list[Tile]:
        """Bursts show once: the top pick if it is in view, else the first shot, with the count of the rest."""
        stacks = self._store.stack_map(ids)
        shown: dict[int, int] = {}
        tiles: list[Tile] = []
        for m in ids:
            info = stacks.get(m)
            if info is None:
                tiles.append(Tile(m))
                continue
            stack_id, is_top, size = info
            if stack_id not in shown:
                shown[stack_id] = len(tiles)
                tiles.append(Tile(m, size))
            elif is_top:
                tiles[shown[stack_id]] = Tile(m, size)
        return tiles

    def timeline(self, filters: SearchFilters, page: int = 0, months_per_page: int = 6) -> dict:
        months = by_month(self._store.dated_media(filters))
        chunk = months[page * months_per_page:(page + 1) * months_per_page]
        return {
            "months": [(key, month_label(key), self._fold(ids)) for key, ids in chunk],
            "jump": [(key, month_label(key), len(ids)) for key, ids in months],
            "page": page,
            "more": (page + 1) * months_per_page < len(months),
            "total": sum(len(ids) for _, ids in months),
        }

    def month_page(self, filters: SearchFilters, key: str, months_per_page: int = 6) -> int:
        """Which timeline page shows month `key` (YYYY-MM). Same filters as `timeline`, or the pages differ."""
        keys = [k for k, _ in by_month(self._store.dated_media(filters))]
        return keys.index(key) // months_per_page if key in keys else 0

    def memories(self, filters: SearchFilters, today: date | None = None) -> list[tuple[int, list[Tile]]]:
        today = today or date.today()
        dated = self._store.dated_media(filters)
        found = on_this_day(dated, today)
        if not found:  # nothing on the exact day: widen to the same week
            found = on_this_day(dated, today, window_days=3)
        return [(years, self._fold(ids)[:12]) for years, ids in found]

    def events(self, filters: SearchFilters) -> list[dict]:
        out = []
        for e in events(self._store.dated_media(filters), self._gap, self._min):
            folders = Counter(p.replace("\\", "/").rpartition("/")[0] for p in self._store.folder_of(e.media_ids).values())
            folder = folders.most_common(1)[0][0].rpartition("/")[2] if folders else ""
            out.append({"key": e.start, "title": e.title, "folder": folder, "count": len(e.media_ids),
                        "cover": e.media_ids[len(e.media_ids) // 2], "media_ids": e.media_ids})
        return out

    def event(self, filters: SearchFilters, key: str) -> dict | None:
        found = next((e for e in self.events(filters) if e["key"] == key), None)
        if found:
            found["tiles"] = self._fold(found["media_ids"])
        return found

    # -- saved searches -----------------------------------------------------------------------------------

    def saved(self) -> list[dict]:
        return self._store.saved_searches()

    def save(self, name: str, params: dict) -> None:
        if not name.strip():
            raise ValueError("A saved search needs a name.")
        query = urlencode([(k, v) for k, v in params.items() if k in SAVED_KEYS and v])
        self._store.save_search(name, query)

    def delete(self, search_id: int) -> None:
        self._store.delete_saved_search(search_id)

    @staticmethod
    def safe_query(query: str) -> str:
        """Only known search parameters survive, so a saved search can only ever open the search page."""
        return urlencode([(k, v) for k, v in parse_qsl(query) if k in SAVED_KEYS])
