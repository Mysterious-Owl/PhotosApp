"""Browsing rules: months, "on this day", events. Pure computation over (media_id, ISO date) pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso[:19])


def by_month(dated: list[tuple[int, str]]) -> list[tuple[str, list[int]]]:
    """[(YYYY-MM, media ids)] in the order given (newest first)."""
    out: list[tuple[str, list[int]]] = []
    for media_id, iso in dated:
        key = iso[:7]
        if not out or out[-1][0] != key:
            out.append((key, []))
        out[-1][1].append(media_id)
    return out


def month_label(key: str) -> str:
    year, month = key.split("-")
    return f"{MONTHS[int(month) - 1]} {year}"


def _in_year(d: date, year: int) -> date:
    try:
        return d.replace(year=year)
    except ValueError:  # 29 February in a non-leap year
        return date(year, 2, 28)


def on_this_day(dated: list[tuple[int, str]], today: date, window_days: int = 0) -> list[tuple[int, list[int]]]:
    """[(years ago, media ids)] taken on today's month and day (± window_days) in earlier years, most recent first."""
    out: dict[int, list[int]] = {}
    for media_id, iso in dated:
        d = _dt(iso).date()
        # The anniversary may fall in last or next year near New Year: 31 Dec 2025 is "1 year ago" on 1 Jan 2027.
        for year in (today.year - 1, today.year, today.year + 1):
            years_ago = year - d.year
            if years_ago >= 1 and abs((_in_year(d, year) - today).days) <= window_days:
                out.setdefault(years_ago, []).append(media_id)
                break
    return sorted(out.items())


@dataclass
class Event:
    start: str  # ISO of the first (oldest) photo: also the event's id in URLs
    end: str
    media_ids: list[int] = field(default_factory=list)

    @property
    def title(self) -> str:
        a, b = _dt(self.start), _dt(self.end)
        if a.date() == b.date():
            return f"{a.day} {MONTHS[a.month - 1]} {a.year}"
        if (a.year, a.month) == (b.year, b.month):
            return f"{a.day}–{b.day} {MONTHS[a.month - 1]} {a.year}"
        return f"{a.day} {MONTHS[a.month - 1][:3]} – {b.day} {MONTHS[b.month - 1][:3]} {b.year}"


def events(dated: list[tuple[int, str]], gap_hours: float = 8.0, min_photos: int = 6) -> list[Event]:
    """Bursts of activity: a new event starts when more than `gap_hours` pass without a photo. Events with fewer
    than `min_photos` photos are left out (an odd snapshot is not an event). Newest first."""
    ordered = sorted(dated, key=lambda x: x[1])
    groups: list[list[tuple[int, str]]] = []
    for media_id, iso in ordered:
        if groups and (_dt(iso) - _dt(groups[-1][-1][1])).total_seconds() <= gap_hours * 3600:
            groups[-1].append((media_id, iso))
        else:
            groups.append([(media_id, iso)])
    out = [Event(g[0][1], g[-1][1], [m for m, _ in g]) for g in groups if len(g) >= min_photos]
    return sorted(out, key=lambda e: e.start, reverse=True)
