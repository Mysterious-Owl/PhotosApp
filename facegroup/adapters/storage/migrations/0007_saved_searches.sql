-- Saved searches ("smart albums"): a name and the search's query string, re-run when opened.
CREATE TABLE saved_search (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    query      TEXT NOT NULL,          -- URL query string of /search, e.g. "q=beach&kind=photos"
    created_at TEXT NOT NULL
);
