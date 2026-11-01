-- File tools (M6): duplicate files across any folders, folder preferences, and the log of moved copies.
-- Paths here are absolute Windows paths, not library-relative: the tools work on any folder.

-- The latest duplicate scan. Replaced by the next scan; preferences and moves are kept.
CREATE TABLE dup_scan (
    id          INTEGER PRIMARY KEY,
    roots       TEXT NOT NULL,             -- JSON list of folders
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    files       INTEGER NOT NULL DEFAULT 0,
    bytes       INTEGER NOT NULL DEFAULT 0,
    cloud_only  INTEGER NOT NULL DEFAULT 0, -- online-only files, never read so never checked
    errors      TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE dup_group (
    id      INTEGER PRIMARY KEY,
    scan_id INTEGER NOT NULL REFERENCES dup_scan(id) ON DELETE CASCADE,
    hash    TEXT NOT NULL,
    size    INTEGER NOT NULL
);

CREATE TABLE dup_file (
    group_id INTEGER NOT NULL REFERENCES dup_group(id) ON DELETE CASCADE,
    path     TEXT NOT NULL,
    mtime    REAL NOT NULL,
    PRIMARY KEY (group_id, path)
) WITHOUT ROWID;

-- "Between folder A and folder B, keep the copy in `keep`." Keys are casefolded, separator-normalised folders.
CREATE TABLE folder_pref (
    folder_a TEXT NOT NULL,
    folder_b TEXT NOT NULL,                -- folder_a < folder_b
    keep     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (folder_a, folder_b)
) WITHOUT ROWID;

-- The user's pick for one group of identical files (by content hash, so it survives a rescan).
CREATE TABLE dup_choice (
    hash TEXT PRIMARY KEY,
    keep TEXT NOT NULL
) WITHOUT ROWID;

-- Every copy moved into a holding folder, so it can be restored.
CREATE TABLE removal_batch (
    id         INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind       TEXT NOT NULL,              -- duplicates | empty_folders
    files      INTEGER NOT NULL DEFAULT 0,
    bytes      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE removed_file (
    id          INTEGER PRIMARY KEY,
    batch_id    INTEGER NOT NULL REFERENCES removal_batch(id),
    original    TEXT NOT NULL,
    holding     TEXT,                      -- NULL for a removed empty folder
    size        INTEGER NOT NULL DEFAULT 0,
    kept        TEXT,                      -- the identical copy that stayed
    restored_at TEXT
);
CREATE INDEX removed_file_batch ON removed_file(batch_id);
