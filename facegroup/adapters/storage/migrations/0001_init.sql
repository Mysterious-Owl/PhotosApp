-- FaceGroup schema v1 (milestone M1). See docs/ARCHITECTURE.md §4.

CREATE TABLE root (
    id                INTEGER PRIMARY KEY,
    path              TEXT    NOT NULL UNIQUE,
    fetch_online_only INTEGER NOT NULL DEFAULT 0,
    enabled           INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE media (
    id            INTEGER PRIMARY KEY,
    root_id       INTEGER NOT NULL REFERENCES root(id) ON DELETE CASCADE,
    rel_path      TEXT    NOT NULL,           -- exactly as on disk, '/' separators
    size          INTEGER NOT NULL,
    mtime         REAL    NOT NULL,
    content_hash  TEXT,                       -- set during derive from bytes already read
    kind          TEXT    NOT NULL DEFAULT 'photo',
    width         INTEGER,
    height        INTEGER,
    duration_ms   INTEGER,
    taken_at      TEXT,
    gps_lat       REAL,
    gps_lon       REAL,
    file_state    TEXT    NOT NULL,           -- active, online_only, changed, missing
    missing_since TEXT,
    UNIQUE (root_id, rel_path)
);
CREATE INDEX media_state ON media(root_id, file_state);
CREATE INDEX media_size ON media(size);

CREATE TABLE item (
    id       INTEGER PRIMARY KEY,
    media_id INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    t_ms     INTEGER NOT NULL DEFAULT 0,      -- 0 for photos
    width    INTEGER NOT NULL,                -- working image
    height   INTEGER NOT NULL,
    UNIQUE (media_id, t_ms)
);

CREATE TABLE job (
    id          INTEGER PRIMARY KEY,
    media_id    INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    item_id     INTEGER REFERENCES item(id) ON DELETE CASCADE,  -- NULL before derive
    stage       TEXT    NOT NULL,
    model_id    TEXT    NOT NULL DEFAULT '',
    state       TEXT    NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    lease_until TEXT,
    not_before  TEXT,
    error       TEXT
);
CREATE UNIQUE INDEX job_key ON job(media_id, IFNULL(item_id, 0), stage, model_id);
CREATE INDEX job_queue ON job(stage, state, id);

CREATE TABLE person (
    id            INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL DEFAULT '',
    cover_face_id INTEGER REFERENCES face(id) ON DELETE SET NULL,
    hidden        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE face (
    id           INTEGER PRIMARY KEY,
    item_id      INTEGER NOT NULL REFERENCES item(id) ON DELETE CASCADE,
    x1           REAL    NOT NULL,            -- working-image pixel coordinates
    y1           REAL    NOT NULL,
    x2           REAL    NOT NULL,
    y2           REAL    NOT NULL,
    landmarks    BLOB    NOT NULL,            -- 5 x 2 float32
    det_score    REAL    NOT NULL,
    quality      REAL    NOT NULL,
    detector_id  TEXT    NOT NULL,
    track_id     INTEGER,                     -- videos (M5)
    person_id    INTEGER REFERENCES person(id) ON DELETE SET NULL,
    assigned_by  TEXT,                        -- auto, user
    assign_score REAL
);
CREATE INDEX face_item ON face(item_id);
CREATE INDEX face_person ON face(person_id);

CREATE TABLE face_embedding (
    face_id  INTEGER NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    model_id TEXT    NOT NULL,
    vector   BLOB    NOT NULL,                -- float32, L2-normalised; canonical copy
    PRIMARY KEY (face_id, model_id)
) WITHOUT ROWID;

CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
