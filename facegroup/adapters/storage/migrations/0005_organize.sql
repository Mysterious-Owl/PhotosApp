-- Organising: categories (declutter), duplicates, stacks, and per-photo flags.
-- All of this lives in FaceGroup only; photo files are never changed.

ALTER TABLE media ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0;
ALTER TABLE media ADD COLUMN rating   INTEGER;                       -- 1..5, NULL = unrated
ALTER TABLE media ADD COLUMN hidden   INTEGER NOT NULL DEFAULT 0;    -- hidden from browsing and search
ALTER TABLE media ADD COLUMN phash    INTEGER;                       -- 64-bit difference hash, for exact-looking copies

-- What kind of picture it is. 'photo' is the default; the rest is "clutter" that can be hidden.
CREATE TABLE media_category (
    media_id INTEGER PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
    category TEXT    NOT NULL,          -- photo, screenshot, document, receipt, meme, illustration
    score    REAL,
    source   TEXT    NOT NULL,          -- auto | user (user choices are never overwritten)
    model_id TEXT
);
CREATE INDEX media_category_cat ON media_category(category);

-- Duplicate sets, recomputed: exact (same bytes) or near (near-identical image).
CREATE TABLE duplicate_member (
    media_id INTEGER PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
    set_id   INTEGER NOT NULL,
    kind     TEXT    NOT NULL,          -- exact | near
    keeper   INTEGER NOT NULL DEFAULT 0 -- the copy FaceGroup suggests keeping
);
CREATE INDEX duplicate_member_set ON duplicate_member(set_id);

-- Stacks of similar shots taken close together, recomputed; the top pick is shown, the rest fold behind it.
CREATE TABLE stack_member (
    media_id INTEGER PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
    stack_id INTEGER NOT NULL,
    top      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX stack_member_stack ON stack_member(stack_id);

-- The user's own top pick survives recomputation.
CREATE TABLE stack_pick (
    media_id INTEGER PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE
) WITHOUT ROWID;

CREATE INDEX media_flags ON media(hidden, favorite);
