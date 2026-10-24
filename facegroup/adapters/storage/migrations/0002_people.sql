-- Milestone M2: people, suggestions, decisions and undo. See docs/ARCHITECTURE.md §4b and §5b.

ALTER TABLE person ADD COLUMN kind TEXT NOT NULL DEFAULT 'person';   -- person | ignored ("not a person")
ALTER TABLE person ADD COLUMN created_at TEXT;

-- "This face is not that person." Every automatic assignment checks it.
CREATE TABLE face_reject (
    face_id   INTEGER NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    PRIMARY KEY (face_id, person_id)
) WITHOUT ROWID;

-- Audit log of people changes; payload holds what is needed to undo the change.
CREATE TABLE people_event (
    id           INTEGER PRIMARY KEY,
    action       TEXT NOT NULL,
    person_id    INTEGER,
    payload_json TEXT NOT NULL,
    at           TEXT NOT NULL,
    undone       INTEGER NOT NULL DEFAULT 0
);

-- Suggested groups of unassigned faces. Recomputed on demand; never edited by the user directly.
CREATE TABLE suggestion (
    id         INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    size       INTEGER NOT NULL
);
CREATE TABLE suggestion_face (
    suggestion_id INTEGER NOT NULL REFERENCES suggestion(id) ON DELETE CASCADE,
    face_id       INTEGER NOT NULL REFERENCES face(id) ON DELETE CASCADE,
    PRIMARY KEY (suggestion_id, face_id)
) WITHOUT ROWID;
CREATE INDEX suggestion_face_face ON suggestion_face(face_id);

-- "Forget this person": embeddings of forgotten faces, so a rescan does not bring them back.
CREATE TABLE suppressed_face (
    id       INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL,
    vector   BLOB NOT NULL
);

CREATE INDEX face_assigned ON face(assigned_by, person_id);
