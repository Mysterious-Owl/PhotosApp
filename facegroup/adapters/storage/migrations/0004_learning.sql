-- Self-improvement: answers to "same person?" questions and a scorecard of automatic matches.

-- "Face 7 and Face 12 are different people": never ask again, never suggest merging them.
CREATE TABLE person_distinct (
    a  INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    b  INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    at TEXT    NOT NULL,
    PRIMARY KEY (a, b),
    CHECK (a < b)
) WITHOUT ROWID;

-- Every time the user confirms or corrects an automatic match: was FaceGroup right? Removed with its event on undo.
CREATE TABLE assign_feedback (
    id        INTEGER PRIMARY KEY,
    event_id  INTEGER NOT NULL REFERENCES people_event(id) ON DELETE CASCADE,
    face_id   INTEGER NOT NULL,
    person_id INTEGER NOT NULL,           -- the person FaceGroup had matched it to
    score     REAL,                       -- its confidence at the time
    correct   INTEGER NOT NULL,           -- 1 = user confirmed, 0 = user rejected or moved it
    at        TEXT    NOT NULL
);
CREATE INDEX assign_feedback_at ON assign_feedback(at);
