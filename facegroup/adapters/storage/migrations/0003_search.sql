-- Milestone M3: image embeddings and keyword search. See docs/ARCHITECTURE.md §5c.

CREATE TABLE item_embedding (
    item_id  INTEGER NOT NULL REFERENCES item(id) ON DELETE CASCADE,
    model_id TEXT    NOT NULL,
    vector   BLOB    NOT NULL,            -- float32, L2-normalised; canonical copy
    PRIMARY KEY (item_id, model_id)
) WITHOUT ROWID;

-- Words from folder and file names ("2023 Goa trip/IMG_0042.jpg" -> "2023 goa trip img 0042 jpg").
-- (OCR text got its own table, ocr_fts, in migration 0006.)
CREATE VIRTUAL TABLE media_text USING fts5(
    media_id UNINDEXED,
    path_words,
    tokenize = 'unicode61 remove_diacritics 2'
);
INSERT INTO media_text(media_id, path_words) SELECT id, rel_path FROM media;

-- Kept in sync by triggers, so no code path can forget it. The tokenizer splits on '/', '_', '.', '-'.
CREATE TRIGGER media_text_ins AFTER INSERT ON media BEGIN
    INSERT INTO media_text(media_id, path_words) VALUES (new.id, new.rel_path);
END;
CREATE TRIGGER media_text_upd AFTER UPDATE OF rel_path ON media BEGIN
    UPDATE media_text SET path_words = new.rel_path WHERE media_id = new.id;
END;
CREATE TRIGGER media_text_del AFTER DELETE ON media BEGIN
    DELETE FROM media_text WHERE media_id = old.id;
END;

CREATE INDEX media_taken ON media(taken_at);
