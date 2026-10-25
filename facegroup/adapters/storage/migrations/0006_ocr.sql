-- Text found in images (OCR), searchable. Never shown in logs (see CLAUDE.md).

CREATE TABLE item_text (
    item_id  INTEGER NOT NULL REFERENCES item(id) ON DELETE CASCADE,
    source   TEXT    NOT NULL,              -- ocr (captions later)
    model_id TEXT    NOT NULL,
    body     TEXT    NOT NULL,
    PRIMARY KEY (item_id, source)
) WITHOUT ROWID;

CREATE VIRTUAL TABLE ocr_fts USING fts5(
    media_id UNINDEXED,
    item_id UNINDEXED,
    body,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Keep the search index in step with item_text, whoever writes or deletes it (including cascades).
CREATE TRIGGER item_text_ins AFTER INSERT ON item_text WHEN new.source = 'ocr' BEGIN
    INSERT INTO ocr_fts(media_id, item_id, body)
    VALUES ((SELECT media_id FROM item WHERE id = new.item_id), new.item_id, new.body);
END;
CREATE TRIGGER item_text_del AFTER DELETE ON item_text WHEN old.source = 'ocr' BEGIN
    DELETE FROM ocr_fts WHERE item_id = old.item_id;
END;
CREATE TRIGGER item_text_upd AFTER UPDATE OF body ON item_text WHEN new.source = 'ocr' BEGIN
    UPDATE ocr_fts SET body = new.body WHERE item_id = new.item_id;
END;
