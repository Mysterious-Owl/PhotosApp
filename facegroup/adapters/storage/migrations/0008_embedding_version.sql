-- A counter that changes whenever item_embedding does, so the in-memory search matrix knows when to reload.
-- Row count and id sum were not enough: a re-indexed photo can get its old item id back.
CREATE TABLE embedding_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    n  INTEGER NOT NULL
);
INSERT INTO embedding_version(id, n) VALUES (1, 0);

CREATE TRIGGER embedding_version_ins AFTER INSERT ON item_embedding BEGIN
    UPDATE embedding_version SET n = n + 1 WHERE id = 1;
END;
CREATE TRIGGER embedding_version_upd AFTER UPDATE ON item_embedding BEGIN
    UPDATE embedding_version SET n = n + 1 WHERE id = 1;
END;
CREATE TRIGGER embedding_version_del AFTER DELETE ON item_embedding BEGIN
    UPDATE embedding_version SET n = n + 1 WHERE id = 1;
END;
