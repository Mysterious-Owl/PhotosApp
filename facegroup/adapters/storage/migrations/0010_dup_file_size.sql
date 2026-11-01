-- Copies in a set can differ in size now: a HEIC photo and the PNG made from it are a "converted" set.
ALTER TABLE dup_file ADD COLUMN size INTEGER;  -- NULL for rows from before: same as dup_group.size
