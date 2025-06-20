CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS documents;

CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(1024),
  collection TEXT DEFAULT 'default',
  hash TEXT UNIQUE  -- ป้องกัน duplicate
);
