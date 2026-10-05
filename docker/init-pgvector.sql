-- Extensions required by the DOC_ONLY_RAG postgres provider.
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector: vector columns, <=> / <#> distance, HNSW/IVFFlat
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- trigram similarity, helps fuzzy hybrid search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- optional uuid generation for chunk ids

-- Sanity check (visible in container logs on first start).
SELECT extname, extversion FROM pg_extension ORDER BY extname;
