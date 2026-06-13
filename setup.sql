-- STEP 1: Enable the vector extension
create extension if not exists vector;

-- STEP 2: Create the table that stores all chunks and their vectors
create table if not exists document_chunks (
  id            uuid primary key default gen_random_uuid(),
  document_id   uuid,         -- links back to asset_documents table in Lovable
  asset_id      uuid,         -- which asset/folder this belongs to
  content       text not null, -- the actual readable text of this chunk
  embedding     vector(1024), -- the meaning of this chunk as numbers (Voyage AI)
  chunk_index   int,          -- position of this chunk within the document
  metadata      jsonb default '{}', -- file name, page info, etc.
  created_at    timestamp with time zone default now()
);

-- STEP 3: Create an index so similarity search is fast
create index if not exists document_chunks_embedding_idx
  on document_chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- STEP 4: Create the search function
-- This is what FastAPI calls when a user asks a question
create or replace function match_chunks(
  query_embedding  vector(1024),  -- the question converted to numbers
  match_count      int default 5, -- how many chunks to return
  filter_asset_id  uuid default null -- optional: search only within one asset
)
returns table (
  id          uuid,
  content     text,
  metadata    jsonb,
  similarity  float
)
language plpgsql
as $$
begin
  return query
  select
    dc.id,
    dc.content,
    dc.metadata,
    1 - (dc.embedding <=> query_embedding) as similarity
  from document_chunks dc
  where
    filter_asset_id is null or dc.asset_id = filter_asset_id
  order by dc.embedding <=> query_embedding
  limit match_count;
end;
$$;
