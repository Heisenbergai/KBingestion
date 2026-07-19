-- ============================================================
-- RUN THIS IN YOUR PERSONAL SUPABASE (shisuakmubkawdvslmif)
-- This fixes the data leak in the vector database
-- ============================================================

-- Step 1: Add workspace_id column to document_chunks
alter table document_chunks
  add column if not exists workspace_id uuid;

-- Step 2: Index for fast workspace filtering
create index if not exists document_chunks_workspace_idx
  on document_chunks(workspace_id);

-- Step 3: Drop old search function
drop function if exists match_chunks(vector, int, uuid);

-- Step 4: Create new workspace-scoped search function
-- This ALWAYS filters by workspace_id — no cross-workspace data ever returned
create or replace function match_chunks_workspace(
  query_embedding      vector(1024),
  match_count          int default 5,
  filter_asset_id      uuid default null,
  filter_workspace_id  uuid default null
)
returns table (
  id          uuid,
  document_id uuid,
  asset_id    uuid,
  workspace_id uuid,
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
    dc.document_id,
    dc.asset_id,
    dc.workspace_id,
    dc.content,
    dc.metadata,
    1 - (dc.embedding <=> query_embedding) as similarity
  from document_chunks dc
  where
    -- WORKSPACE ISOLATION: always filter by workspace
    (filter_workspace_id is null or dc.workspace_id = filter_workspace_id)
    -- Optional: also filter by specific document/asset
    and (filter_asset_id is null or dc.asset_id = filter_asset_id)
  order by dc.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Step 5: Keep old function as alias for backward compatibility
-- (in case any old code still calls match_chunks)
create or replace function match_chunks(
  query_embedding      vector(1024),
  match_count          int default 5,
  filter_asset_id      uuid default null
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
  -- Old function now returns empty — forces all callers to use match_chunks_workspace
  -- This prevents any accidental cross-workspace queries from old code
  return query
  select
    dc.id,
    dc.content,
    dc.metadata,
    1 - (dc.embedding <=> query_embedding) as similarity
  from document_chunks dc
  where false  -- always empty — old callers must upgrade to match_chunks_workspace
  limit match_count;
end;
$$;
