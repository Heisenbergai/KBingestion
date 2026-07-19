-- ============================================================
-- HIREFLOW MULTI-TENANT SCHEMA
-- Run this in Lovable's Supabase SQL Editor
-- ============================================================

-- ── 1. PLANS (defined by you, not editable by tenants) ────────────────────────
create table if not exists plans (
  id                          text primary key,  -- 'starter_50', 'growth_100'
  name                        text not null,
  price_inr_monthly           integer not null,
  price_inr_annual            integer not null,
  max_users                   integer not null,
  max_bots                    integer not null,
  max_knowledge_folders       integer not null,
  daily_bot_queries           integer not null,
  monthly_bot_queries         integer not null,
  daily_ai_search_queries     integer not null,
  monthly_presentations       integer not null,
  monthly_course_generations  integer not null,
  storage_type                text not null,     -- 'standard', 'extended'
  support_sla_hours           integer not null,
  is_active                   boolean default true,
  created_at                  timestamptz default now()
);

-- Insert your two plans
insert into plans (
  id, name,
  price_inr_monthly, price_inr_annual,
  max_users, max_bots, max_knowledge_folders,
  daily_bot_queries, monthly_bot_queries,
  daily_ai_search_queries,
  monthly_presentations, monthly_course_generations,
  storage_type, support_sla_hours
) values
(
  'starter_50', '50-Employee Plan',
  8250, 99000,
  50, 5, 10,
  500, 15000,
  200,
  20, 10,
  'standard', 48
),
(
  'growth_100', '100-Employee Plan',
  14917, 179000,
  100, 15, 30,
  1200, 36000,
  500,
  60, 30,
  'extended', 24
)
on conflict (id) do nothing;


-- ── 2. WORKSPACES (one per company) ───────────────────────────────────────────
create table if not exists workspaces (
  id              uuid primary key default gen_random_uuid(),
  name            text not null,                 -- "Acme Corp"
  slug            text unique not null,          -- "acme-corp"
  plan_id         text references plans(id) not null default 'starter_50',
  owner_id        uuid references auth.users(id),
  logo_url        text,
  primary_color   text default '#1E2761',
  is_active       boolean default true,
  is_suspended    boolean default false,
  suspension_reason text,
  billing_email   text,
  plan_started_at timestamptz default now(),
  plan_expires_at timestamptz,                   -- null = no expiry
  created_at      timestamptz default now(),
  updated_at      timestamptz default now()
);

create index if not exists workspaces_owner_idx on workspaces(owner_id);
create index if not exists workspaces_slug_idx  on workspaces(slug);


-- ── 3. WORKSPACE MEMBERS (users within a workspace) ───────────────────────────
create table if not exists workspace_members (
  id            uuid primary key default gen_random_uuid(),
  workspace_id  uuid references workspaces(id) on delete cascade not null,
  user_id       uuid references auth.users(id) on delete cascade not null,
  role          text not null default 'employee'
                check (role in ('owner','admin','hr','manager','employee')),
  invited_by    uuid references auth.users(id),
  status        text not null default 'active'
                check (status in ('active','invited','suspended')),
  department    text,
  job_title     text,
  created_at    timestamptz default now(),
  unique(workspace_id, user_id)
);

create index if not exists workspace_members_workspace_idx on workspace_members(workspace_id);
create index if not exists workspace_members_user_idx      on workspace_members(user_id);


-- ── 4. WORKSPACE USAGE (live counters, reset monthly/daily) ───────────────────
create table if not exists workspace_usage (
  id                              uuid primary key default gen_random_uuid(),
  workspace_id                    uuid references workspaces(id) on delete cascade unique not null,

  -- Daily counters (reset each day at midnight)
  daily_bot_queries_used          integer default 0,
  daily_ai_search_queries_used    integer default 0,
  daily_reset_at                  date default current_date,

  -- Monthly counters (reset on billing anniversary)
  monthly_bot_queries_used        integer default 0,
  monthly_presentations_used      integer default 0,
  monthly_course_generations_used integer default 0,
  monthly_reset_at                date default date_trunc('month', now())::date,

  -- Current state
  current_users                   integer default 0,
  current_bots                    integer default 0,
  current_folders                 integer default 0,

  -- Blocking state
  is_ai_blocked                   boolean default false,  -- true when daily limit hit
  block_reason                    text,

  updated_at                      timestamptz default now()
);

create index if not exists workspace_usage_workspace_idx on workspace_usage(workspace_id);


-- ── 5. USAGE LOG (audit trail for every AI call) ──────────────────────────────
create table if not exists usage_log (
  id            uuid primary key default gen_random_uuid(),
  workspace_id  uuid references workspaces(id) on delete cascade not null,
  user_id       uuid references auth.users(id),
  feature       text not null
                check (feature in (
                  'ai_search', 'chatbot_internal', 'chatbot_external',
                  'presentation', 'course_generation', 'video_generation',
                  'document_ingestion'
                )),
  query_count   integer default 1,         -- number of queries this call represents
  created_at    timestamptz default now()
);

create index if not exists usage_log_workspace_idx on usage_log(workspace_id);
create index if not exists usage_log_created_idx   on usage_log(created_at);
create index if not exists usage_log_feature_idx   on usage_log(workspace_id, feature);


-- ── 6. WORKSPACE INVITATIONS (pending email invites) ──────────────────────────
create table if not exists workspace_invitations (
  id            uuid primary key default gen_random_uuid(),
  workspace_id  uuid references workspaces(id) on delete cascade not null,
  email         text not null,
  role          text not null default 'employee'
                check (role in ('admin','hr','manager','employee')),
  invited_by    uuid references auth.users(id),
  token         text unique default encode(gen_random_bytes(24), 'hex'),
  status        text default 'pending'
                check (status in ('pending','accepted','expired')),
  expires_at    timestamptz default (now() + interval '7 days'),
  created_at    timestamptz default now()
);

create index if not exists workspace_invitations_token_idx on workspace_invitations(token);
create index if not exists workspace_invitations_email_idx on workspace_invitations(email);


-- ── 7. SUPER ADMIN TABLE (you — cross-workspace access) ───────────────────────
create table if not exists super_admins (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid references auth.users(id) unique not null,
  created_at timestamptz default now()
);


-- ── 8. HELPER FUNCTION: auto-create usage row when workspace is created ────────
create or replace function create_workspace_usage()
returns trigger language plpgsql as $$
begin
  insert into workspace_usage (workspace_id)
  values (new.id)
  on conflict (workspace_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_workspace_created on workspaces;
create trigger on_workspace_created
  after insert on workspaces
  for each row execute function create_workspace_usage();


-- ── 9. HELPER FUNCTION: check and enforce daily limits ────────────────────────
create or replace function check_and_increment_usage(
  p_workspace_id  uuid,
  p_feature       text,
  p_user_id       uuid default null
)
returns jsonb language plpgsql as $$
declare
  v_usage     workspace_usage%rowtype;
  v_plan      plans%rowtype;
  v_workspace workspaces%rowtype;
  v_allowed   boolean := true;
  v_reason    text    := null;
begin
  -- Get workspace + plan
  select w.*, p.*
  into v_workspace
  from workspaces w
  where w.id = p_workspace_id;

  select * into v_plan
  from plans
  where id = v_workspace.plan_id;

  -- Get usage, reset daily counters if needed
  select * into v_usage
  from workspace_usage
  where workspace_id = p_workspace_id;

  -- Reset daily counters if date has changed
  if v_usage.daily_reset_at < current_date then
    update workspace_usage set
      daily_bot_queries_used       = 0,
      daily_ai_search_queries_used = 0,
      daily_reset_at               = current_date,
      is_ai_blocked                = false,
      block_reason                 = null,
      updated_at                   = now()
    where workspace_id = p_workspace_id;
    -- Refresh
    select * into v_usage from workspace_usage where workspace_id = p_workspace_id;
  end if;

  -- Reset monthly counters if month has changed
  if v_usage.monthly_reset_at < date_trunc('month', now())::date then
    update workspace_usage set
      monthly_bot_queries_used        = 0,
      monthly_presentations_used      = 0,
      monthly_course_generations_used = 0,
      monthly_reset_at                = date_trunc('month', now())::date,
      updated_at                      = now()
    where workspace_id = p_workspace_id;
    select * into v_usage from workspace_usage where workspace_id = p_workspace_id;
  end if;

  -- Check workspace suspension
  if v_workspace.is_suspended then
    return jsonb_build_object(
      'allowed', false,
      'reason', 'workspace_suspended',
      'message', 'Your workspace has been suspended. Please contact support.'
    );
  end if;

  -- Enforce limits per feature
  case p_feature
    when 'ai_search' then
      if v_usage.daily_ai_search_queries_used >= v_plan.daily_ai_search_queries then
        v_allowed := false;
        v_reason  := 'daily_ai_search_limit';
      else
        update workspace_usage set
          daily_ai_search_queries_used = daily_ai_search_queries_used + 1,
          updated_at = now()
        where workspace_id = p_workspace_id;
        -- Log usage
        insert into usage_log (workspace_id, user_id, feature)
        values (p_workspace_id, p_user_id, 'ai_search');
      end if;

    when 'chatbot_internal', 'chatbot_external' then
      if v_usage.daily_bot_queries_used >= v_plan.daily_bot_queries then
        v_allowed := false;
        v_reason  := 'daily_bot_limit';
      elsif v_usage.monthly_bot_queries_used >= v_plan.monthly_bot_queries then
        v_allowed := false;
        v_reason  := 'monthly_bot_limit';
      else
        update workspace_usage set
          daily_bot_queries_used    = daily_bot_queries_used + 1,
          monthly_bot_queries_used  = monthly_bot_queries_used + 1,
          updated_at                = now()
        where workspace_id = p_workspace_id;
        insert into usage_log (workspace_id, user_id, feature)
        values (p_workspace_id, p_user_id, p_feature);
      end if;

    when 'presentation' then
      if v_usage.monthly_presentations_used >= v_plan.monthly_presentations then
        v_allowed := false;
        v_reason  := 'monthly_presentation_limit';
      else
        update workspace_usage set
          monthly_presentations_used = monthly_presentations_used + 1,
          updated_at                 = now()
        where workspace_id = p_workspace_id;
        insert into usage_log (workspace_id, user_id, feature)
        values (p_workspace_id, p_user_id, 'presentation');
      end if;

    when 'course_generation' then
      if v_usage.monthly_course_generations_used >= v_plan.monthly_course_generations then
        v_allowed := false;
        v_reason  := 'monthly_course_limit';
      else
        update workspace_usage set
          monthly_course_generations_used = monthly_course_generations_used + 1,
          updated_at                      = now()
        where workspace_id = p_workspace_id;
        insert into usage_log (workspace_id, user_id, feature)
        values (p_workspace_id, p_user_id, 'course_generation');
      end if;

    else
      -- document_ingestion, video_generation — log only, no hard limit
      insert into usage_log (workspace_id, user_id, feature)
      values (p_workspace_id, p_user_id, p_feature);
  end case;

  if not v_allowed then
    return jsonb_build_object(
      'allowed', false,
      'reason', v_reason,
      'message', case v_reason
        when 'daily_bot_limit'          then 'Daily bot query limit reached. Resets tomorrow.'
        when 'monthly_bot_limit'        then 'Monthly bot query limit reached. Upgrade or wait for next billing cycle.'
        when 'daily_ai_search_limit'    then 'Daily AI Search limit reached. Resets tomorrow.'
        when 'monthly_presentation_limit' then 'Monthly presentation limit reached. Upgrade your plan.'
        when 'monthly_course_limit'     then 'Monthly course generation limit reached. Upgrade your plan.'
        else 'Usage limit reached.'
      end,
      'upgrade_available', true
    );
  end if;

  return jsonb_build_object('allowed', true);
end;
$$;
