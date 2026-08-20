"""
Company Brain — connector framework (Phase 2).

The shared layer under every integration (Slack first, then Google/Zoom):
  - token encryption (Fernet)
  - raw-item capture into ingest_items (dedup)
  - THE FILTRATION ENGINE: the GBrain "signal detector" equivalent — batches
    raw messages into conversations, asks the LLM which contain durable company
    knowledge (vs. noise), and DISTILLS keepers into clean knowledge_notes.
    Only distilled notes get embedded — raw chat logs never pollute the brain.
  - note → document_chunks pipeline (tier-3 by default, so official docs still
    outrank chat in hybrid search)
  - generic REST routes for the frontend (list connections, list/delete notes,
    trigger a sync)

Provider-specific code (OAuth, message fetching) lives in connector_*.py files.
Railway owns all of this end to end — no Lovable DB access required.
"""
import os
import json
import time
import threading
import ai
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from auth import AuthContext, current_user
from pydantic import BaseModel
from typing import Optional, Callable
from supabase import create_client
from dotenv import load_dotenv

from ingest import chunk_text, embed_chunks, classify_document

load_dotenv()

router = APIRouter()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# ── Token encryption ────────────────────────────────────────────────────────────
# CONNECTOR_ENCRYPTION_KEY is a Fernet key (generate once:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# ) stored in Railway env. Without it we refuse to store OAuth tokens — never
# persist third-party access tokens in plaintext.
_FERNET = None
def _fernet():
    global _FERNET
    if _FERNET is None:
        from cryptography.fernet import Fernet
        key = os.getenv("CONNECTOR_ENCRYPTION_KEY")
        if not key:
            raise HTTPException(
                status_code=500,
                detail="CONNECTOR_ENCRYPTION_KEY is not set — cannot securely store connector tokens.",
            )
        _FERNET = Fernet(key.encode() if isinstance(key, str) else key)
    return _FERNET


def encrypt_secret(value: str) -> str:
    if not value:
        return ""
    return _fernet().encrypt(value.encode()).decode()


def decrypt_secret(value: str) -> str:
    if not value:
        return ""
    return _fernet().decrypt(value.encode()).decode()


# ── Per-workspace provider app credentials ──────────────────────────────────────
# Single-tenant model (decided 2026-07-26): each customer registers their OWN
# Slack/Google/Microsoft/Zoom app in their own tenant and pastes its client_id/
# client_secret here, rather than every customer sharing one Knova-owned app —
# that would need Google OAuth verification + an annual CASA assessment,
# Microsoft publisher verification, and Teams Protected-APIs approval, none of
# which are survivable pre-revenue. See 09_company_brain_roadmap.md.

def get_provider_credentials(workspace_id: str, provider: str) -> Optional[dict]:
    """
    Returns {"client_id", "client_secret", "webhook_secret"} for this
    workspace+provider, or None if the workspace hasn't registered its own
    app for that provider yet. webhook_secret is "" if not set (not every
    provider has one — it's only needed by webhook-verified connectors).

    Callers that need a fallback (e.g. keeping the one pre-existing Slack
    connection working through this rollout) should fall back to the
    provider's own env vars themselves when this returns None — that decision
    is provider-specific, not made here.
    """
    row = supabase.table("provider_credentials") \
        .select("client_id, client_secret_enc, webhook_secret_enc") \
        .eq("workspace_id", workspace_id).eq("provider", provider).execute().data
    if not row:
        return None
    return {
        "client_id":      row[0]["client_id"],
        "client_secret":  decrypt_secret(row[0]["client_secret_enc"]),
        "webhook_secret": decrypt_secret(row[0]["webhook_secret_enc"]) if row[0].get("webhook_secret_enc") else "",
    }


def save_provider_credentials(workspace_id: str, provider: str, client_id: str,
                              client_secret: str, webhook_secret: str = "",
                              created_by: str = "") -> None:
    """Upserts one (workspace, provider) app credential set, encrypted at rest."""
    row = {
        "workspace_id":      workspace_id,
        "provider":          provider,
        "client_id":         client_id,
        "client_secret_enc": encrypt_secret(client_secret),
        "created_by":        created_by,
        "updated_at":        datetime.now(timezone.utc).isoformat(),
    }
    if webhook_secret:
        row["webhook_secret_enc"] = encrypt_secret(webhook_secret)
    supabase.table("provider_credentials").upsert(
        row, on_conflict="workspace_id,provider"
    ).execute()


def has_provider_credentials(workspace_id: str, provider: str) -> bool:
    row = supabase.table("provider_credentials").select("id") \
        .eq("workspace_id", workspace_id).eq("provider", provider).execute().data
    return bool(row)


# ── Multi-integration management (Phase 2, 2026-08-16) ──────────────────────────
# A workspace may have MULTIPLE connections for the SAME provider (e.g. two
# Slack teams, two Google accounts) -- the real unique index on `connections`
# is (workspace_id, provider, external_team_id), confirmed live
# (connections_provider_team_idx), so this was already schema-legal; the
# gap was application code assuming one-per-provider in several places.
# `connection_id` is the only real identity from here on -- workspace_id is
# used solely to verify OWNERSHIP of a given connection_id, never as part of
# how a connection is looked up or resolved.

MAX_ACTIVE_CONNECTIONS_PER_WORKSPACE = 10
# Statuses that occupy a slot. 'revoked' (soft-deleted via disconnect) never
# counts. 'pending' (mid-setup, OAuth not finished) and 'error' (broken,
# needs reconnect/disconnect) DO count -- they're real rows a user would
# need to explicitly disconnect to free up, per the locked requirement
# "if a pending/stuck connection counts temporarily, Disconnect must
# immediately free that slot" (DELETE /connections/{id} already does,
# unconditionally, regardless of status -- see disconnect() below).
ACTIVE_CONNECTION_STATUSES = ("active", "error", "pending")


def count_active_connections(workspace_id: str) -> int:
    res = supabase.table("connections").select("id", count="exact") \
        .eq("workspace_id", workspace_id).in_("status", list(ACTIVE_CONNECTION_STATUSES)) \
        .execute()
    return res.count or 0


def create_pending_connection(workspace_id: str, provider: str,
                              display_name: Optional[str] = None,
                              connected_by: str = "") -> dict:
    """
    The row a brand-new "+ Add integration" click creates, BEFORE OAuth ever
    starts -- this is what makes Disconnect always available regardless of
    how far setup got (credentials saved but OAuth not started, popup
    opened, cancelled, failed, partially completed): there's a real
    connection_id to act on from the very first click, not just once OAuth
    happens to succeed.

    external_team_id is left NULL (not yet known -- OAuth hasn't resolved a
    real account) -- Postgres treats multiple NULLs as distinct under the
    unique index, so several pending rows for the same provider never
    collide with each other or with a real connected row.

    Raises HTTPException(400) if the workspace is already at the 10-active
    limit -- enforced HERE, server-side, not just in the UI.
    """
    if count_active_connections(workspace_id) >= MAX_ACTIVE_CONNECTIONS_PER_WORKSPACE:
        raise HTTPException(
            status_code=400,
            detail=f"This workspace already has {MAX_ACTIVE_CONNECTIONS_PER_WORKSPACE} active "
                   f"integrations, the maximum. Disconnect one before adding another.",
        )
    row = {
        "workspace_id":  workspace_id,
        "provider":      provider,
        "external_team_id": None,
        "status":        "pending",
        "config":        {},
        "connected_by":  connected_by,
        "display_name":  (display_name or "").strip() or None,
    }
    result = supabase.table("connections").insert(row).execute()
    return result.data[0]


def get_connection_for_workspace(connection_id: str, workspace_id: str) -> Optional[dict]:
    """
    THE cross-tenant verification helper: a connection_id alone is never
    trusted -- every operation that names one must also prove it belongs to
    the CALLER'S authenticated workspace_id (never a client-supplied one
    used for anything but that proof). Returns None both when the
    connection doesn't exist AND when it belongs to a different workspace --
    deliberately identical, same "never confirm existence to a non-member"
    principle auth.py's assert_workspace already uses.
    """
    rows = supabase.table("connections").select("*").eq("id", connection_id).execute().data
    if not rows or rows[0]["workspace_id"] != workspace_id:
        return None
    return rows[0]


def rename_connection(connection_id: str, workspace_id: str, display_name: Optional[str]) -> dict:
    """Renames ONLY this connection -- display_name is purely descriptive
    metadata (never used for auth/routing/provider identity), so this is a
    plain single-row update once ownership is verified."""
    conn = get_connection_for_workspace(connection_id, workspace_id)
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found.")
    name = (display_name or "").strip() or None
    supabase.table("connections").update({"display_name": name}).eq("id", connection_id).execute()
    return {**conn, "display_name": name}


# ── OAuth state (workspace_id + user_id, Fernet-signed so it can't be forged) ──
# Shared by every OAuth connector (connector_slack.py, connector_google.py, …) —
# the shape (which workspace + which user started the flow, plus a timestamp)
# is identical regardless of provider.
def encode_oauth_state(workspace_id: str, user_id: str, extra: Optional[dict] = None) -> str:
    """extra: optional additive fields merged into the signed state -- e.g.
    connector_google uses {"surfaces": [...]} to carry which Google Workspace
    surfaces were requested through to the callback, since Google's redirect
    URI is fixed and can't carry a custom query param of its own. Backward
    compatible: every existing caller (Slack, Zoom) omits it and gets exactly
    the previous {"w", "u", "t"} shape."""
    payload = {"w": workspace_id, "u": user_id, "t": int(time.time())}
    if extra:
        payload.update(extra)
    return encrypt_secret(json.dumps(payload))


def decode_oauth_state(state: str) -> dict:
    from fastapi import HTTPException
    try:
        return json.loads(decrypt_secret(state))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")


def get_provider_credentials_by_external_team(provider: str, external_team_id: str) -> Optional[dict]:
    """
    Resolves a workspace's provider credentials starting from the external
    team/tenant id instead of workspace_id — what an inbound webhook has to
    work with (Slack's event payload carries team_id, not our workspace_id).
    Used to verify a webhook against the RIGHT customer's signing secret
    before trusting anything else in the payload.

    Multi-integration management (2026-08-16): the same external_team_id can
    now legitimately belong to connections in MORE THAN ONE workspace (the
    real, already-documented case: one Slack team connected into both "Test
    company 1" and "Magic Smart Homes"). Picking an arbitrary row here would
    be non-deterministic and could verify against the wrong workspace's
    secret -- fails safe (signature check then fails, request rejected) but
    is not something to leave to chance. Fails closed (returns None) when
    more than one DISTINCT workspace has an active connection for this
    external_team_id -- same "never guess" contract as
    connector_slack._resolve_slack_connection.
    """
    conn = supabase.table("connections").select("workspace_id") \
        .eq("provider", provider).eq("external_team_id", external_team_id) \
        .eq("status", "active").execute().data
    if not conn:
        return None
    workspaces = {c["workspace_id"] for c in conn}
    if len(workspaces) > 1:
        print(f"[connections] AMBIGUOUS credential lookup: provider={provider} "
              f"external_team_id={external_team_id} matches {len(workspaces)} workspaces — refusing to guess.")
        return None
    return get_provider_credentials(conn[0]["workspace_id"], provider)


# ── In-memory sync job status (same pattern as ingest) ──────────────────────────
SYNC_JOBS: dict[str, dict] = {}


# ── Raw item capture ────────────────────────────────────────────────────────────

def save_ingest_items(workspace_id: str, connection_id: str, provider: str,
                      items: list[dict]) -> int:
    """
    Inserts normalized raw items, skipping duplicates (unique on
    connection_id+external_id). Returns how many NEW items were stored.
    items: [{external_id, kind, raw}]
    """
    if not items:
        return 0
    rows = [{
        "workspace_id":  workspace_id,
        "connection_id": connection_id,
        "provider":      provider,
        "external_id":   it["external_id"],
        "kind":          it.get("kind", "message"),
        "raw":           it["raw"],
        "status":        "pending",
    } for it in items]
    stored = 0
    # upsert with ignore-duplicates so re-running backfill is safe
    for i in range(0, len(rows), 200):
        batch = rows[i:i + 200]
        try:
            res = supabase.table("ingest_items").upsert(
                batch, on_conflict="connection_id,external_id", ignore_duplicates=True
            ).execute()
            stored += len(res.data or [])
        except Exception as e:
            print(f"[connectors] item upsert error: {e}")
    return stored


# ── THE FILTRATION ENGINE ───────────────────────────────────────────────────────

def batch_conversations(items: list[dict]) -> list[list[dict]]:
    """
    Groups raw message items into conversation units for classification:
      - messages sharing a thread_ts stay together (a thread = one topic)
      - remaining standalone messages in a channel are grouped into rolling
        windows of up to 12 messages
    Each returned batch is a list of the original item dicts (with .raw).
    """
    threads: dict[str, list[dict]] = {}
    loose:   dict[str, list[dict]] = {}   # keyed by channel

    for it in items:
        raw = it.get("raw", {})
        thread_ts = raw.get("thread_ts")
        channel   = raw.get("channel", "unknown")
        if thread_ts:
            threads.setdefault(f"{channel}:{thread_ts}", []).append(it)
        else:
            loose.setdefault(channel, []).append(it)

    batches: list[list[dict]] = [v for v in threads.values() if v]
    for channel, msgs in loose.items():
        msgs.sort(key=lambda m: m.get("raw", {}).get("ts", ""))
        for i in range(0, len(msgs), 12):
            batches.append(msgs[i:i + 12])
    return batches


def _format_batch(batch: list[dict]) -> tuple[str, str, list[int]]:
    """
    Renders a batch as an INDEXED transcript ("[0] alice: ...", "[1] bob:
    ...") the classifier can cite specific lines from, plus index_map: the
    printed index i refers to batch[index_map[i]] -- NOT necessarily i
    itself, since messages with empty text are skipped entirely (nothing
    for a citation to point at) without shifting anything already printed.
    Returns (transcript, channel, index_map).
    """
    lines, channel, index_map = [], "unknown", []
    for pos, it in enumerate(batch):
        raw = it.get("raw", {})
        channel = raw.get("channel_name") or raw.get("channel", channel)
        who = raw.get("user_name") or raw.get("user", "someone")
        text = (raw.get("text") or "").strip()
        if text:
            idx = len(index_map)
            lines.append(f"[{idx}] {who}: {text}")
            index_map.append(pos)
    return "\n".join(lines), channel, index_map


CLASSIFY_SYSTEM = """You are the filter that decides what enters a company's permanent knowledge base.
You will see a NUMBERED workplace conversation window -- each line is prefixed with its index in
brackets, e.g. "[0] alice: ...". The window may contain ZERO, ONE, or MULTIPLE independent pieces
of durable knowledge, mixed with noise -- messages in the same window are NOT guaranteed to be
about the same topic just because they were posted close together.

For each GENUINELY independent, durable, REUSABLE piece of company knowledge that a colleague
might search for later -- a decision made, a process or how-to, an announcement, a factual answer,
or a policy -- produce ONE item. Casual chatter, greetings, logistics ("running 5 min late"),
reactions, and banter are NOISE -- do not produce an item for them.

Do NOT invent a connection between unrelated messages just because they're in the same window --
combine messages into ONE item only when they are genuinely the same topic/discussion. Keep
genuinely separate topics as separate items, each with its own entry. Preserve qualifiers and
uncertainty -- an opinion or suggestion ("I think we should...", "maybe we could...") is NOT a
settled decision; never rewrite it as one, and do not treat it as durable knowledge unless the
window shows it was actually decided.

If worth keeping, rewrite each item as a clean, standalone knowledge note written in the third
person as settled fact -- NOT "someone said". Include the concrete substance (numbers, names,
decisions).

Every item MUST cite exactly which message indices (from the numbered list above) support it, in
"source_message_indices" -- only indices that genuinely contain that item's content, never a
nearby index just because it's close by. An item with no genuinely supporting index must not be
produced at all.

Respond ONLY with valid JSON, no markdown fences:
{"items": [
  {"title": "concise, searchable title",
   "note": "1-4 sentences of standalone knowledge",
   "category": "decision" | "process" | "announcement" | "fact" | "qa",
   "participants": ["first names of key people involved"],
   "source_message_indices": [0, 2]}
]}
If nothing in the window is worth keeping: {"items": []}"""


def classify_batch(transcript: str, channel: str, index_map: list[int],
                   workspace_id: Optional[str] = None) -> list[dict]:
    """
    Runs one conversation WINDOW through the LLM filter. Returns a list of
    note dicts (possibly empty -- pure noise, an empty transcript, or a
    call/parse failure all fail safe to zero items, never a guess).

    Each returned item carries "source_batch_positions": real positions in
    the ORIGINAL batch list (already translated through index_map), not the
    printed transcript indices -- callers never have to re-derive the
    mapping themselves.

    VALIDATION, not trust (2026-08-15 provenance fix): every
    source_message_indices value the model returns is checked against
    index_map's real bounds before being used. An item with a missing,
    empty, non-list, non-integer, or out-of-range index is DROPPED
    ENTIRELY -- never partially attached to the wrong messages, never
    guessed at. This is what "the model must not create a note without
    identifying which messages support it" is enforced as code, not just
    a prompt instruction the model might ignore. One bad item in a
    multi-item response does not discard the others.
    """
    if not transcript.strip():
        return []
    try:
        verdict = ai.chat_json(
            messages=[{"role": "user",
                       "content": f"Channel: #{channel}\n\nConversation:\n{transcript}"}],
            system=CLASSIFY_SYSTEM, max_tokens=1200, temperature=0.2,
            workspace_id=workspace_id, feature="filtration",
        )
    except Exception as e:
        print(f"[filtration] classify failed (discarding batch, non-fatal): {e}")
        return []
    if not isinstance(verdict, dict):
        return []
    raw_items = verdict.get("items")
    if not isinstance(raw_items, list):
        return []

    n = len(index_map)
    results: list[dict] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        if not raw_item.get("note") or not raw_item.get("title"):
            continue
        indices = raw_item.get("source_message_indices")
        if not isinstance(indices, list) or not indices:
            continue  # no cited support at all -- never create a note without attribution
        seen: set[int] = set()
        positions: list[int] = []
        valid = True
        for idx in indices:
            # isinstance(idx, bool) excluded deliberately: bool is a subclass
            # of int in Python, and True/False must never be silently
            # treated as 1/0 array indices.
            if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx >= n:
                valid = False
                break
            if idx in seen:
                continue
            seen.add(idx)
            positions.append(index_map[idx])
        if not valid or not positions:
            continue  # malformed/out-of-range citation -- drop THIS item, not the whole batch
        results.append({
            "category":     raw_item.get("category", "fact"),
            "title":        str(raw_item["title"])[:200],
            "body":         str(raw_item["note"]),
            "participants": [str(p) for p in (raw_item.get("participants") or [])][:10],
            "source_batch_positions": positions,
        })
    return results


MEETING_SYSTEM = """You are the filter that decides what enters a company's permanent knowledge base.
You will see a FULL MEETING TRANSCRIPT (Zoom). Unlike a short chat snippet, a meeting that
actually took place almost always has SOME durable value — decisions made, action items assigned,
or a discussion worth a written record — so default to keeping it. Only discard if the transcript
is empty, corrupted, or pure small talk with zero substantive content (e.g. a call that never
really started, or one that's entirely "can you hear me / one sec").

If worth keeping, distill it into a structured note — NOT a transcript dump:
  - decisions: concrete decisions made, if any
  - action_items: who is doing what, if assigned
  - summary: 2-4 sentences of what was discussed, in third person as settled fact

Respond ONLY with valid JSON, no markdown fences:
{
  "worth_keeping": true,
  "title": "concise, searchable title (the meeting's actual topic, not \"Meeting Notes\")",
  "decisions": ["decision 1", "..."],
  "action_items": ["who does what", "..."],
  "summary": "2-4 sentences of discussion summary",
  "participants": ["first names of key people involved"]
}
If truly nothing of value: {"worth_keeping": false}"""


def distill_meeting_transcript(transcript: str, meeting_title: str,
                               workspace_id: Optional[str] = None) -> Optional[dict]:
    """
    The meeting-transcript equivalent of classify_batch() — used by
    connector_zoom.py (any future meeting-recording connector reuses this
    too), since a full meeting transcript needs a
    differently-shaped prompt than a 12-message chat window: richer output
    (decisions / action items / summary, not one terse sentence) and a bias
    toward KEEPING rather than discarding, since a meeting that happened
    almost always has some record value where idle chat usually doesn't.

    Returns a note dict compatible with create_note_and_embed(), or None if
    the transcript is empty/unparseable/genuinely worthless (fail safe by
    discarding, same convention as classify_batch).
    """
    if not transcript.strip():
        return None
    try:
        verdict = ai.chat_json(
            messages=[{"role": "user",
                       "content": f"Meeting: {meeting_title}\n\nTranscript:\n{transcript}"}],
            system=MEETING_SYSTEM, max_tokens=800, temperature=0.2,
            workspace_id=workspace_id, feature="filtration",
        )
    except Exception as e:
        print(f"[filtration] meeting distillation failed (discarding): {e}")
        return None
    if not isinstance(verdict, dict) or not verdict.get("worth_keeping"):
        return None
    if not verdict.get("title"):
        return None

    parts = []
    if verdict.get("summary"):
        parts.append(verdict["summary"])
    if verdict.get("decisions"):
        parts.append("Decisions:\n" + "\n".join(f"- {d}" for d in verdict["decisions"]))
    if verdict.get("action_items"):
        parts.append("Action items:\n" + "\n".join(f"- {a}" for a in verdict["action_items"]))
    body = "\n\n".join(parts).strip()
    if not body:
        return None

    return {
        "category":     "meeting",
        "title":        str(verdict["title"])[:200],
        "body":         body,
        "participants": [str(p) for p in (verdict.get("participants") or [])][:10],
    }


# Same ranking every retrieval/classification axis in this project treats as
# an ordered ladder (see Phase C's locked sensitivity spec) — used ONLY to
# decide whether an automated classification is allowed to move sensitivity,
# never to rank authority/doc_class/lifecycle (those are not access-control
# axes and have no "more/less restrictive" ordering).
_SENSITIVITY_RANK = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
_SAFE_BASELINE_SENSITIVITY = "internal"


def _classify_connector_note(title: str, body: str, source_type: str,
                             workspace_id: Optional[str]) -> dict:
    """
    Phase 2A (2026-08-15): reuses ingest.classify_document() — the SAME
    classifier uploaded documents already go through — instead of letting
    connector notes silently fall back to raw document_chunks column
    defaults (internal/working/null/active) regardless of actual content.
    No second classifier, no connector-specific prompt.

    SECURITY RULE, deliberately asymmetric between sensitivity and the other
    three axes: sensitivity starts at the safe baseline 'internal' and may
    ONLY be RAISED by the classifier, never lowered. This mirrors the
    upload flow's own raise-only/review-queue split (src/lib/knowledge.ts)
    — except a connector note has no prior human-chosen value to protect
    and no frontend review step in the loop, so there is no safe way to
    apply a WEAKER-than-baseline verdict at all; it is silently ignored
    rather than silently trusted. Concretely:
        classifier says public,       baseline internal -> effective internal (ignored, would lower)
        classifier says internal,     baseline internal -> effective internal (no change)
        classifier says confidential, baseline internal -> effective confidential (raised)
        classifier says restricted,   baseline internal -> effective restricted (raised)
    Authority/doc_class/lifecycle_status have no such asymmetry -- they are
    not access-control axes, so the classifier's result is used directly
    (or the safe default, if invalid/missing/classification failed).

    classify_document() itself never raises (see its own docstring) and
    already validates every field against its own allowed-value sets,
    falling back to internal/working/None/active on any failure or
    malformed output -- this function still wraps the call in its own
    try/except as defense in depth, so a change to that contract can never
    let a connector note escape with a broader access level than intended.
    """
    safe_defaults = {"sensitivity": _SAFE_BASELINE_SENSITIVITY, "authority": "working",
                     "doc_class": None, "lifecycle_status": "active"}
    try:
        classification = classify_document(
            title=title, raw_text=f"{title}\n\n{body}",
            source_type=source_type, workspace_id=workspace_id,
        )
    except Exception as e:
        print(f"[connectors] classification failed, using safe defaults (non-fatal): {e}")
        return safe_defaults

    proposed_sensitivity = classification.get("sensitivity")
    if (proposed_sensitivity in _SENSITIVITY_RANK
            and _SENSITIVITY_RANK[proposed_sensitivity] > _SENSITIVITY_RANK[_SAFE_BASELINE_SENSITIVITY]):
        sensitivity = proposed_sensitivity  # raised -- allowed
    else:
        sensitivity = _SAFE_BASELINE_SENSITIVITY  # same, lower, or invalid -- never auto-lowered

    return {
        "sensitivity":      sensitivity,
        "authority":        classification.get("authority") or safe_defaults["authority"],
        "doc_class":        classification.get("doc_class"),
        "lifecycle_status": classification.get("lifecycle_status") or safe_defaults["lifecycle_status"],
    }


def create_note_and_embed(workspace_id: str, connection_id: Optional[str], provider: str,
                          note: dict, source_type: str = "slack", source_tier: int = 3,
                          source_ref: str = None, occurred_at: str = None,
                          extra_metadata: Optional[dict] = None,
                          sources: Optional[list[dict]] = None) -> str:
    """
    Inserts a knowledge_note and embeds its body into document_chunks
    (document_id = note id, so its chunks are searchable via hybrid search
    and deletable together). Returns the note id.

    extra_metadata is merged into every chunk's metadata dict. First use:
    bot_learning.py stamps {"learned_for_bot_id": bot_id} on a note created
    from an admin's answer to an escalated question — a note has no natural
    knowledge_folders placement, so chatbot.py's folder-scope filter checks
    this instead to keep a bot-specific learned answer retrievable regardless
    of that bot's folder scope. bot_learning.py also does its own explicit
    sensitivity UPDATE on both tables right after calling this (an admin's
    deliberate choice) — that still runs afterward and still wins over the
    classification below, unchanged by this function.

    Phase 2A: classifies via _classify_connector_note() (see its docstring
    for the full sensitivity-never-auto-lowers rule) so sensitivity/
    authority/doc_class/lifecycle_status reflect the note's real content
    instead of raw column defaults. The SAME effective values are written
    to knowledge_notes AND every document_chunks row -- they must never
    disagree, since knowledge_notes is what a future Library UI would show
    and document_chunks is what retrieval's SQL-side filtering actually reads.

    Phase 2B provenance fix (2026-08-15): `source_ref` alone is a SINGLE
    nullable column -- it stays populated for backward compatibility
    (bot_learning.py and any other single-source caller still gets exactly
    the old behavior, untouched), but it can only ever name ONE contributor.
    `sources` -- optional, additive, only used by callers that pass it
    (currently: run_filtration's Slack path) -- is a list of
    {channel_id, message_ts, thread_ts, source_ref, occurred_at} dicts, one
    per REAL message that actually supports this note (never the whole
    batch a message happened to be captured alongside), written to the new
    knowledge_note_sources table. A caller that omits `sources` (bot_learning,
    Zoom's single-transcript notes) creates zero rows there -- source_ref
    remains the only provenance for those, exactly as before this fix.
    """
    classification = _classify_connector_note(note["title"], note["body"], source_type, workspace_id)

    note_row = {
        "workspace_id":     workspace_id,
        "connection_id":    connection_id,
        "provider":         provider,
        "source_type":      source_type,
        "source_tier":      source_tier,
        "category":         note.get("category"),
        "title":            note["title"],
        "body":             note["body"],
        "participants":     note.get("participants", []),
        "source_ref":       source_ref,
        "occurred_at":      occurred_at,
        "sensitivity":      classification["sensitivity"],
        "authority":        classification["authority"],
        "doc_class":        classification["doc_class"],
        "lifecycle_status": classification["lifecycle_status"],
    }
    res = supabase.table("knowledge_notes").insert(note_row).execute()
    note_id = res.data[0]["id"]

    if sources:
        source_rows = [{
            "note_id":       note_id,
            "workspace_id":  workspace_id,
            "provider":      provider,
            "source_type":   source_type,
            "connection_id": connection_id,
            "channel_id":    s.get("channel_id"),
            "message_ts":    s.get("message_ts"),
            "thread_ts":     s.get("thread_ts"),
            "source_ref":    s.get("source_ref"),
            "occurred_at":   s.get("occurred_at"),
        } for s in sources]
        supabase.table("knowledge_note_sources").insert(source_rows).execute()

    # Embed the note body into the searchable brain (tier 3 by default)
    full_text = f"{note['title']}\n\n{note['body']}"
    chunks = chunk_text(full_text) or [full_text]
    embeddings = embed_chunks(chunks, workspace_id=workspace_id, feature="filtration")
    rows = [{
        "document_id":  note_id,
        "asset_id":     note_id,
        "workspace_id": workspace_id,
        "content":      chunks[i],
        "embedding":    embeddings[i],
        "chunk_index":  i,
        "source_type":  source_type,
        "source_tier":  source_tier,
        "doc_date":     occurred_at,
        "sensitivity":      classification["sensitivity"],
        "authority":        classification["authority"],
        "doc_class":        classification["doc_class"],
        "lifecycle_status": classification["lifecycle_status"],
        "metadata": {
            "file_name":    note["title"],
            "chunk_index":  i,
            "total_chunks": len(chunks),
            "workspace_id": workspace_id,
            "source_type":  source_type,
            "note_id":      note_id,
            **(extra_metadata or {}),
        },
    } for i in range(len(chunks))]
    supabase.table("document_chunks").insert(rows).execute()
    return note_id


def run_filtration(workspace_id: str, connection_id: str, provider: str,
                   job: Optional[dict] = None,
                   resolve_permalink: Optional[Callable[[dict], Optional[str]]] = None,
                   on_note_created: Optional[Callable[[str, list[dict]], None]] = None) -> dict:
    """
    Processes all pending ingest_items for a connection:
    batch → classify (possibly multiple items per batch) → distill each
    KEEP item into its own note, attributed only to its real contributing
    messages → mark items. This is the step that turns raw chat into
    curated company knowledge.

    Phase 2B provenance fix (2026-08-15): a batch is a CONTEXT WINDOW, not
    a guarantee of one topic. classify_batch() now returns a LIST of items,
    each carrying which real batch positions support it
    (source_batch_positions, already validated -- see classify_batch's
    docstring). An ingest_item is marked 'noted' with a note_id ONLY if it
    was actually cited by that note's classification -- never because it
    merely shared a batch with something that got kept. Anything in the
    batch not cited by any KEEP item is marked 'discarded', exactly the
    same as before this fix for genuinely pure-noise batches.

    resolve_permalink(raw_message_dict) -> Optional[str], if given, is
    called ONLY for messages that actually end up contributing to a KEEP
    item (never for the whole batch, never for discarded messages) --
    provider-specific (currently Slack's real chat.getPermalink), passed in
    by the caller rather than looked up here, since this function is
    provider-agnostic and must not import a specific connector module.
    A None return (lookup failed, or no resolver given) is stored as-is --
    never fabricated into a guessed URL, and never blocks note creation.

    on_note_created(note_id, contributing_items) -> None, if given, fires
    right after a note is created, with the exact list of raw ingest_item
    dicts that contributed to it (the same `contributing` list used to
    build `sources` above -- each still carries its own untouched `.raw`).
    This is the hook Google Chat uses for Drive-reference resolution
    (2026-08-16 fix): it must scan the ORIGINAL raw message text, not the
    distilled note body, since the classifier may paraphrase a Drive URL
    out of the note entirely -- see connector_google.py's
    resolve_drive_references_in_text(). Optional and additive: omitted (the
    Slack/default path), behavior is byte-for-byte unchanged from before
    this parameter existed. A hook failure is caught and logged, never
    allowed to cost the note itself -- same non-fatal contract as
    resolve_permalink above.
    """
    pending = supabase.table("ingest_items").select("*") \
        .eq("connection_id", connection_id).eq("status", "pending") \
        .limit(2000).execute().data or []

    if job is not None:
        job["items_pending"] = len(pending)

    batches = batch_conversations(pending)
    notes_created = 0
    discarded = 0

    for bi, batch in enumerate(batches):
        transcript, channel, index_map = _format_batch(batch)
        items = classify_batch(transcript, channel, index_map, workspace_id=workspace_id)

        attributed_ids: set = set()
        for item in items:
            contributing = [batch[p] for p in item["source_batch_positions"]]
            item_ids = [it["id"] for it in contributing]

            sources = []
            for it in contributing:
                raw = it.get("raw", {})
                permalink = None
                if resolve_permalink is not None:
                    try:
                        permalink = resolve_permalink(raw)
                    except Exception as e:
                        # A permalink lookup failure must never cost the
                        # note itself -- real channel/ts metadata is kept
                        # regardless, source_ref just stays None instead of
                        # a fabricated URL.
                        print(f"[connectors] permalink lookup failed (non-fatal): {e}")
                sources.append({
                    "channel_id":  raw.get("channel"),
                    "message_ts":  raw.get("ts"),
                    "thread_ts":   raw.get("thread_ts"),
                    "source_ref":  permalink,
                    "occurred_at": raw.get("iso_ts"),
                })

            note_id = create_note_and_embed(
                workspace_id, connection_id, provider, item,
                source_type=provider if provider != "google_drive" else "document",
                source_tier=3 if provider == "slack" else 2,
                # Legacy single-value column: the first real contributor's
                # permalink, for backward-compat callers that only ever
                # read knowledge_notes.source_ref (see create_note_and_embed's
                # docstring) -- knowledge_note_sources is the complete record.
                source_ref=sources[0]["source_ref"] if sources else None,
                occurred_at=sources[0]["occurred_at"] if sources else None,
                sources=sources,
            )
            supabase.table("ingest_items").update(
                {"status": "noted", "note_id": note_id}
            ).in_("id", item_ids).execute()
            attributed_ids.update(item_ids)
            notes_created += 1

            if on_note_created is not None:
                try:
                    on_note_created(note_id, contributing)
                except Exception as e:
                    # Same non-fatal contract as resolve_permalink above -- a
                    # broken hook (e.g. Drive-reference resolution) must never
                    # cost the note that was just correctly created.
                    print(f"[connectors] on_note_created hook failed (non-fatal): {e}")

        unattributed_ids = [it["id"] for it in batch if it["id"] not in attributed_ids]
        if unattributed_ids:
            supabase.table("ingest_items").update(
                {"status": "discarded"}
            ).in_("id", unattributed_ids).execute()
            discarded += len(unattributed_ids)

        if job is not None:
            job["batches_done"] = bi + 1
            job["notes_created"] = notes_created

    return {"batches": len(batches), "notes_created": notes_created, "items_discarded": discarded}


def delete_note(note_id: str) -> None:
    """Removes a note, every chunk it produced (chunks share document_id =
    note id), and every provenance row it has (knowledge_note_sources also
    has an ON DELETE CASCADE FK to knowledge_notes as a second layer, but
    this stays explicit here to match the existing document_chunks pattern
    and to not depend solely on the FK firing)."""
    supabase.table("document_chunks").delete().eq("document_id", note_id).execute()
    supabase.table("knowledge_note_sources").delete().eq("note_id", note_id).execute()
    supabase.table("knowledge_notes").delete().eq("id", note_id).execute()


# ── Generic REST routes (frontend drives these; Railway is the API) ─────────────

@router.get("/connections")
async def list_connections(workspace_id: str,
                           auth: AuthContext = Depends(current_user)):
    """All external connections for a workspace (status shown in Settings)."""
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    auth.assert_workspace(workspace_id)
    rows = supabase.table("connections").select(
        "id, provider, external_team_name, status, error_detail, config, connected_by, created_at"
    ).eq("workspace_id", workspace_id).execute().data or []
    return {"connections": rows}


def _connection_workspace(connection_id: str) -> str:
    """
    Resolves which workspace owns a connection, so routes keyed only by an opaque
    id can still be authorised. Without this, knowing a connection UUID was enough
    to revoke someone else's integration.
    """
    row = supabase.table("connections").select("workspace_id") \
        .eq("id", connection_id).execute().data
    if not row:
        raise HTTPException(status_code=404, detail="Connection not found.")
    return row[0]["workspace_id"]


@router.delete("/connections/{connection_id}")
async def disconnect(connection_id: str, delete_notes: bool = False,
                     auth: AuthContext = Depends(current_user)):
    """Revokes a connection. Optionally deletes all knowledge notes it produced."""
    auth.assert_workspace(_connection_workspace(connection_id))
    if delete_notes:
        notes = supabase.table("knowledge_notes").select("id") \
            .eq("connection_id", connection_id).execute().data or []
        for n in notes:
            delete_note(n["id"])
    supabase.table("connections").update({"status": "revoked"}).eq("id", connection_id).execute()
    return {"success": True, "notes_deleted": delete_notes}


def _resolve_allowed_sensitivities(role: Optional[str], is_super_admin: bool) -> list[str]:
    """Same ladder as query.py's/chatbot.py's identical helper -- kept as a
    small local duplicate rather than a cross-module import, matching this
    codebase's existing convention of small per-file helpers over shared
    coupling (see query.py's own docstring on this exact point)."""
    if is_super_admin or role == "owner":
        return ["public", "internal", "confidential", "restricted"]
    if role == "admin":
        return ["public", "internal", "confidential"]
    return ["public", "internal"]


@router.get("/knowledge-notes")
async def list_knowledge_notes(workspace_id: str, limit: int = 100,
                               auth: AuthContext = Depends(current_user)):
    """Distilled notes captured from integrations — shown in Library.

    SECURITY FIX (2026-08-17): this route previously applied zero
    sensitivity filtering -- any workspace member could see every note
    regardless of its real classified sensitivity, unlike /document-tables
    which already resolves the caller's real role server-side. The ladder
    is resolved here from the caller's own role/is_super_admin (never from
    anything the client sends -- there is no client-supplied sensitivity
    parameter on this route at all) and applied as a server-side filter,
    identical in shape to /document-tables' existing mechanism.
    """
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    auth.assert_workspace(workspace_id)

    role = auth.role_in(workspace_id)
    allowed = _resolve_allowed_sensitivities(role, auth.is_super_admin)

    rows = supabase.table("knowledge_notes").select(
        "id, provider, source_type, category, title, body, participants, source_ref, occurred_at, created_at"
    ).eq("workspace_id", workspace_id).eq("status", "active") \
        .in_("sensitivity", allowed) \
        .order("created_at", desc=True).limit(limit).execute().data or []
    return {"notes": rows}


@router.delete("/knowledge-notes/{note_id}")
async def delete_knowledge_note(note_id: str,
                                auth: AuthContext = Depends(current_user)):
    """Deletes a note and its chunks (admin curation)."""
    row = supabase.table("knowledge_notes").select("workspace_id") \
        .eq("id", note_id).execute().data
    if not row:
        raise HTTPException(status_code=404, detail="Note not found.")
    auth.assert_workspace(row[0]["workspace_id"])
    delete_note(note_id)
    return {"success": True}


class SyncRequest(BaseModel):
    connection_id: str


@router.post("/connectors/sync")
async def trigger_filtration(body: SyncRequest,
                             auth: AuthContext = Depends(current_user)):
    """
    Runs filtration over any pending captured items for a connection
    (in the background). Provider-specific FETCH (pulling new messages into
    ingest_items) is triggered by the provider's own sync route or webhook;
    this endpoint distills whatever is already captured.
    """
    conn = supabase.table("connections").select("*").eq("id", body.connection_id).execute().data
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found.")
    conn = conn[0]

    auth.assert_workspace(conn["workspace_id"])

    import uuid as _uuid
    job_id = str(_uuid.uuid4())
    SYNC_JOBS[job_id] = {"job_id": job_id, "connection_id": body.connection_id,
                         "status": "processing", "notes_created": 0, "batches_done": 0}

    def _work():
        try:
            resolver = None
            if conn["provider"] == "slack":
                # Deferred import: connector_slack.py imports THIS module at
                # module load time, so a top-level import here would be
                # circular. Safe as a call-time import -- both modules are
                # already fully loaded by the time a request reaches this
                # background thread.
                import connector_slack
                resolver = connector_slack.build_permalink_resolver(conn)
            result = run_filtration(conn["workspace_id"], conn["id"], conn["provider"],
                                    job=SYNC_JOBS[job_id], resolve_permalink=resolver)
            SYNC_JOBS[job_id].update({"status": "completed", **result})
        except Exception as e:
            import traceback; print(f"[connectors] filtration job failed: {e}"); print(traceback.format_exc())
            SYNC_JOBS[job_id].update({"status": "failed", "error": str(e)})

    threading.Thread(target=_work, daemon=True).start()
    return {"success": True, "job_id": job_id, "status": "processing"}


@router.get("/connectors/sync-status/{job_id}")
async def sync_status(job_id: str,
                      auth: AuthContext = Depends(current_user)):
    """Progress of a filtration/backfill job, authorised via the connection it runs on."""
    job = SYNC_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    auth.assert_workspace(_connection_workspace(job["connection_id"]))
    return job
