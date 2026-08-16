"""
Google Chat connector — reuses the EXISTING Slack-shaped filtration pipeline
verbatim (Google Workspace scope lock: "do not build a second filtration
engine"). No new keep/discard logic, no new batching logic, no new note
schema — only the fetch/normalize side is new.

Flow: poll_connection() (worker.py, scheduled, per Chat-enabled connection)
  → spaces.list (which spaces this account can see)
  → spaces.messages.list per space
  → normalize each message into the SAME raw shape connector_slack.py
    already produces (channel/channel_name, user/user_name, text, ts,
    thread_ts) so brain_connectors.batch_conversations()/_format_batch()/
    classify_batch() work completely unmodified
  → save_ingest_items() (existing, provider-parameterized)
  → run_filtration(..., provider="google_chat", resolve_permalink=...)
    (existing, provider-parameterized) — same KEEP/DISCARD → knowledge_notes
    → knowledge_note_sources → embeddings pipeline Slack already uses.

Permalink note: Google Chat has no single documented "get permalink" RPC the
way Slack's chat.getPermalink is — the resolver below constructs a deep link
from the message's own resource name, following Google Chat's documented
resource-name format (spaces/{space}/messages/{message}). This has NOT been
verified against a live Google Chat account (same honest caveat
connector_zoom.py already carries for its own not-yet-verified scope names) —
worth a live check before trusting it in production. Never fabricated
speculatively beyond that documented format.
"""
from datetime import timedelta, datetime, timezone
import httpx

import brain_connectors as bc
import connector_google as google

CHAT_API_BASE = "https://chat.googleapis.com/v1"
LOOKBACK = timedelta(days=1)  # how far back each poll looks for new messages


def _chat_get(path: str, token: str, params: dict = None) -> dict:
    res = httpx.get(f"{CHAT_API_BASE}/{path}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params or {}, timeout=30)
    if res.status_code != 200:
        raise Exception(f"Google Chat API error ({res.status_code}): {res.text[:300]}")
    return res.json()


def _list_spaces(token: str) -> list[dict]:
    spaces, page_token = [], None
    while True:
        params = {"pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        data = _chat_get("spaces", token, params)
        spaces.extend(data.get("spaces", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return spaces


def _list_messages(token: str, space_name: str) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - LOOKBACK).isoformat()
    messages, page_token = [], None
    while True:
        params = {"filter": f'createTime > "{cutoff}"', "pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        data = _chat_get(f"{space_name}/messages", token, params)
        messages.extend(data.get("messages", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return messages


def _permalink(message: dict) -> str:
    """See module docstring's Permalink note — best-effort, documented
    resource-name format, not live-verified."""
    name = message.get("name", "")  # "spaces/{space}/messages/{message}"
    parts = name.split("/")
    if len(parts) >= 4:
        return f"https://chat.google.com/room/{parts[1]}/{parts[3]}"
    return f"https://chat.google.com/{name}"


def _normalize_message(message: dict, space: dict) -> dict:
    """Slack-shaped raw dict so the existing filtration pipeline needs zero
    changes -- see module docstring."""
    sender = message.get("sender", {})
    thread = message.get("thread", {})
    return {
        "external_id": message.get("name"),
        "kind": "message",
        "raw": {
            "channel":      space.get("name"),
            "channel_name": space.get("displayName") or space.get("name"),
            "user":         sender.get("name"),
            "user_name":    sender.get("displayName") or sender.get("name"),
            "text":         message.get("text", ""),
            "ts":           message.get("createTime", ""),
            "thread_ts":    thread.get("name") if thread.get("name") != space.get("name") else None,
            "permalink":    _permalink(message),
        },
    }


def build_permalink_resolver(conn: dict):
    """Mirrors connector_slack.build_permalink_resolver's shape (see its
    docstring) -- resolves from the already-normalized raw dict's own
    'permalink' field rather than a second API call, since _permalink() is
    computed once at normalize time from data already in hand."""
    if conn.get("provider") != "google_drive":
        return None
    if "chat" not in (conn.get("config") or {}).get("enabled_surfaces", []):
        return None

    def _resolve(raw: dict):
        return raw.get("permalink")

    return _resolve


def poll_connection(connection_id: str, workspace_id: str) -> dict:
    """The main job — see module docstring. Called by worker.py per active,
    Chat-enabled connection."""
    conn = bc.supabase.table("connections").select("*").eq("id", connection_id).execute().data
    if not conn:
        raise Exception("Connection not found.")
    conn = conn[0]
    if conn.get("provider") != "google_drive" or conn.get("status") != "active":
        raise Exception("Connection is not an active Google connection.")
    if "chat" not in (conn.get("config") or {}).get("enabled_surfaces", []):
        raise Exception("Chat is not enabled for this connection.")

    token = google._valid_access_token(conn)
    if not token:
        raise Exception("Google connection needs to be reconnected.")

    spaces = _list_spaces(token)
    all_items = []
    for space in spaces:
        for message in _list_messages(token, space.get("name")):
            if (message.get("text") or "").strip():
                all_items.append(_normalize_message(message, space))

    stored = bc.save_ingest_items(workspace_id, connection_id, "google_chat", all_items)

    resolver = build_permalink_resolver(conn)
    filtration_result = bc.run_filtration(workspace_id, connection_id, "google_chat",
                                          resolve_permalink=resolver)

    return {"spaces_checked": len(spaces), "messages_seen": len(all_items),
            "new_items_stored": stored, **filtration_result}
