"""
Query routing — infer which part of the knowledge tree a question belongs to,
then BOOST that branch during retrieval. Never filter to it.

WHY BOOST, NOT FILTER. The obvious design ("a Sales question searches only the
Sales folder") fails on the questions that matter most. "Which policies affect
sales commissions?" legitimately spans Sales, HR, Finance and Legal; routing it
to one branch returns "I don't know" while the answer sits one branch away.
Phase E already removed a silent over-broadening fallback and locked in "a
scoped bot that can't answer says I don't know" (F-27) — hard routing would
extend that failure mode from configured bots to EVERY query in the app.

Boosting degrades to exactly today's behaviour when the router is wrong or
unsure, which is the whole point: a bad route costs a little ranking quality,
never the answer. It also reuses the multiplier mechanism R-A already proved
(authority/lifecycle) rather than inventing a second, riskier one.

FAIL OPEN, ALWAYS. Every failure path here returns "no routing" — no LLM, bad
JSON, an unknown department name, a timeout, a missing token. Retrieval then
runs exactly as it did before this module existed. Same bias as
escalation_triage (fail open), and the opposite of reformulation's fail-closed
bias — deliberately: a missing boost costs ranking, while a wrongly-applied
one could bury a correct answer.

ACCESS. Routing is a RELEVANCE signal and never an access boundary. The
document ids resolved here are passed as `boost_document_ids`, which only ever
multiplies a score — the sensitivity ladder and folder scoping stay exactly
where they are, in the SQL's WHERE clause. A wrong or even forged department
cannot widen what a caller may see.
"""
from typing import Optional

import httpx

import ai
import auth as auth_mod

# Kept deliberately small. doc_class values must match the document_class enum
# exactly, or the boost silently matches nothing.
_DOC_CLASSES = [
    "financial", "strategy", "policy_sop", "legal", "product",
    "people", "sales_marketing", "research_reference", "meeting",
]

_PROMPT = """You route a question to the part of a company's knowledge base most
likely to answer it. You are NOT answering the question.

Available departments: {departments}
Available document classes: {doc_classes}

Return JSON only:
{{"department": "<one of the departments, or null>",
  "doc_class": "<one of the document classes, or null>"}}

Rules:
- Use null when genuinely unsure. Null is a good answer — it costs nothing,
  while a confident wrong route makes the right document harder to find.
- Pick the department that OWNS the answer, not one merely mentioned.
- A question spanning several departments should return null, not a guess.

Question: {question}"""


def route_question(
    question: str,
    department_names: list[str],
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    Returns {"department": str|None, "doc_class": str|None}.
    Never raises — an unroutable question is normal, not an error.

    ai.chat_json takes messages as list[dict], NOT a bare prompt string.
    Passing a string made ai.chat() iterate it CHARACTER BY CHARACTER and
    raise TypeError on m["role"], every single call — swallowed by the
    fail-open handler below, so routing silently never applied a boost from
    the day it shipped. Same failure class as P0-13: the safety net that
    makes a feature non-critical is exactly what hides that it never runs.
    """
    empty = {"department": None, "doc_class": None}
    if not question or not department_names:
        return empty

    try:
        raw = ai.chat_json(
            [{"role": "user", "content": _PROMPT.format(
                departments=", ".join(department_names),
                doc_classes=", ".join(_DOC_CLASSES),
                question=question[:500],
            )}],
            max_tokens=100,
            temperature=0,
            workspace_id=workspace_id,
            user_id=user_id,
            feature="query_routing",
        )
    except Exception as e:
        print(f"[routing] classification failed, continuing unrouted: {e}")
        return empty

    if not isinstance(raw, dict):
        return empty

    dept = raw.get("department")
    klass = raw.get("doc_class")

    # Validate against the REAL lists rather than trusting the model: a
    # hallucinated department name would resolve to zero documents and a
    # hallucinated doc_class would match no chunk, so both are dropped here
    # where it's visible instead of failing silently downstream.
    if not isinstance(dept, str) or dept not in department_names:
        dept = None
    if not isinstance(klass, str) or klass not in _DOC_CLASSES:
        klass = None

    return {"department": dept, "doc_class": klass}


def resolve_department_document_ids(token: str, workspace_id: str,
                                    department_name: str) -> list[str]:
    """
    Documents filed in folders belonging to one department.

    Uses the caller's OWN bearer token against the app DB's PostgREST API —
    the same forwarded-token pattern bot_analytics and query.py's restricted-
    grant fetch already use, because Railway holds no service-role credential
    for the app DB. A side effect worth stating: the caller's own RLS applies,
    so this can never surface a document id the caller couldn't already see.
    """
    if not token or not workspace_id or not department_name:
        return []
    if not auth_mod.APP_SUPABASE_URL or not auth_mod.APP_SUPABASE_ANON_KEY:
        return []

    headers = {
        "apikey": auth_mod.APP_SUPABASE_ANON_KEY,
        "Authorization": token if token.startswith("Bearer ") else f"Bearer {token}",
    }
    base = auth_mod.APP_SUPABASE_URL.rstrip("/")

    try:
        with httpx.Client(timeout=8) as client:
            dept = client.get(
                f"{base}/rest/v1/departments",
                headers=headers,
                params={"select": "id", "name": f"eq.{department_name}",
                        "workspace_id": f"eq.{workspace_id}", "limit": "1"},
            )
            dept.raise_for_status()
            dept_rows = dept.json() or []
            if not dept_rows:
                return []
            dept_id = dept_rows[0]["id"]

            folders = client.get(
                f"{base}/rest/v1/knowledge_folders",
                headers=headers,
                params={"select": "id", "department_id": f"eq.{dept_id}",
                        "workspace_id": f"eq.{workspace_id}"},
            )
            folders.raise_for_status()
            folder_ids = [f["id"] for f in (folders.json() or [])]
            if not folder_ids:
                return []

            items = client.get(
                f"{base}/rest/v1/knowledge_items",
                headers=headers,
                params={"select": "id",
                        "folder_id": f"in.({','.join(folder_ids)})",
                        "workspace_id": f"eq.{workspace_id}",
                        "deleted_at": "is.null"},
            )
            items.raise_for_status()
            return [i["id"] for i in (items.json() or [])]
    except Exception as e:
        print(f"[routing] department resolution failed, continuing unrouted: {e}")
        return []


def fetch_department_names(token: str, workspace_id: str) -> list[str]:
    """The routing vocabulary for this workspace. Empty list disables routing."""
    if not token or not workspace_id:
        return []
    if not auth_mod.APP_SUPABASE_URL or not auth_mod.APP_SUPABASE_ANON_KEY:
        return []
    headers = {
        "apikey": auth_mod.APP_SUPABASE_ANON_KEY,
        "Authorization": token if token.startswith("Bearer ") else f"Bearer {token}",
    }
    try:
        with httpx.Client(timeout=8) as client:
            res = client.get(
                f"{auth_mod.APP_SUPABASE_URL.rstrip('/')}/rest/v1/departments",
                headers=headers,
                params={"select": "name", "workspace_id": f"eq.{workspace_id}"},
            )
            res.raise_for_status()
            return [d["name"] for d in (res.json() or []) if d.get("name")]
    except Exception as e:
        print(f"[routing] department list failed, continuing unrouted: {e}")
        return []


def compute_boosts(question: str, token: Optional[str], workspace_id: Optional[str]) -> dict:
    """
    One call for the whole routing step. Returns
    {"boost_document_ids": list|None, "boost_doc_classes": list|None,
     "routed_department": str|None}.

    Any failure anywhere yields all-None, i.e. retrieval unchanged.
    """
    none_result = {"boost_document_ids": None, "boost_doc_classes": None,
                   "routed_department": None}
    if not question or not token or not workspace_id:
        return none_result

    try:
        names = fetch_department_names(token, workspace_id)
        if not names:
            return none_result

        routed = route_question(question, names, workspace_id=workspace_id)
        dept, klass = routed["department"], routed["doc_class"]
        if not dept and not klass:
            return none_result

        doc_ids = resolve_department_document_ids(token, workspace_id, dept) if dept else []

        return {
            # An empty list would boost nothing but still costs a SQL array
            # comparison per row, so send None instead — same reason the
            # existing filter_* params are None rather than [] when unused.
            "boost_document_ids": doc_ids or None,
            "boost_doc_classes": [klass] if klass else None,
            "routed_department": dept if doc_ids else None,
        }
    except Exception as e:
        print(f"[routing] boost computation failed, continuing unrouted: {e}")
        return none_result
