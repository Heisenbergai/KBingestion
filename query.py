import os
import re
import ai
import httpx
import auth as auth_mod
import query_reasoning
import query_routing
import grounding
import signals
from fastapi import APIRouter, HTTPException, Depends, Header
from auth import AuthContext, current_user
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

GAP_MARKER = "⚠️ Not in your knowledge base:"


class QueryRequest(BaseModel):
    question:     str
    workspace_id: str           # ← REQUIRED — only search this workspace's chunks
    asset_id:     Optional[str] = None
    match_count:  Optional[int] = 8
    # Optional narrowing to specific documents (a dashboard card scoped to a
    # folder resolves that folder to its document ids client-side, through the
    # caller's own RLS, and sends them here). This is a RELEVANCE filter, not a
    # security boundary — filter_sensitivities below remains the access control,
    # and is always resolved server-side from the caller's real role. Sending a
    # document id you cannot see gains you nothing: both filters are ANDed
    # inside match_chunks_hybrid's SQL.
    filter_document_ids: Optional[list[str]] = None
    # Recent turns from this same conversation, oldest first -- {"question":
    # str, "answer": str}. Optional and stateless-safe: omitting it (or an
    # empty list) reproduces the exact single-shot behavior this endpoint
    # always had. Capped hard server-side below, never trusting the caller's
    # own length.
    history: Optional[list[dict]] = None

_MAX_HISTORY_TURNS = 6  # hard cap regardless of what the caller sends


def _resolve_allowed_sensitivities(role: Optional[str], is_super_admin: bool) -> list[str]:
    """Same ladder as chatbot.py's identical helper — kept as a small local
    duplicate rather than a cross-module import, matching this codebase's
    existing convention of small per-file helpers over shared coupling."""
    if is_super_admin or role == "owner":
        return ["public", "internal", "confidential", "restricted"]
    if role == "admin":
        return ["public", "internal", "confidential"]
    return ["public", "internal"]


def _fetch_my_restricted_grants(token: str, user_id: str) -> list[str]:
    """Same forwarded-token pattern as chatbot.py's _fetch_my_chatbot_ids."""
    if not auth_mod.APP_SUPABASE_URL or not auth_mod.APP_SUPABASE_ANON_KEY:
        return []
    try:
        with httpx.Client(timeout=10) as client:
            res = client.get(
                f"{auth_mod.APP_SUPABASE_URL}/rest/v1/knowledge_item_grants",
                params={
                    "select":             "knowledge_item_id",
                    "granted_to_user_id": f"eq.{user_id}",
                },
                headers={
                    "apikey":        auth_mod.APP_SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {token}",
                    "Accept":        "application/json",
                },
            )
            res.raise_for_status()
            return [row["knowledge_item_id"] for row in res.json()]
    except Exception as e:
        print(f"QUERY: failed to resolve caller's restricted grants (failing closed): {e}")
        return []


def hybrid_search(question: str, workspace_id: str,
                  match_count: int = 8, asset_id: str = None,
                  filter_sensitivities: Optional[list[str]] = None,
                  filter_restricted_grant_ids: Optional[list[str]] = None,
                  filter_document_ids: Optional[list[str]] = None,
                  boost_document_ids: Optional[list[str]] = None,
                  boost_doc_classes: Optional[list[str]] = None) -> list[dict]:
    """
    Company-brain retrieval: vector + keyword fused with Reciprocal Rank
    Fusion, then boosted by source tier (official docs > curated notes >
    chat) and freshness. Falls back to pure-vector match_chunks_workspace
    if the hybrid RPC is unavailable (safety net during rollout).
    Always workspace-isolated. Phase E: also sensitivity-filtered — AI
    Search previously let any workspace member search up a Confidential
    document with no check at all; the caller's real ladder is resolved by
    the route handler and passed in here, same as chatbot.py's bots.

    Returns up to `match_count` chunks, deduplicated (see query_reasoning.
    deduplicate_chunks) so a highly repetitive source document can't crowd
    out genuinely distinct evidence — found live 2026-08-15: a document
    with the same paragraph repeated 18x filled 7 of 8 result slots with
    near-duplicates of itself. To leave room for dedup to still return a
    full match_count of DISTINCT chunks, this asks the RPC for a wider pool
    than the caller requested and truncates after deduping, not before.
    """
    embedding = ai.embed_texts([question], workspace_id=workspace_id, feature="ai_search")[0]
    # Defense in depth: ai.embed_texts() always raises on failure today (it
    # never returns None), so this is not a fix for an observed live bug —
    # but a null/empty embedding must NEVER be allowed to silently reach the
    # RPC, which was confirmed live to accept one and rank by meaningless
    # NULL-cosine-distance instead of erroring. Fail loudly here instead.
    if not embedding:
        raise ValueError("hybrid_search: embedding generation returned no vector for the query")

    # Ask for more candidates than the caller needs so deduplicate_chunks()
    # below has real distinct evidence to choose from, not just whatever
    # survives after match_count already truncated the ranked list down to
    # (say) 8. Bounded at the RPC's own candidate_pool default (40) — this
    # adds no extra ranking work server-side, match_chunks_hybrid already
    # computes scores over the full candidate pool regardless of match_count.
    fetch_count = min(match_count * 4, 40)
    rpc_args = {
        "query_text":                 question,
        "query_embedding":             embedding,
        "match_count":                 fetch_count,
        "filter_workspace_id":         workspace_id,
        "filter_asset_id":             asset_id,
        "filter_sensitivities":        filter_sensitivities,
        "filter_restricted_grant_ids": filter_restricted_grant_ids,
        "filter_document_ids":         filter_document_ids or None,
        # Soft routing (query_routing.py). RELEVANCE ONLY — these multiply a
        # score, they never gate a row. Both None = scoring identical to
        # before routing existed.
        "boost_document_ids":          boost_document_ids or None,
        "boost_doc_classes":           boost_doc_classes or None,
    }
    try:
        result = supabase.rpc("match_chunks_hybrid", rpc_args).execute()
        chunks = result.data or []
    except Exception as e:
        print(f"[query] hybrid search unavailable, falling back to vector-only: {e}")
        fallback_args = dict(rpc_args)
        fallback_args.pop("query_text")
        # match_chunks_workspace (vector-only safety net) does NOT implement
        # the boost params — dropping them here rather than adding an unused
        # signature to a second function. Consequence, stated rather than
        # hidden: if the hybrid RPC is ever unavailable, retrieval still works
        # and stays correctly access-filtered, but is unrouted. Ranking-only
        # divergence, never an access divergence.
        fallback_args.pop("boost_document_ids", None)
        fallback_args.pop("boost_doc_classes", None)
        result = supabase.rpc("match_chunks_workspace", fallback_args).execute()
        chunks = result.data or []

    return query_reasoning.deduplicate_chunks(chunks)[:match_count]


def build_context_and_citations(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Numbers each chunk as a citable source [n] and returns the LLM
    context block + a structured citations list for the frontend."""
    context_parts, citations = [], []
    for i, ch in enumerate(chunks, 1):
        meta = ch.get("metadata") or {}
        file_name = meta.get("file_name", "Unknown document")
        stype = ch.get("source_type") or meta.get("source_type") or "document"
        label = {"document": "company document", "meeting": "meeting note",
                 "slack": "team chat", "note": "curated note"}.get(stype, stype)
        context_parts.append(f"[{i}] {file_name} ({label}):\n{ch['content']}")
        citations.append({
            "index":       i,
            "file_name":   file_name,
            "snippet":     ch["content"][:200],
            "source_type": stype,
            "source_tier": ch.get("source_tier", 1),
        })
    return "\n\n---\n\n".join(context_parts), citations


def split_answer_and_gaps(text: str) -> tuple[str, Optional[str]]:
    """Separates the gap note (what the brain doesn't know) from the answer
    so the frontend can style it distinctly."""
    if GAP_MARKER in text:
        answer, gap = text.split(GAP_MARKER, 1)
        gap = gap.strip()
        return answer.strip(), (gap or None)
    return text.strip(), None


@router.get("/document-tables")
async def list_document_tables(workspace_id: str,
                               auth: AuthContext = Depends(current_user),
                               authorization: Optional[str] = Header(None)):
    """
    Phase I: the structured sheets (Phase H) a caller is allowed to read, so a
    dashboard can build a metric from a real spreadsheet cell.

    ACCESS: the sensitivity ladder is resolved SERVER-SIDE from the caller's real
    role and applied in the query — identical to /query. The client never says
    which documents it may see; that would be the client-trusted access control
    this project already had to fix once on the public widget path.

    Row payloads are deliberately NOT returned here — only the shape (sheet
    names, headers, which columns are numeric, row counts). A dashboard card
    picks a column from this, then fetches just that column's values. Shipping
    every cell of every sheet to build a picker would be both slow and a wider
    exposure than the feature needs.
    """
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    auth.assert_workspace(workspace_id)

    role = auth.role_in(workspace_id)
    allowed = _resolve_allowed_sensitivities(role, auth.is_super_admin)

    try:
        res = (supabase.table("document_tables")
               .select("id, document_id, sheet_name, headers, numeric_columns, row_count, sensitivity")
               .eq("workspace_id", workspace_id)
               .in_("sensitivity", allowed)
               .is_("deleted_at", "null")
               .execute())
        return {"tables": res.data or [], "workspace_id": workspace_id}
    except Exception as e:
        print(f"DOCUMENT-TABLES ERROR: {e}")
        raise HTTPException(status_code=500, detail="Could not load spreadsheet tables.")


@router.get("/document-table-rows/{table_id}")
async def get_document_table_rows(table_id: str, workspace_id: str,
                                  auth: AuthContext = Depends(current_user)):
    """
    The actual cell values for ONE sheet the caller has already been shown.

    The sensitivity check is repeated here rather than assumed from the listing
    call — an endpoint that trusts "you must have listed it first" is trusting
    the client's word about a previous request.
    """
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    auth.assert_workspace(workspace_id)

    role = auth.role_in(workspace_id)
    allowed = _resolve_allowed_sensitivities(role, auth.is_super_admin)

    try:
        res = (supabase.table("document_tables")
               .select("id, document_id, sheet_name, headers, rows, numeric_columns, row_count")
               .eq("id", table_id)
               .eq("workspace_id", workspace_id)   # never trust the id alone
               .in_("sensitivity", allowed)
               .is_("deleted_at", "null")
               .limit(1)
               .execute())
        rows = res.data or []
        if not rows:
            # Deliberately indistinguishable from "does not exist": telling a
            # caller that a sheet exists but is above their tier leaks its
            # existence.
            raise HTTPException(status_code=404, detail="Table not found.")
        return rows[0]
    except HTTPException:
        raise
    except Exception as e:
        print(f"DOCUMENT-TABLE-ROWS ERROR: {e}")
        raise HTTPException(status_code=500, detail="Could not load sheet rows.")


class WidgetSuggestRequest(BaseModel):
    workspace_id: str
    description: str


_WIDGET_SUGGEST_EMPTY = {
    "tableId": None, "valueColumn": None, "groupColumn": None,
    "aggregation": None, "chartKind": None,
}

_WIDGET_SUGGEST_PROMPT = """You help someone build a chart for their company dashboard by picking the
best-matching data source and settings from a list of REAL spreadsheet
sheets already in their workspace. You are NOT computing or inventing any
number -- you only select and configure from what is listed below.

Available sheets:
{catalog}

Return JSON only:
{{"tableId": "<id of the best-matching sheet above, or null>",
  "valueColumn": "<a numeric column from THAT sheet's own numeric_columns, or null>",
  "groupColumn": "<a header from THAT sheet to group by, or null>",
  "aggregation": "<one of sum, average, latest, count, or null>",
  "chartKind": "<one of stat, bar, line, area, pie, table, or null>"}}

Rules:
- tableId MUST be one of the ids listed above, or null if nothing matches well.
- valueColumn MUST come from that sheet's own numeric_columns, or null.
- groupColumn MUST come from that sheet's own headers and differ from valueColumn, or null.
- Use null for anything you are not reasonably confident about -- a missing
  suggestion costs nothing, a wrong one wastes the person's time undoing it.
- Prefer "line" or "area" for a trend over time, "bar" for comparing
  categories, "pie" for a share/breakdown, "table" only if they explicitly
  want raw rows, "stat" for a single headline number.

What they asked for: {description}"""


def _widget_catalog_for_prompt(tables: list[dict]) -> str:
    # Bounded so a workspace with many sheets can't blow out the prompt --
    # 30 sheets is already far more than one workspace has today (14's own
    # audit found ~13 real spreadsheets total across the whole corpus).
    lines = []
    for t in tables[:30]:
        lines.append(
            f'- id="{t["id"]}" sheet="{t["sheet_name"]}" headers={t["headers"]} '
            f'numeric_columns={t["numeric_columns"]}'
        )
    return "\n".join(lines)


@router.post("/widget-suggest")
async def widget_suggest(request: WidgetSuggestRequest,
                         auth: AuthContext = Depends(current_user)):
    """
    G7 of the dashboard-upgrade thread (2026-08-13) -- the optional "describe
    what you want" input in the Create Widget wizard (G6).

    WHAT THE MODEL SEES, AND DOESN'T. It is handed sheet/column LABELS only
    -- id, sheet_name, headers, numeric_columns -- resolved from
    document_tables with the caller's own sensitivity ladder applied
    server-side, exactly like /document-tables. It never sees a single cell
    VALUE, so there is nothing here for it to hallucinate about an actual
    figure -- it can only pick a wrong SHEET or COLUMN, which the wizard's
    existing live preview (G6) already makes obvious and correctable before
    anything is added to a dashboard. Same "AI configures, never computes
    the number" boundary spreadsheet_metric was built around from the start.

    FAILS OPEN on every path, same convention as query_routing.py: no
    description, no catalog, no Bedrock, bad JSON, a hallucinated
    id/column -- all return an all-null suggestion, never a 4xx/5xx the
    frontend has to special-case. The manual picker (G6) is always still
    right there.
    """
    if not request.workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    auth.assert_workspace(request.workspace_id)

    description = (request.description or "").strip()
    if not description:
        return _WIDGET_SUGGEST_EMPTY

    role = auth.role_in(request.workspace_id)
    allowed = _resolve_allowed_sensitivities(role, auth.is_super_admin)

    try:
        res = (supabase.table("document_tables")
               .select("id, sheet_name, headers, numeric_columns")
               .eq("workspace_id", request.workspace_id)
               .in_("sensitivity", allowed)
               .execute())
        tables = [t for t in (res.data or []) if t.get("numeric_columns")]
    except Exception as e:
        print(f"[widget-suggest] catalog fetch failed, no suggestion: {e}")
        return _WIDGET_SUGGEST_EMPTY

    if not tables:
        return _WIDGET_SUGGEST_EMPTY

    try:
        raw = ai.chat_json(
            [{"role": "user", "content": _WIDGET_SUGGEST_PROMPT.format(
                catalog=_widget_catalog_for_prompt(tables),
                description=description[:300],
            )}],
            max_tokens=200,
            temperature=0,
            workspace_id=request.workspace_id,
            user_id=auth.user_id,
            feature="widget_suggest",
        )
    except Exception as e:
        print(f"[widget-suggest] suggestion failed, falling back to manual picker: {e}")
        return _WIDGET_SUGGEST_EMPTY

    if not isinstance(raw, dict):
        return _WIDGET_SUGGEST_EMPTY

    # Validate against the REAL catalog rather than trusting the model, same
    # discipline query_routing.py already established for department names:
    # a hallucinated id/column would otherwise silently reach the frontend.
    by_id = {t["id"]: t for t in tables}
    table = by_id.get(raw.get("tableId")) if isinstance(raw.get("tableId"), str) else None
    if not table:
        return _WIDGET_SUGGEST_EMPTY

    value_col = raw.get("valueColumn")
    if value_col not in (table.get("numeric_columns") or []):
        value_col = None

    group_col = raw.get("groupColumn")
    if group_col not in (table.get("headers") or []) or group_col == value_col:
        group_col = None

    aggregation = raw.get("aggregation")
    if aggregation not in ("sum", "average", "latest", "count"):
        aggregation = None

    chart_kind = raw.get("chartKind")
    if chart_kind not in ("stat", "bar", "line", "area", "pie", "table"):
        chart_kind = None

    return {
        "tableId": table["id"],
        "valueColumn": value_col,
        "groupColumn": group_col,
        "aggregation": aggregation,
        "chartKind": chart_kind,
    }


@router.post("/query")
async def query_documents(request: QueryRequest,
                          auth: AuthContext = Depends(current_user),
                          authorization: Optional[str] = Header(None)):
    """
    Company-brain search over ONLY the caller's workspace. Returns a
    synthesized answer with inline [n] citations, a structured citations
    list, and an explicit "gaps" note describing what the knowledge base
    does not yet cover. No cross-workspace data is ever returned.
    """
    try:
        if not request.workspace_id:
            raise HTTPException(status_code=400, detail="workspace_id is required for all queries.")

        # The workspace_id below decides which chunks are searched, so it has to be
        # authorised before it is used — not merely present.
        auth.assert_workspace(request.workspace_id)

        # Phase E: resolve this caller's real sensitivity access, same as
        # chatbot.py's internal-query path — closes the gap where any
        # workspace member could previously search up a Confidential
        # document via AI Search with no check at all.
        role = auth.role_in(request.workspace_id)
        filter_sensitivities = _resolve_allowed_sensitivities(role, auth.is_super_admin)
        filter_restricted_grant_ids = None
        if not auth.is_super_admin and role == "admin":
            token = ""
            if authorization and authorization.lower().startswith("bearer "):
                token = authorization[7:].strip()
            filter_restricted_grant_ids = _fetch_my_restricted_grants(token, auth.user_id) or None

        # Conversational memory, retrieval side. A follow-up like "what are
        # the challenges in achieving THESE targets" means nothing to a
        # search index on its own — it has no idea what "these targets" is.
        # Rewriting it into a standalone question BEFORE retrieval (and
        # before routing, which infers department/class from the same
        # wording) is what actually fixes this, not just handing the raw
        # follow-up to the answer-generation step. Capped hard here, not
        # left to the caller's own length.
        history_turns = (request.history or [])[-_MAX_HISTORY_TURNS:]
        search_question = query_reasoning.condense_followup(
            request.question, history_turns,
            workspace_id=request.workspace_id, user_id=auth.user_id,
        )

        # Soft routing: infer which department/class the question belongs to
        # and BOOST that branch. Never filters — see query_routing.py for why
        # hard routing was rejected. Fails open to no boost, so a routing
        # outage leaves retrieval exactly as it was.
        _caller_token = ""
        if authorization and authorization.lower().startswith("bearer "):
            _caller_token = authorization[7:].strip()
        routing = query_routing.compute_boosts(
            search_question, _caller_token, request.workspace_id,
        )

        chunks = hybrid_search(
            search_question, request.workspace_id,
            match_count=request.match_count or 8, asset_id=request.asset_id,
            filter_sensitivities=filter_sensitivities,
            filter_restricted_grant_ids=filter_restricted_grant_ids,
            filter_document_ids=request.filter_document_ids or None,
            boost_document_ids=routing["boost_document_ids"],
            boost_doc_classes=routing["boost_doc_classes"],
        )

        # R-B: retry with a reformulated query when the first attempt came back
        # weak (top_sim < 0.30 is the 'low' floor / 'none' when chunks is
        # empty). Every retry reuses the SAME filter_sensitivities/
        # filter_restricted_grant_ids/filter_document_ids as the primary call —
        # hybrid_search takes those as fixed params, so a retry cannot widen
        # access, only find more of what this caller could already see. One
        # round, capped at query_reasoning.REFORMULATION_CAP — see that
        # module's docstring for the full cost/safety case.
        first_top_sim = max((c.get("similarity") or 0) for c in chunks) if chunks else 0.0
        if first_top_sim < 0.30:
            alt_queries = query_reasoning.reformulate_query(
                search_question, workspace_id=request.workspace_id,
            )
            retry_batches = []
            for alt_q in alt_queries:
                try:
                    retry_batches.append(hybrid_search(
                        alt_q, request.workspace_id,
                        match_count=request.match_count or 8, asset_id=request.asset_id,
                        filter_sensitivities=filter_sensitivities,
                        filter_restricted_grant_ids=filter_restricted_grant_ids,
                        filter_document_ids=request.filter_document_ids or None,
                        # Same routing on retries: the department a question
                        # belongs to doesn't change just because the wording did.
                        boost_document_ids=routing["boost_document_ids"],
                        boost_doc_classes=routing["boost_doc_classes"],
                    ))
                except Exception as e:
                    print(f"QUERY: reformulated retrieval failed for one alt query (non-fatal): {e}")
            if retry_batches:
                chunks = query_reasoning.merge_chunk_results(
                    chunks, retry_batches, match_count=request.match_count or 8,
                )
                # merge_chunk_results dedupes by chunk ID across batches —
                # a different reformulated query can surface a DIFFERENT
                # chunk id that's still near-duplicate CONTENT of one
                # already kept from the primary attempt. Each individual
                # hybrid_search() call already deduped its own batch; this
                # catches duplicates introduced by combining batches.
                chunks = query_reasoning.deduplicate_chunks(chunks)

        if not chunks:
            return {
                "answer":  "I couldn't find anything about this in your knowledge base.",
                "citations": [],
                "sources": [],
                "chunks":  [],
                "gaps":    "Your knowledge base has no documents covering this topic yet. "
                           "Upload relevant documents or connect a data source, then try again.",
                "confidence":   "none",
                "workspace_id": request.workspace_id,
            }

        context, citations = build_context_and_citations(chunks)

        system_prompt = f"""You are the company's knowledge assistant. Answer the employee's \
question using ONLY the numbered sources below.

Citation rules:
- After each fact or claim, cite the source number(s) it came from, e.g. [1] or [2][3].
- Use ONLY information present in the sources. Never use outside knowledge or guess.
- Prefer official company documents over informal chat when sources disagree, and say so.
- Earlier turns of this conversation may be included below for CONTEXT ONLY (so
  you understand what "these", "that", or "compared to before" refers to). They
  are not a source — every factual claim in your answer must still be cited to
  the numbered sources, never to something only your own earlier answer said.

Honesty about gaps (important):
- Only write a gap note if part of what the person actually ASKED is missing from the
  sources -- never because the general topic could have more detail, a related-but-
  unasked fact isn't mentioned, or the answer could simply be more comprehensive. If
  you have fully answered the question as asked, do NOT write a gap note, even if the
  sources could say more about the broader subject.
- Do NOT invent a more specific follow-on question the person didn't ask, then flag
  ITS answer as missing. Example: if asked "what are the key priorities for X", and
  you answer with the priorities the sources support, do NOT then add a gap saying
  "specific metrics/execution details/Q3 figures for X are not provided" -- those are
  a MORE SPECIFIC question than the one actually asked, not a gap in the one asked.
- If the sources answer the question but a SPECIFIC part of what was actually asked is
  missing, answer what you can with citations, then on a new final line write
  "{GAP_MARKER}" followed by ONE plain sentence naming exactly what part of the
  question is unanswered -- no markdown headers, no bold labels, no restating the answer.
- If the sources do NOT answer the question at all, write ONLY a one-line plain-sentence
  note under "{GAP_MARKER}" explaining what's missing, and nothing else -- no partial
  answer, no markdown formatting.

Formatting:
- Markdown. Start longer answers with a one-sentence summary.
- Match structure to content (numbered steps, bullets for options, tables for comparisons).
- **Bold** key terms and numbers. Keep paragraphs short."""

        # Conversational memory, generation side. Retrieval above searched for
        # what the follow-up actually MEANS (search_question); the model
        # answering it needs the actual prior turns, in the person's own
        # words, so it can naturally continue the conversation ("similarly to
        # Engineering...") rather than answering as if this were the first
        # question ever asked. Original wording here, not the condensed
        # rewrite -- condensation is a search aid, not what the person asked.
        history_block = "\n\n".join(
            f"Q: {h['question']}\nA: {h['answer']}"
            for h in history_turns if h.get("question") and h.get("answer")
        )
        user_content = (
            (f"Earlier in this conversation:\n{history_block}\n\n" if history_block else "")
            + f"Sources:\n{context}\n\nQuestion: {request.question}\n\nAnswer with citations:"
        )

        raw = ai.chat(
            messages=[{
                "role": "user",
                "content": user_content,
            }],
            system=system_prompt,
            max_tokens=1000,
            temperature=0.2,
            workspace_id=request.workspace_id, user_id=auth.user_id, feature="ai_search",
        )
        answer, gaps = split_answer_and_gaps(raw)

        # R-B-2: a second, narrow validation pass on the model's OWN gap note
        # -- found live 2026-08-15 that the generation prompt's gap
        # instructions alone (tightened above) still weren't enough: the
        # model kept flagging a MORE SPECIFIC, unasked-for follow-on
        # question ("Q3 capacity expansion metrics") as though it were a gap
        # in the question actually asked ("what are the key priorities").
        # Only runs when a gap exists -- zero added cost on the (much more
        # common) no-gap path. See validate_gap_relevance's docstring for
        # the fail-open reasoning.
        if gaps and not query_reasoning.validate_gap_relevance(
            request.question, gaps, workspace_id=request.workspace_id, user_id=auth.user_id,
        ):
            gaps = None

        # Confidence from the best chunk's semantic similarity
        top_sim = max((c.get("similarity") or 0) for c in chunks)
        confidence = "high" if top_sim >= 0.45 else "medium" if top_sim >= 0.3 else "low"
        # A gap-only completion (answer is empty, the whole response was under
        # GAP_MARKER) means the model found NOTHING it could answer with --
        # top_sim-based confidence describes how close the RETRIEVED chunks
        # were, not whether an answer exists, so showing "High confidence"
        # next to no answer is actively misleading (found live 2026-08-15:
        # a genuine gap query still returned "high"). Reuses the SAME "none"
        # value the zero-chunks branch above already returns for an
        # unanswerable question -- not a new confidence state.
        if not answer.strip():
            confidence = "none"

        # Only surface citations the model actually referenced (keeps the UI clean)
        used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        cited = [c for c in citations if c["index"] in used] or citations

        # R-C: grounded self-critique, at ZERO extra AI cost — both functions
        # are pure/programmatic, working off the answer's own citation markers
        # and which documents the CITED chunks (not the whole candidate pool)
        # came from. citations[i-1] and chunks[i-1] are built from the same
        # enumerate() in build_context_and_citations, so indices line up.
        cited_chunks = [chunks[i - 1] for i in used if 1 <= i <= len(chunks)]
        coverage = grounding.citation_coverage(answer)
        corroboration = grounding.corroboration_level(cited_chunks or chunks)

        # coverage_ratio still feeds confidence downgrade below and the
        # `grounding` response field (both unchanged) -- but the raw
        # uncited-claim TEXT is deliberately never appended to the
        # user-facing `gaps` field anymore. It used to be (see git history),
        # and that was the exact source of two real live bugs (2026-08-15
        # gap-semantics pass): (1) a synthesizing "**Summary:** ..." lead-in
        # sentence -- which restates points that ARE individually cited
        # right below it -- got flagged as its own "uncited claim" and its
        # raw markdown text leaked straight into the gap callout; (2) this
        # fired on fully-answered questions, producing a gap callout on an
        # answer that had none. The model's OWN gap note (from GAP_MARKER,
        # tightened in the system prompt above to only fire on a
        # question-relevant missing part) is the sole source of `gaps` now.

        # Confidence can only move MORE cautious here, never more confident —
        # retrieval similarity remains the ceiling.
        confidence = grounding.downgrade_for_weak_grounding(confidence, coverage["coverage_ratio"])

        # R-D: dark signal collection — nothing reads this back yet. Reuses
        # cited_chunks already computed for R-C rather than recomputing.
        # 'source_cited' specifically, not 'source_used_in_context' — the
        # model chose to attribute a claim to these, a real citation.
        signals.log_scope_used(
            request.workspace_id, auth.user_id, "ai_search",
            request.question, request.filter_document_ids or [],
        )
        signals.log_sources_cited(
            request.workspace_id, auth.user_id, "ai_search", request.question,
            [c.get("document_id") for c in (cited_chunks or chunks)], confidence,
        )

        return {
            "answer":       answer,
            "citations":    cited,
            "sources":      list(dict.fromkeys(c["file_name"] for c in cited)),  # unique, ordered
            "chunks":       [c["content"] for c in chunks],  # backward compat: decks/visuals flows
            "gaps":         gaps,
            "confidence":   confidence,
            "grounding":    {"coverage_ratio": coverage["coverage_ratio"], "corroboration": corroboration},
            "workspace_id": request.workspace_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"QUERY ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
