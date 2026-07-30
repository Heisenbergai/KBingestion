import os
import re
import ai
import httpx
import auth as auth_mod
import query_reasoning
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
                  filter_document_ids: Optional[list[str]] = None) -> list[dict]:
    """
    Company-brain retrieval: vector + keyword fused with Reciprocal Rank
    Fusion, then boosted by source tier (official docs > curated notes >
    chat) and freshness. Falls back to pure-vector match_chunks_workspace
    if the hybrid RPC is unavailable (safety net during rollout).
    Always workspace-isolated. Phase E: also sensitivity-filtered — AI
    Search previously let any workspace member search up a Confidential
    document with no check at all; the caller's real ladder is resolved by
    the route handler and passed in here, same as chatbot.py's bots.
    """
    embedding = ai.embed_texts([question], workspace_id=workspace_id, feature="ai_search")[0]
    rpc_args = {
        "query_text":                 question,
        "query_embedding":             embedding,
        "match_count":                 match_count,
        "filter_workspace_id":         workspace_id,
        "filter_asset_id":             asset_id,
        "filter_sensitivities":        filter_sensitivities,
        "filter_restricted_grant_ids": filter_restricted_grant_ids,
        "filter_document_ids":         filter_document_ids or None,
    }
    try:
        result = supabase.rpc("match_chunks_hybrid", rpc_args).execute()
        return result.data or []
    except Exception as e:
        print(f"[query] hybrid search unavailable, falling back to vector-only: {e}")
        fallback_args = dict(rpc_args)
        fallback_args.pop("query_text")
        result = supabase.rpc("match_chunks_workspace", fallback_args).execute()
        return result.data or []


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

        chunks = hybrid_search(
            request.question, request.workspace_id,
            match_count=request.match_count or 8, asset_id=request.asset_id,
            filter_sensitivities=filter_sensitivities,
            filter_restricted_grant_ids=filter_restricted_grant_ids,
            filter_document_ids=request.filter_document_ids or None,
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
                request.question, workspace_id=request.workspace_id,
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
                    ))
                except Exception as e:
                    print(f"QUERY: reformulated retrieval failed for one alt query (non-fatal): {e}")
            if retry_batches:
                chunks = query_reasoning.merge_chunk_results(
                    chunks, retry_batches, match_count=request.match_count or 8,
                )

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

Honesty about gaps (important):
- If the sources only PARTIALLY answer the question, answer what you can with citations,
  then on a new final line write "{GAP_MARKER}" followed by a short description of what is missing.
- If the sources do NOT answer the question at all, write a one-line note under "{GAP_MARKER}"
  explaining what's missing, and nothing else.

Formatting:
- Markdown. Start longer answers with a one-sentence summary.
- Match structure to content (numbered steps, bullets for options, tables for comparisons).
- **Bold** key terms and numbers. Keep paragraphs short."""

        raw = ai.chat(
            messages=[{
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {request.question}\n\nAnswer with citations:"
            }],
            system=system_prompt,
            max_tokens=1000,
            temperature=0.2,
            workspace_id=request.workspace_id, user_id=auth.user_id, feature="ai_search",
        )
        answer, gaps = split_answer_and_gaps(raw)

        # Confidence from the best chunk's semantic similarity
        top_sim = max((c.get("similarity") or 0) for c in chunks)
        confidence = "high" if top_sim >= 0.45 else "medium" if top_sim >= 0.3 else "low"

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

        # Cut, don't embellish: an uncited claim is not deleted from the
        # answer (fragile text surgery risks mangling valid prose) — it is
        # surfaced through the SAME gaps mechanism the model already uses to
        # admit what it doesn't know, so the reader sees it either way.
        if coverage["uncited"]:
            preview = "; ".join(coverage["uncited"][:2])
            note = f"Not tied to a specific source: {preview}"
            gaps = f"{gaps} {note}" if gaps else note

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
