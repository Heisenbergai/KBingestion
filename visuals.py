"""
AI Create — data visuals generated from workspace documents.

POST /generate-visual
    Takes a query + retrieved chunks (from /query, same pattern as
    presentations) and returns render-ready JSON for the frontend:
      visual_type "chart"       → Recharts-compatible chart spec
      visual_type "pivot_table" → columns + rows for a table component
      visual_type "auto"        → the AI picks whichever fits the data

GET /workspace-stats/{workspace_id}
    Vector-DB-side numbers for the dashboard (documents indexed, chunks).
    Everything else on the dashboard (files, storage, courses, chats)
    lives in Lovable's own DB or GET /bot-analytics.
"""
import os
import json
import ai
from fastapi import APIRouter, HTTPException
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

VALID_VISUAL_TYPES = ("auto", "chart", "pivot_table")
VALID_CHART_TYPES  = ("column", "bar", "line", "area", "pie")


class GenerateVisualRequest(BaseModel):
    query:       str
    chunks:      list[str]              # retrieved chunks from /query
    visual_type: Optional[str] = "auto" # auto | chart | pivot_table
    instruction: Optional[str] = None   # optional refinement ("group by quarter")


@router.post("/generate-visual")
def generate_visual(request: GenerateVisualRequest):
    """
    Builds one chart or pivot table from document data. Returns JSON the
    frontend renders directly (Recharts for charts, a table for pivots).
    """
    try:
        if request.visual_type not in VALID_VISUAL_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"visual_type must be one of {VALID_VISUAL_TYPES}"
            )
        if not request.chunks:
            raise HTTPException(
                status_code=400,
                detail="chunks cannot be empty — run /query first and pass its chunks."
            )

        context = "\n\n---\n\n".join(request.chunks[:12])

        if request.visual_type == "chart":
            type_rule = 'You MUST return visual_type "chart".'
        elif request.visual_type == "pivot_table":
            type_rule = 'You MUST return visual_type "pivot_table".'
        else:
            type_rule = ('Choose the best visual_type for the data: "chart" for trends/'
                         'comparisons of numeric series, "pivot_table" for categorical '
                         'breakdowns with multiple attributes.')

        system_prompt = f"""You turn company document data into ONE clear visual for executives.
{type_rule}

Respond ONLY with valid JSON, no markdown fences, in ONE of these shapes:

For a chart:
{{
  "visual_type": "chart",
  "title": "insight-driven title (a finding, not a topic)",
  "chart": {{
    "type": "column" | "bar" | "line" | "area" | "pie",
    "labels": ["label1", "label2"],
    "datasets": [ {{"label": "series name", "data": [1, 2]}} ]
  }},
  "insight": "one-sentence takeaway from this data",
  "source_note": "which document(s) the numbers come from"
}}

For a pivot table:
{{
  "visual_type": "pivot_table",
  "title": "insight-driven title",
  "columns": ["Column A", "Column B"],
  "rows": [ ["value", "value"] ],
  "insight": "one-sentence takeaway",
  "source_note": "which document(s) the data comes from"
}}

Rules:
- Use ONLY numbers and facts found in the source content. Mark estimates with "(est.)".
- Every dataset's data array must match labels length (use null for missing points).
- Pie charts: single dataset only, max 6 slices.
- Pivot tables: max 8 columns, max 15 rows; first column is the row dimension.
- If the source content has no usable data for the request, return:
  {{"visual_type": "none", "reason": "short explanation of what data is missing"}}"""

        instruction = f"\nRefinement from the user: {request.instruction}" if request.instruction else ""
        user_prompt = f"""Create a visual answering: "{request.query}"{instruction}

Source content from company documents:
{context}

Respond only with the JSON object."""

        data = ai.chat_json(
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            max_tokens=2000,
            temperature=0.2,
        )

        vt = data.get("visual_type")
        if vt == "none":
            return {"visual_type": "none", "reason": data.get("reason", "No usable data found.")}

        if vt == "chart":
            chart = data.get("chart") or {}
            if chart.get("type") not in VALID_CHART_TYPES:
                chart["type"] = "column"
            labels = chart.get("labels") or []
            # Pad/trim datasets so the frontend never gets ragged arrays
            for ds in chart.get("datasets") or []:
                d = list(ds.get("data") or [])
                ds["data"] = (d + [None] * len(labels))[:len(labels)]
            data["chart"] = chart
        elif vt == "pivot_table":
            cols = [str(c) for c in (data.get("columns") or [])][:8]
            rows = [
                ([str(v) if v is not None else "" for v in row] + [""] * len(cols))[:len(cols)]
                for row in (data.get("rows") or [])[:15]
            ]
            if not cols or not rows:
                return {"visual_type": "none", "reason": "The AI could not build a table from this data."}
            data["columns"], data["rows"] = cols, rows
        else:
            raise HTTPException(status_code=500, detail=f"AI returned unknown visual_type: {vt}")

        return data

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        import traceback
        print(f"GENERATE-VISUAL ERROR: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Visual generation failed: {e}")


@router.get("/workspace-stats/{workspace_id}")
async def workspace_stats(workspace_id: str):
    """Documents + chunks indexed in the vector DB for this workspace."""
    try:
        result = supabase.rpc("workspace_doc_stats",
                              {"filter_workspace_id": workspace_id}).execute()
        row = (result.data or [{}])[0]
        return {
            "workspace_id":       workspace_id,
            "documents_indexed":  row.get("documents", 0),
            "chunks_indexed":     row.get("chunks", 0),
        }
    except Exception as e:
        print(f"WORKSPACE-STATS ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Stats failed: {e}")
