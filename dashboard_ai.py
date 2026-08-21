"""
Phase 8G -- natural language to a VALIDATED dashboard, and grounded widget
explanation.

THE ARCHITECTURE, IN ONE LINE:

    model PROPOSES  ->  registry VALIDATES  ->  Brain API EXECUTES

The model is never the source of truth. It writes a proposal; every dataset,
field, aggregation, bucket, series, ranking and visualization in that proposal
is then checked against the same `semantic_datasets` registry the Brain API
itself validates against. Anything the registry does not recognise is dropped
or the widget is rejected -- there is no "just try it and see if the API
accepts it", because a 400 at query time would mean an invalid widget already
reached a dashboard.

WHY VALIDATION IS A PURE FUNCTION. `validate_intent()` takes a dict and
returns a verdict. It calls no model and touches no network, so the part that
actually protects the product is exhaustively testable without credentials,
without mocks, and without flakiness. The model-calling wrapper around it is
thin by design.

SECURITY ORDER (Part 15) is fixed and non-negotiable:

    authenticate -> workspace -> sensitivity ceiling -> Brain query
    -> visible result -> MODEL

The model receives only what the caller was already allowed to see. It never
decides what is visible, never sees a restricted statement, and never sees a
hidden row count. `explain_widget` takes an already-resolved response and
cannot fetch anything itself.

WHAT THE MODEL MAY NOT DO (Parts 13/27):
  * calculate a number -- every value comes from the real query result;
  * assert causation that evidence does not establish;
  * change sharing, permissions, memory, the graph, the Wiki, or source data.
A proposal is a draft until a human applies it.

PROVIDER: the currently approved KNOVA path -- Amazon Nova Lite on Bedrock via
`ai.chat_json`. No Anthropic, deliberately: the previously exposed
ANTHROPIC_API_KEY must be rotated before any Anthropic model is introduced,
and this phase does not unblock that.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import semantic_datasets as sd

# Layout vocabulary, mirrored from the frontend's own fixed enums so a model
# can never propose a width the grid cannot render (Part 8).
VALID_SPANS = (2, 3, 4, 6, 8, 12)
VALID_HEIGHTS = ("short", "medium", "tall", "auto")
DEFAULT_SPAN, DEFAULT_HEIGHT = 6, "tall"

MAX_WIDGETS = 8

# Prose for the interpretation summary. Local on purpose -- the registry stays
# free of presentation concerns, and this is not a second source of truth for
# which aggregations exist (ALLOWED_AGGREGATIONS remains the only one).
_AGG_PROSE = {
    "count": "counted", "count_distinct": "counted uniquely",
    "min": "earliest", "max": "latest",
}

# Every key a widget proposal may carry. An unknown key is a signal the model
# invented something, so it is reported rather than silently ignored.
WIDGET_KEYS = {
    "dataset", "fields", "group_by", "group_bucket", "series_by", "series_bucket",
    "aggregation", "value_field", "top_n", "top_direction", "percent",
    "temporal_mode", "window_days", "as_of", "compare", "visualization",
    "title", "span", "height",
}

# Keys that must NEVER appear. These are not merely unknown -- each would be an
# attempt to reach past the semantic layer into SQL, storage, or authorization.
FORBIDDEN_KEYS = {
    "sql", "query", "raw_sql", "table", "table_name", "column", "columns",
    "role", "is_super_admin", "allowed_sensitivities", "sensitivity_ceiling",
    "workspace_id", "user_id", "object_id", "code", "eval", "exec",
}


@dataclass
class WidgetProposal:
    config: dict
    title: str
    span: int = DEFAULT_SPAN
    height: str = DEFAULT_HEIGHT


@dataclass
class IntentValidation:
    """The verdict. `widgets` are safe to render; `rejected` explains, per
    widget, exactly why it did not survive -- so the UI can tell a user what
    KNOVA could not do rather than silently showing fewer widgets."""
    dashboard_name: str
    widgets: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    clarification_needed: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.widgets)


def _clamp_span(v) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return DEFAULT_SPAN
    # Snap to the nearest legal width rather than rejecting the widget: an
    # out-of-range span is a cosmetic mistake, not a semantic one.
    return min(VALID_SPANS, key=lambda s: abs(s - n))


def _clamp_height(v) -> str:
    return v if v in VALID_HEIGHTS else DEFAULT_HEIGHT


def validate_widget(raw: dict) -> tuple[Optional[WidgetProposal], Optional[str]]:
    """Validates ONE proposed widget against the registry. Pure: no model, no
    network, no database. Returns (proposal, None) or (None, reason)."""
    if not isinstance(raw, dict):
        return None, "Widget was not an object."

    present_forbidden = FORBIDDEN_KEYS & set(raw)
    if present_forbidden:
        # Not a stray key -- an attempt to reach past the semantic layer.
        return None, f"Widget contained forbidden keys: {sorted(present_forbidden)}."

    unknown = set(raw) - WIDGET_KEYS
    if unknown:
        return None, f"Widget contained unknown keys: {sorted(unknown)}."

    dataset_key = raw.get("dataset")
    try:
        ds = sd.get_dataset(dataset_key)
    except sd.DatasetError:
        if isinstance(dataset_key, str) and "project" in dataset_key.lower():
            return None, ("KNOVA does not currently have a verified Project dataset, "
                           "so a project widget cannot be built.")
        return None, f"Unknown dataset {dataset_key!r}."

    cfg: dict = {"dataset": ds.key}

    # Fields -- silently drop unknown ones rather than failing the widget, but
    # say so, because a model naming one bad field among five is a partial
    # success worth keeping.
    dropped_fields = []
    if isinstance(raw.get("fields"), list):
        keep = []
        for f in raw["fields"]:
            (keep if ds.field(f) else dropped_fields).append(f)
        if keep:
            cfg["fields"] = keep

    for key in ("group_by", "series_by", "value_field"):
        v = raw.get(key)
        if v is not None:
            if ds.field(v) is None:
                return None, f"{key} names unknown field {v!r} on {ds.key!r}."
            cfg[key] = v
    for key in ("group_bucket", "series_bucket"):
        if raw.get(key) is not None:
            cfg[key] = raw[key]

    if raw.get("aggregation") is not None:
        if raw["aggregation"] not in sd.ALLOWED_AGGREGATIONS:
            return None, (f"Unsupported aggregation {raw['aggregation']!r}. "
                           f"Allowed: {sorted(sd.ALLOWED_AGGREGATIONS)}.")
        cfg["aggregation"] = raw["aggregation"]

    if raw.get("top_n") is not None:
        cfg["top_n"] = raw["top_n"]
        cfg["top_direction"] = raw.get("top_direction") or "top"
    if raw.get("percent"):
        cfg["percent"] = True

    mode = raw.get("temporal_mode") or sd.MODE_CURRENT
    if mode not in ds.temporal_modes:
        return None, (f"Dataset {ds.key!r} does not support temporal mode {mode!r}. "
                       f"Supported: {list(ds.temporal_modes)}.")
    cfg["temporal_mode"] = mode
    if mode == sd.MODE_WINDOW:
        cfg["window_days"] = raw.get("window_days") or 30
    if mode == sd.MODE_AS_OF:
        if not raw.get("as_of"):
            return None, "temporal_mode 'as_of' requires as_of."
        cfg["as_of"] = raw["as_of"]
    if raw.get("compare"):
        if mode != sd.MODE_WINDOW:
            return None, "A comparison requires a time window."
        cfg["compare"] = True

    viz = raw.get("visualization") or ds.default_visualization
    if viz not in _VALID_VISUALIZATIONS:
        return None, f"Unknown visualization {viz!r}."
    cfg["visualization"] = viz
    cfg["drilldown"] = True

    # THE DECISIVE CHECK. Everything above verified names; this verifies the
    # COMBINATION, by running the registry's own aggregation validator over an
    # empty row set. If the real API would reject this widget, it is rejected
    # here -- so an invalid widget can never reach a dashboard.
    try:
        sd._aggregate(
            [], ds, cfg.get("group_by"), cfg.get("aggregation"), cfg.get("value_field"),
            cfg.get("group_bucket"), series_by=cfg.get("series_by"),
            series_bucket=cfg.get("series_bucket"), top_n=cfg.get("top_n"),
            top_direction=cfg.get("top_direction", "top"), percent=cfg.get("percent", False),
        )
    except sd.DatasetError as e:
        return None, str(e)

    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        title = ds.label
    cfg["title"] = title.strip()

    note = (f" ({len(dropped_fields)} unrecognised field(s) dropped)" if dropped_fields else "")
    return WidgetProposal(
        config=cfg, title=cfg["title"],
        span=_clamp_span(raw.get("span", DEFAULT_SPAN)),
        height=_clamp_height(raw.get("height", DEFAULT_HEIGHT)),
    ), (None if not dropped_fields else f"__note__{note}")


_VALID_VISUALIZATIONS = frozenset({
    "kpi", "table", "bar", "line", "area", "timeline", "summary", "donut",
})


def validate_intent(raw: dict) -> IntentValidation:
    """Validates a whole proposed dashboard. Pure and deterministic."""
    if not isinstance(raw, dict):
        return IntentValidation(dashboard_name="Untitled",
                                rejected=[{"reason": "Model output was not an object."}])

    name = raw.get("dashboard_name")
    name = name.strip() if isinstance(name, str) and name.strip() else "New dashboard"

    out = IntentValidation(dashboard_name=name)

    clar = raw.get("clarification_needed")
    if isinstance(clar, str) and clar.strip():
        out.clarification_needed = clar.strip()

    widgets = raw.get("widgets")
    if not isinstance(widgets, list) or not widgets:
        if not out.clarification_needed:
            out.rejected.append({"reason": "Model proposed no widgets."})
        return out

    for i, w in enumerate(widgets[:MAX_WIDGETS]):
        proposal, reason = validate_widget(w)
        if proposal is None:
            out.rejected.append({
                "index": i,
                "requested": (w.get("dataset") if isinstance(w, dict) else None),
                "reason": reason,
            })
            continue
        if reason and reason.startswith("__note__"):
            out.notes.append(f"{proposal.title}{reason[len('__note__'):]}")
        out.widgets.append(proposal)

    if len(widgets) > MAX_WIDGETS:
        out.notes.append(
            f"Only the first {MAX_WIDGETS} proposed widgets were kept.")
    return out


# =====================================================================
# Natural language -> intent. The ONLY place a model is asked to propose.
# =====================================================================

def _registry_summary() -> str:
    """A compact description of what genuinely exists, so the model chooses
    from real options instead of guessing. Bounded on purpose (Part 19): the
    schema only, never workspace data."""
    lines = []
    for ds in sd.DATASETS.values():
        groupable = [f.key for f in ds.fields if f.groupable]
        lines.append(
            f"- {ds.key}: {ds.description} | modes={list(ds.temporal_modes)} "
            f"| groupable={groupable}")
    return "\n".join(lines)


_SYSTEM = """You translate a request into a KNOVA dashboard configuration.

You may ONLY use datasets and fields from the list given to you. You must not
invent a dataset, a field, a table, a column, or SQL. If the request asks for
something KNOVA has no dataset for -- projects, tasks, revenue, headcount --
do not substitute something else: leave it out and say so in `unavailable`.

If the request is too vague to configure (for example "show me activity",
which could mean changes, meetings, evidence or attention), set
`clarification_needed` to ONE short question and return no widgets.

Aggregations allowed: count, count_distinct, min, max. Nothing else.
Buckets allowed: day, week, month, quarter, year -- only for date fields.
Visualizations: kpi, table, bar, line, area, timeline, summary, donut.

Respond ONLY with JSON:
{"dashboard_name": str,
 "widgets": [{"dataset": str, "group_by": str|null, "group_bucket": str|null,
              "series_by": str|null, "aggregation": str|null, "top_n": int|null,
              "percent": bool, "temporal_mode": str, "window_days": int|null,
              "visualization": str, "title": str, "span": int, "height": str}],
 "unavailable": [str],
 "clarification_needed": str|null}"""


def generate_dashboard(request: str, chat_json_fn: Callable = None,
                       workspace_id: str = None, user_id: str = None) -> dict:
    """Natural language -> a validated DRAFT (Part 7/20).

    Returns a draft the caller may render and a human may apply. It creates
    nothing, overwrites nothing, and changes no sharing or permission -- those
    remain explicit user actions.

    A model failure is a structured error, never a fabricated dashboard: the
    manual builder stays fully usable, because the Studio must never depend on
    the model (Part 17)."""
    if chat_json_fn is None:
        import ai
        chat_json_fn = ai.chat_json

    try:
        raw = chat_json_fn(
            messages=[{"role": "user", "content":
                        f"Available datasets:\n{_registry_summary()}\n\nRequest: {request}"}],
            system=_SYSTEM, max_tokens=1500, temperature=0.1,
            workspace_id=workspace_id, user_id=user_id,
            feature="dashboard_ai_intent",
        )
    except Exception as e:
        return {
            "ok": False,
            "error": "KNOVA couldn't generate a dashboard right now.",
            "detail": type(e).__name__,
            "fallback": "manual_builder",
            "draft": None,
        }

    verdict = validate_intent(raw if isinstance(raw, dict) else {})
    unavailable = [u for u in (raw.get("unavailable") or [])
                   if isinstance(u, str)] if isinstance(raw, dict) else []

    return {
        "ok": verdict.ok,
        "draft": {
            "dashboard_name": verdict.dashboard_name,
            "widgets": [{"config": w.config, "title": w.title,
                          "span": w.span, "height": w.height} for w in verdict.widgets],
        } if verdict.ok else None,
        "interpretation": [_describe(w) for w in verdict.widgets],
        "rejected": verdict.rejected,
        "unavailable": unavailable,
        "notes": verdict.notes,
        "clarification_needed": verdict.clarification_needed,
        "fallback": None if verdict.ok else "manual_builder",
    }


def _describe(w: WidgetProposal) -> str:
    """Plain-English restatement of what KNOVA understood, so a user can see
    the interpretation before applying it (Part 9)."""
    c = w.config
    ds = sd.get_dataset(c["dataset"])
    bits = [ds.label]
    if c.get("group_by"):
        f = ds.field(c["group_by"])
        bits.append(f"grouped by {f.label if f else c['group_by']}"
                    + (f" by {c['group_bucket']}" if c.get("group_bucket") else ""))
    if c.get("series_by"):
        f = ds.field(c["series_by"])
        bits.append(f"split by {f.label if f else c['series_by']}")
    if c.get("aggregation"):
        bits.append(_AGG_PROSE.get(c["aggregation"], c["aggregation"]))
    if c.get("top_n"):
        bits.append(f"{c.get('top_direction', 'top')} {c['top_n']}")
    if c.get("percent"):
        bits.append("as percentages")
    if c.get("temporal_mode") == sd.MODE_WINDOW:
        bits.append(f"last {c.get('window_days', 30)} days")
    elif c.get("temporal_mode") == sd.MODE_AS_OF:
        bits.append(f"as of {c.get('as_of')}")
    bits.append(f"shown as {c['visualization']}")
    return f"{w.title} - " + ", ".join(bits[1:]) if len(bits) > 1 else w.title


# =====================================================================
# Explain -- grounded, claim-bounded, and incapable of inventing a number.
# =====================================================================

# Words that assert one thing MADE another happen. The Brain can establish
# that two things are connected and that something changed; it cannot
# establish cause. An explanation using these is rewritten, not patched.
_CAUSAL_TERMS = (
    "caused", "because of", "due to", "led to", "resulted in", "drove",
    "triggered", "as a result of", "responsible for", "blame", "owing to",
)

_EXPLAIN_SYSTEM = """You explain what a KNOVA dashboard widget shows, using ONLY
the facts given to you.

Rules you must not break:
- Never state a number that is not in the facts. Never compute a total, a
  difference, a percentage or a rate yourself.
- Never say one thing caused another. KNOVA can show that things are connected
  and that something changed; it cannot establish cause. If a cause is not
  given, write "A cause is not established."
- Never name a person, department, policy or entity that is not in the facts.
- If the facts are for a past date, speak about that date, not about now.

Respond ONLY as JSON:
{"observed": str, "derived": str|null, "connections": str|null, "unknown": str}

observed    = what the widget literally shows, using only given numbers.
derived     = a comparison that follows from the given numbers, or null.
connections = verified relationships from the facts, or null.
unknown     = what the evidence does NOT establish."""


@dataclass
class WidgetExplanation:
    observed: str
    derived: Optional[str]
    connections: Optional[str]
    unknown: str
    temporal_context: str
    grounded: bool
    source: str                      # "model" | "deterministic"
    rejected_reasons: list = field(default_factory=list)


def _numbers_in(text: str) -> set:
    import re
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _allowed_numbers(response: dict, comparison: Optional[dict]) -> set:
    """Every number the model may repeat: the ones the real query actually
    produced. Anything else was invented (Part 13)."""
    nums = set()

    def add(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            nums.add(str(v))
            if float(v).is_integer():
                nums.add(str(int(v)))

    add(response.get("row_count"))
    agg = response.get("aggregation") or {}
    add(agg.get("percent_basis"))
    add(agg.get("top_n"))
    for b in agg.get("buckets") or []:
        add(b.get("value"))
        add(b.get("row_count"))
        add(b.get("percent"))
        # Years/quarters/weeks in a REAL bucket label are legitimate to quote.
        nums |= _numbers_in(str(b.get("group", "")))
        nums |= _numbers_in(str(b.get("series", "")))
    if comparison:
        add(comparison.get("value"))
        cur = (agg.get("buckets") or [{}])[0].get("value")
        if isinstance(cur, (int, float)) and isinstance(comparison.get("value"), (int, float)):
            # A delta between two REAL periods is real; the model may state it.
            add(abs(cur - comparison["value"]))
    nums |= _numbers_in(str(response.get("temporal_context", "")))
    return {n for n in nums if n}


def _validate_explanation(parts: dict, response: dict,
                          comparison: Optional[dict]) -> tuple[bool, list]:
    """The claim boundary applied to generated prose.

    Same discipline as the Phase 6F Wiki validator: the presence of plausible
    words is not enough. Every NUMBER must be one the query really produced,
    and no sentence may assert a cause the evidence does not establish."""
    reasons = []
    allowed = _allowed_numbers(response, comparison)
    for key in ("observed", "derived", "connections", "unknown"):
        text = parts.get(key)
        if not isinstance(text, str):
            continue
        invented = _numbers_in(text) - allowed
        if invented:
            reasons.append(f"{key}: invented number(s) {sorted(invented)}")
        low = text.lower()
        hit = next((t for t in _CAUSAL_TERMS if t in low), None)
        if hit:
            reasons.append(f"{key}: asserts causation ({hit!r})")
    return (not reasons), reasons


def _deterministic_explanation(response: dict, config: dict) -> WidgetExplanation:
    """The fallback, and the floor. Built only from real values, so it is
    always safe -- a model failure costs richness, never correctness."""
    agg = response.get("aggregation") or {}
    buckets = agg.get("buckets") or []
    temporal = response.get("temporal_context", "current")
    when = "currently" if temporal == "current" else f"as of {temporal}"

    if buckets and agg.get("group_by"):
        top = max(buckets, key=lambda b: (b.get("value") or 0))
        observed = (f"This shows {response.get('row_count', 0)} record(s) {when}, "
                    f"grouped by {agg['group_by']}. The largest group is "
                    f"{top.get('group')} with {top.get('value')}.")
    else:
        observed = f"This shows {response.get('row_count', 0)} record(s) {when}."

    unknown_bits = list(response.get("not_established") or [])
    unknown = (unknown_bits[0] if unknown_bits
               else "A cause for this pattern is not established by the evidence.")
    return WidgetExplanation(
        observed=observed, derived=None, connections=None, unknown=unknown,
        temporal_context=temporal, grounded=True, source="deterministic",
    )


def explain_widget(response: dict, config: dict, comparison: Optional[dict] = None,
                   detail: Optional[dict] = None, chat_json_fn: Callable = None,
                   workspace_id: str = None, user_id: str = None) -> WidgetExplanation:
    """Explains an ALREADY-RESOLVED widget result.

    `response` is whatever the Brain API returned for THIS caller, so the model
    only ever sees authorized data. It cannot fetch, and it does not choose
    what it is shown (Parts 11/15). On a shared dashboard this runs per viewer,
    so an owner and a viewer legitimately receive different explanations from
    their own different results (Part 16).

    Nothing is persisted -- an explanation is derived at view time (Part 21)."""
    temporal = response.get("temporal_context", "current")
    fallback = _deterministic_explanation(response, config)

    if chat_json_fn is None:
        import ai
        chat_json_fn = ai.chat_json

    # The bounded fact sheet (Part 19): only what the caller already received,
    # never the corpus and never anything the ceiling excluded.
    facts = {
        "widget_title": config.get("title"),
        "dataset": response.get("dataset"),
        "temporal_context": temporal,
        "row_count": response.get("row_count"),
        "aggregation": response.get("aggregation"),
        "not_established": response.get("not_established"),
        "undetectable_changes": (detail or {}).get("undetectable_changes"),
        "comparison": comparison,
        "verified_connections": (detail or {}).get("affected"),
        "evidence": [
            {k: e.get(k) for k in ("statement", "provider", "captured_at")}
            for e in ((detail or {}).get("evidence") or [])[:5]
        ],
    }

    try:
        raw = chat_json_fn(
            messages=[{"role": "user", "content":
                        f"Facts:\n{facts}\n\nExplain this widget."}],
            system=_EXPLAIN_SYSTEM, max_tokens=500, temperature=0.1,
            workspace_id=workspace_id, user_id=user_id,
            feature="dashboard_ai_explain",
        )
    except Exception:
        # Model unavailable: the widget still works and still explains itself.
        return fallback

    if not isinstance(raw, dict):
        fallback.rejected_reasons = ["Model output was not an object."]
        return fallback

    ok, reasons = _validate_explanation(raw, response, comparison)
    if not ok:
        # Discarded entirely rather than patched: a half-trusted explanation
        # is worse than a plain one, because the reader cannot tell which half.
        fallback.rejected_reasons = reasons
        return fallback

    return WidgetExplanation(
        observed=str(raw.get("observed") or fallback.observed),
        derived=(str(raw["derived"]) if raw.get("derived") else None),
        connections=(str(raw["connections"]) if raw.get("connections") else None),
        unknown=str(raw.get("unknown") or fallback.unknown),
        temporal_context=temporal, grounded=True, source="model",
    )
