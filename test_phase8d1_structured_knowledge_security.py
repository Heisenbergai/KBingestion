"""
Phase 8D.1 -- structured-knowledge sensitivity hardening.

WHAT THIS LOCKS DOWN. `graph_query.get_structured_knowledge_graph` and the
two endpoint-label resolvers used to read a claim's STATEMENT with no
sensitivity check, so a restricted claim's text was reachable by any caller
that had (or could guess) its id, and by any caller traversing a
relationship that pointed at it.

Phase 8D closed this at the dashboard endpoint. That was necessary but not
sufficient: the defect was in the shared primitive, and `impact_analysis`
was independently affected -- it uses the claim's statement as an ImpactNode
label without pre-filtering, so restricted text surfaced in impact paths in
BOTH directions.

Every test builds its own isolated workspace and cleans up after itself.

Run with: python -m pytest test_phase8d1_structured_knowledge_security.py -v
"""
import asyncio
import inspect
import uuid

import pytest
from fastapi import HTTPException

from query import supabase
from auth import AuthContext
import graph_query as gq
import impact_analysis as ia
import wiki_projection as wp
import dashboard_brain_api as api

LOW = gq.resolve_allowed_sensitivities(None, False)          # public + internal
ADMIN = gq.resolve_allowed_sensitivities("admin", False)     # + confidential
OWNER = gq.resolve_allowed_sensitivities("owner", False)     # + restricted

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"


class Fixture:
    """An isolated workspace with claims at any sensitivity, one entity, and
    a relationship whose OWN evidence is public -- so the relationship stays
    visible to a low-clearance caller and the only thing that can leak is the
    claim's text."""

    def __init__(self):
        self.ws = str(uuid.uuid4())
        self.sk, self.ent, self.rel = [], [], []

    def claim(self, statement, sensitivity):
        i = supabase.table("structured_knowledge").insert({
            "workspace_id": self.ws, "canonical_source_type": "knowledge_note",
            "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
            "primitive_type": "fact", "statement": statement, "raw_subject_phrase": "x",
            "qualifier_words": [], "sensitivity": sensitivity, "authority": "official",
            "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
            "captured_at": "2026-08-19T00:00:00Z", "extraction_run_id": str(uuid.uuid4()),
            "primitive_fingerprint": f"t8d1-{uuid.uuid4()}"}).execute().data[0]["id"]
        self.sk.append(i)
        return i

    def entity(self, label="Product"):
        i = supabase.table("knowledge_entities").insert({
            "workspace_id": self.ws, "entity_type": "department",
            "canonical_label": label, "status": "active"}).execute().data[0]["id"]
        self.ent.append(i)
        return i

    def link(self, src, src_t, tgt, tgt_t, evidence_sk, rtype="requires_approval_from"):
        i = supabase.table("knowledge_relationships").insert({
            "workspace_id": self.ws, "source_object_type": src_t, "source_object_id": src,
            "target_object_type": tgt_t, "target_object_id": tgt,
            "relationship_type": rtype, "status": "active",
            "valid_from": "2026-08-01T00:00:00Z"}).execute().data[0]["id"]
        self.rel.append(i)
        supabase.table("knowledge_relationship_evidence").insert({
            "workspace_id": self.ws, "relationship_id": i,
            "evidence_type": "structured_knowledge", "evidence_id": evidence_sk,
            "stance": "supports", "captured_at": "2026-08-19T00:00:00Z"}).execute()
        return i

    def cleanup(self):
        for i in self.rel:
            supabase.table("knowledge_relationship_evidence").delete().eq("relationship_id", i).execute()
            supabase.table("knowledge_relationships").delete().eq("id", i).execute()
        for i in self.ent:
            supabase.table("knowledge_entities").delete().eq("id", i).execute()
        for i in self.sk:
            supabase.table("structured_knowledge").delete().eq("id", i).execute()


@pytest.fixture
def fx():
    f = Fixture()
    try:
        yield f
    finally:
        f.cleanup()


# =====================================================================
# 1-3. The ladder is honoured at every level.
# =====================================================================

def test_public_claim_visible_to_everyone(fx):
    sk = fx.claim("PUBLIC-8D1 anyone may read this", "public")
    for ladder in (LOW, ADMIN, OWNER):
        g = gq.get_structured_knowledge_graph(sk, fx.ws, ladder)
        assert g is not None and "PUBLIC-8D1" in g["statement"]


def test_internal_claim_visible_to_internal_caller(fx):
    sk = fx.claim("INTERNAL-8D1 ordinary company knowledge", "internal")
    assert gq.get_structured_knowledge_graph(sk, fx.ws, LOW) is not None
    assert gq.get_structured_knowledge_graph(sk, fx.ws, OWNER) is not None


def test_confidential_respects_the_admin_boundary(fx):
    sk = fx.claim("CONFIDENTIAL-8D1 admin and above", "confidential")
    assert gq.get_structured_knowledge_graph(sk, fx.ws, LOW) is None
    assert gq.get_structured_knowledge_graph(sk, fx.ws, ADMIN) is not None
    assert gq.get_structured_knowledge_graph(sk, fx.ws, OWNER) is not None


# =====================================================================
# 4-7. Restricted content must not escape by any route.
# =====================================================================

def test_restricted_claim_hidden_from_low_clearance(fx):
    sk = fx.claim("RESTRICTED-8D1 the secret gate", "restricted")
    assert gq.get_structured_knowledge_graph(sk, fx.ws, LOW) is None
    authorized = gq.get_structured_knowledge_graph(sk, fx.ws, OWNER)
    assert authorized is not None and "RESTRICTED-8D1" in authorized["statement"]


def test_restricted_statement_never_appears_in_any_field(fx):
    """Not in a statement, not in a label, not in a rationale, not in a
    count -- the whole returned structure is searched."""
    sk = fx.claim("RESTRICTED-8D1 the secret gate", "restricted")
    pub = fx.claim("PUBLIC-8D1 relationship evidence", "public")
    ent = fx.entity()
    fx.link(sk, "structured_knowledge", ent, "entity", pub)

    assert gq.get_structured_knowledge_graph(sk, fx.ws, LOW) is None
    assert "RESTRICTED-8D1" not in str(gq.get_entity_graph(ent, fx.ws, LOW))


def test_restricted_endpoint_label_is_withheld_not_disclosed(fx):
    """The relationship itself stays visible -- its own evidence is public,
    which is the existing Phase 5 rule and deliberately unchanged here. What
    must disappear is the claim's TEXT as the endpoint label."""
    sk = fx.claim("RESTRICTED-8D1 the secret gate", "restricted")
    pub = fx.claim("PUBLIC-8D1 relationship evidence", "public")
    ent = fx.entity()
    rel_id = fx.link(sk, "structured_knowledge", ent, "entity", pub)

    low = gq.get_relationship(rel_id, fx.ws, LOW)
    assert low is not None, "relationship visibility must not change"
    assert low.source.label is None
    assert low.target.label == "Product"

    owner = gq.get_relationship(rel_id, fx.ws, OWNER)
    assert "RESTRICTED-8D1" in owner.source.label


def test_restricted_claim_does_not_leak_through_impact_paths(fx):
    """Both traversal directions. This is the path Phase 8D missed, and the
    reason the fix had to move into the primitive."""
    sk = fx.claim("RESTRICTED-8D1 the secret gate", "restricted")
    pub = fx.claim("PUBLIC-8D1 relationship evidence", "public")
    ent = fx.entity()
    fx.link(sk, "structured_knowledge", ent, "entity", pub)

    outward = ia.analyze_impact("structured_knowledge", sk, fx.ws, LOW, max_hops=1)
    assert outward.paths == [], "an invisible origin must produce no paths"

    inward = ia.analyze_impact("entity", ent, fx.ws, LOW, max_hops=1)
    assert "RESTRICTED-8D1" not in str(inward)

    authorized = ia.analyze_impact("entity", ent, fx.ws, OWNER, max_hops=1)
    assert any("RESTRICTED-8D1" in str(p.target.label) for p in authorized.paths)


# =====================================================================
# 8-9. Workspace and existence semantics unchanged.
# =====================================================================

def test_wrong_workspace_returns_nothing(fx):
    sk = fx.claim("PUBLIC-8D1 anyone may read this", "public")
    assert gq.get_structured_knowledge_graph(sk, str(uuid.uuid4()), OWNER) is None


def test_hidden_and_nonexistent_are_indistinguishable(fx):
    hidden = fx.claim("RESTRICTED-8D1 the secret gate", "restricted")
    missing = str(uuid.uuid4())
    assert gq.get_structured_knowledge_graph(hidden, fx.ws, LOW) is None
    assert gq.get_structured_knowledge_graph(missing, fx.ws, LOW) is None


# =====================================================================
# 10-13. Caller compatibility -- unchanged where it should be.
# =====================================================================

def test_visible_claims_behave_exactly_as_before():
    """The real corpus is entirely public/internal, so every existing caller
    must see precisely what it saw before this change."""
    rows = supabase.table("structured_knowledge").select("id,statement") \
        .eq("workspace_id", REAL_WORKSPACE).limit(5).execute().data or []
    assert rows, "real corpus should have claims"
    for r in rows:
        g = gq.get_structured_knowledge_graph(r["id"], REAL_WORKSPACE, OWNER)
        assert g is not None
        assert g["statement"] == r["statement"]


def test_wiki_projection_still_builds_real_pages():
    pages = wp.list_available_pages(REAL_WORKSPACE, OWNER, None)
    assert pages, "wiki must still enumerate real pages"
    built = sum(
        1 for p in pages[:4]
        if wp.build_page(p["page_type"], p["object_id"], REAL_WORKSPACE, OWNER, None) is not None
    )
    assert built > 0, "wiki page building must be unaffected"


def test_impact_analysis_still_works_on_real_data():
    ent = supabase.table("knowledge_entities").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "department") \
        .limit(1).execute().data[0]["id"]
    res = ia.analyze_impact("entity", ent, REAL_WORKSPACE, OWNER, max_hops=2)
    assert res.relationships_examined >= 0
    assert isinstance(res.paths, list)


def test_dashboard_drilldown_still_resolves_visible_evidence():
    sk = supabase.table("structured_knowledge").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).limit(1).execute().data[0]["id"]
    auth = AuthContext(user_id="u", workspaces={REAL_WORKSPACE: "owner"},
                       enforced=True, caller="pytest")
    d = asyncio.run(api.drilldown(api.DrillRequest(
        workspace_id=REAL_WORKSPACE, dataset="evidence",
        object_kind="structured_knowledge", object_id=sk), auth))
    assert d["header"]["label"]


# =====================================================================
# 14-15. No bypass, and the fixture leaves nothing behind.
# =====================================================================

def test_no_authorization_bypass_through_the_primitive(fx):
    """The endpoint gate and the primitive gate are now independent. Calling
    the primitive DIRECTLY -- the shortcut a future caller would accidentally
    take -- must not return restricted content either."""
    sk = fx.claim("RESTRICTED-8D1 the secret gate", "restricted")

    assert gq.get_structured_knowledge_graph(sk, fx.ws, LOW) is None

    auth = AuthContext(user_id="u", workspaces={fx.ws: "member"},
                       enforced=True, caller="pytest")
    with pytest.raises(HTTPException) as e:
        asyncio.run(api.drilldown(api.DrillRequest(
            workspace_id=fx.ws, dataset="evidence",
            object_kind="structured_knowledge", object_id=sk), auth))
    assert e.value.status_code == 404


def test_every_label_resolver_requires_the_ceiling():
    """Structural: neither resolver may be called without a ceiling, so the
    defect cannot be reintroduced by a future caller simply forgetting."""
    for fn in (gq._resolve_endpoint_label, gq._resolve_endpoint_labels_batch):
        params = inspect.signature(fn).parameters
        assert "allowed_sensitivities" in params, f"{fn.__name__} must take the ceiling"
        assert params["allowed_sensitivities"].default is inspect.Parameter.empty, \
            f"{fn.__name__} ceiling must be required, not optional"


def test_fixture_cleanup_leaves_no_residue():
    f = Fixture()
    sk = f.claim("temp-8d1", "internal")
    assert supabase.table("structured_knowledge").select("id").eq("id", sk).execute().data
    f.cleanup()
    assert (supabase.table("structured_knowledge").select("id")
            .eq("workspace_id", f.ws).execute().data or []) == []
