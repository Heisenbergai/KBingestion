"""
Escalation triage — decides which unanswered bot questions deserve a human, and
how urgently.

WHY THIS EXISTS. Every question a bot couldn't answer used to land in the admin
queue as "please answer", including "hy i want assistance" and "hey what about
you". A queue full of noise is a queue nobody works, which means the real
questions rot in it.

TWO DELIBERATE BIASES, both in the same direction:
  * Rules run first and the LLM only sees what rules can't call — the same
    cost-conscious pattern ingest.classify_document uses.
  * **Every failure path returns 'actionable'.** Wrongly showing an admin one
    extra question costs a second of their attention; wrongly filing a real
    question as noise hides it from the only people who could answer, and
    nobody ever finds out. The asymmetry is not close, so the tie always goes
    to escalating.

Nothing is ever deleted. A rejected question is stored with triage
'not_actionable' AND the reason, so the filter's own decisions stay auditable
and tunable — a filter you cannot inspect is a filter you cannot trust.
"""
import re
from typing import Optional

import ai

# ── rule layer ────────────────────────────────────────────────────────────────

# Whole-message greetings and pleasantries. Matched against the ENTIRE message,
# never as a substring: "hi" must not reject "what is our hiring policy".
_GREETING_RE = re.compile(
    r"^(hi|hii+|hey+|hello+|yo|sup|hola|namaste|good\s*(morning|afternoon|evening|night)"
    r"|thanks?|thank\s*you|ty|thx|ok(ay)?|k|cool|nice|great|awesome|lol|haha+"
    r"|bye|goodbye|see\s*ya|good\s*bye|test|testing)"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)

# Questions aimed at the bot itself rather than at company knowledge. A human
# admin cannot usefully "answer" these into the knowledge base.
# NOTE the shape here, it matters twice over:
#  * ONE `^` in front of a non-capturing alternation. Anchoring each branch
#    separately (`^a|^b|c`) leaves later branches unanchored, so "are you a bot"
#    would have matched in the MIDDLE of a real question and rejected it.
#  * An optional leading greeting, because "hey what about you" is exactly the
#    message this is meant to catch and the greeting would otherwise defeat `^`.
_BOT_DIRECTED_RE = re.compile(
    r"^(?:(?:hi+|hey+|hello+|yo|hola)[\s,!.]*)?"
    r"(?:(?:what|how)\s+(?:about|are)\s+you"
    r"|who\s+are\s+you"
    r"|what(?:\s+is)?\s+your\s+name"
    r"|are\s+you\s+(?:a\s+)?(?:bot|human|ai|real)"
    r"|can\s+you\s+(?:hear|see|talk)"
    r"|how\s+(?:are|r)\s+(?:you|u)\b)",
    re.IGNORECASE,
)

# Vague pleas with no subject — "i want assistance", "help me", "can you help".
_VAGUE_HELP_RE = re.compile(
    r"^(i\s+(want|need)\s+(some\s+)?(help|assistance|support)"
    r"|(can|could)\s+(you|u)\s+help(\s+me)?"
    r"|help(\s+me)?|assist(\s+me)?|support)"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def rule_triage(question: str) -> Optional[tuple[str, str]]:
    """
    Returns (triage, reason) when the rules are CONFIDENT, else None so the LLM
    decides. Only ever returns 'not_actionable' — rules never confirm that
    something IS a real question, they only recognise clear noise.
    """
    q = _normalize(question)

    if not q:
        return ("not_actionable", "Empty message")
    if len(q) < 3:
        return ("not_actionable", "Too short to be a question")
    if _GREETING_RE.match(q):
        return ("not_actionable", "Greeting or pleasantry, not a question")
    if _BOT_DIRECTED_RE.match(q):
        return ("not_actionable", "Small talk aimed at the bot, not company knowledge")
    if _VAGUE_HELP_RE.match(q):
        return ("not_actionable", "Asks for help without saying what about")

    # A single word is almost never answerable, but a single word that is
    # clearly a company noun ("payroll", "onboarding") often is — so this only
    # rejects a lone word with no letters beyond noise.
    words = q.split()
    if len(words) == 1 and not re.search(r"[a-zA-Z]{4,}", q):
        return ("not_actionable", "Single token with no answerable subject")

    return None


_LLM_PROMPT = """You triage questions asked of a company's internal knowledge bot.

Decide whether a HUMAN COLLEAGUE could usefully answer this question by writing
something into the company knowledge base.

actionable   = a real question about the company, its policies, people, products,
               numbers or processes. Even if vague, if a colleague could answer it.
not_actionable = greetings, small talk, questions about the bot itself, tests,
               gibberish, or anything no colleague could write a useful answer to.

Reply ONLY with JSON: {"triage": "actionable"|"not_actionable", "reason": "<8 words max>"}"""


def llm_triage(question: str, workspace_id: Optional[str] = None) -> tuple[str, str]:
    """
    Fail-safe by construction: ANY problem — exception, malformed JSON,
    unexpected value — returns 'actionable'. See the module docstring on why the
    asymmetry always favours escalating.
    """
    try:
        result = ai.chat_json(
            messages=[{"role": "user", "content": f"Question: {question}"}],
            system=_LLM_PROMPT,
            max_tokens=100,
            temperature=0,
            workspace_id=workspace_id,
            feature="escalation_triage",
        )
        triage = (result or {}).get("triage")
        if triage == "not_actionable":
            reason = str((result or {}).get("reason") or "Not answerable by a colleague")
            return ("not_actionable", reason[:120])
        return ("actionable", "")
    except Exception as e:
        print(f"[triage] LLM triage failed, defaulting to actionable (non-fatal): {e}")
        return ("actionable", "")


def triage_question(question: str, workspace_id: Optional[str] = None) -> tuple[str, str]:
    """Rules first, LLM only for what rules can't call."""
    ruled = rule_triage(question)
    if ruled is not None:
        return ruled
    return llm_triage(question, workspace_id=workspace_id)


# ── priority ──────────────────────────────────────────────────────────────────

def score_priority(question: str, confidence: Optional[str], asker_role: Optional[str],
                   times_asked: int) -> str:
    """
    P1 asked by 3+ distinct people, or by owner/admin, or the bot had NO idea.
    P2 asked more than once, or the bot was merely unsure.
    P3 everything else that survived triage.

    Repeat frequency is the strongest signal available and needs nobody to tag
    anything: if three people hit the same wall, that gap is costing real time.
    """
    if times_asked >= 3:
        return "p1"
    if asker_role in ("owner", "admin"):
        return "p1"
    if confidence == "none" and times_asked >= 2:
        return "p1"
    if times_asked >= 2 or confidence == "none":
        return "p2"
    return "p3"
