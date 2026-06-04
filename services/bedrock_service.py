"""
RAG retrieval + generation against Amazon Bedrock Knowledge Bases.

The Knowledge Base stores chunks from several insurance policies side by side.
Three failure modes are explicitly engineered against here:

* Amnesia on broad queries  -> scope-aware Top-K + per-company/tier retrieval fan-out
  so every policy is represented with enough context, instead of a single vague
  search returning too few, generic chunks.
* Cross-contamination        -> chunks are grouped into clearly delimited,
  "sealed" per-policy sections (never interleaved by score) and every chunk is
  labeled with its company + tier + source file.
* Mirroring / symmetry       -> explicit empty-section placeholders for policies
  with no hits, plus strict anti-mirroring rules in the system prompt.
"""
from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from botocore.exceptions import ClientError, ReadTimeoutError

from config import (
    MODEL_ID,
    KNOWLEDGE_BASE_ID,
    SEARCH_TYPE,
    METADATA_KEY,
    COMPANY_ALIASES,
    MAX_RESULTS_PER_CALL,
    NARROW_NUMBER_OF_RESULTS,
    BROAD_NUMBER_OF_RESULTS,
    RESULTS_PER_COMPANY,
    RESULTS_PER_COMPANY_BROAD,
    MAX_TOTAL_RESULTS,
    MIN_RESULTS_PER_COMPANY,
    MAX_COMPARE_COMPANIES,
    alias_in_text,
)
from services import policy_registry
from services.aws_clients import bedrock_agent_runtime_client, bedrock_runtime_client

logger = logging.getLogger(__name__)

# Keywords that signal a comparison / gap / double-insurance analysis.
_COMPARISON_HINTS = (
    "השווא", "השוו", "השווה", "מול", "לעומת", "הבדל", "הבדלים", "כפל",
    "חופף", "חפיפ", "עדיף", "differ", "compare", "comparison", "versus",
    "vs", "gap", "overlap", "double",
)

# Keywords that signal a broad / exhaustive request (raise Top-K).
_BROAD_HINTS = (
    "כל ", "כל ה", "הכל", "כלל ה", "מלא", "מלאה", "מקיף", "סך הכל",
    "all", "every", "everything", "full", "complete", "comprehensive",
    "entire", "overall",
)

SYSTEM_PROMPT = (
    "You are an elite Israeli health-insurance analyst and data extraction AI. Provide BRIEF, PUNCHY, ACTIONABLE summaries and answers in highly professional Hebrew.\n"
    "Your ONLY output should be a clean, beautifully formatted Markdown analysis designed to help a user make a financial and medical decision.\n\n"
    "=== INPUT CONTEXT ===\n"
    "Sections are separated by '======== התחלת מסמכי פוליסה: <Company> - <Tier> ========'.\n"
    "Treat each as an impenetrable vault.\n\n"
    "=== EXTRACTION & ANALYSIS RULES ===\n"
    "1. CHAIN OF THOUGHT (CRITICAL): You MUST start your response with exactly `<thinking>` and end with `</thinking>`. Inside, QUOTE THE EXACT SENTENCES containing numbers, coverages, and conditions. This prevents hallucinations.\n"
    "2. LEXICAL GAP AWARENESS: Different companies use vastly different terminology. 'Pregnancy' in one might be 'סל הריון', while in another it's 'בדיקות היריון פרטיות', 'בדיקות גנטיות', 'סקירת מערכות', or 'הבראה לאחר לידה'. Actively search the provided context for these variations before claiming a policy lacks coverage.\n"
    "3. THE 'APPLES TO ORANGES' RULE: HMOs (Shaban) use specific service lists and copays. Private insurances use general financial umbrellas (budgets). Explain how each covers the requested topic based on its own logic. Do not blindly say 'Not covered' if it is covered under a broader budget or a specific clinic arrangement.\n"
    "4. CONDITIONS & CAVEATS: Always extract waiting periods (תקופת אכשרה), deductibles (השתתפות עצמית), and specific eligibility rules (תנאי זכאות).\n"
    "5. NO CROSS-CONTAMINATION: Never copy data, numbers, or services from one company's vault to another.\n\n"
    "=== OUTPUT STRUCTURE ===\n"
    "Do NOT use a comparison table. Build your response EXACTLY in this order:\n"
    "<thinking>\n"
    "[Your internal reasoning and exact verbatim quotes here]\n"
    "</thinking>\n"
    "1. הקדמה: A professional 1-2 sentence introduction.\n"
    "2. פירוט כיסויים - <Company 1>: Detailed, well-spaced bullet points of what the first policy covers regarding the user's query. Include exact amounts, deductibles, waiting periods, and citations like '(סעיף X)'.\n"
    "3. פירוט כיסויים - <Company 2>: Detailed, well-spaced bullet points for the second policy, just like the first.\n"
    "4. כפל ביטוח והבדלים מרכזיים: Clear bullet points explicitly comparing the two. Where are the exact differences? Where do they overlap? Does the private insurance require exhausting the HMO benefits first (מיצוי זכאות)?\n"
    "5. סיכום תמציתי (Decision Support): An objective summary to help the user understand which policy gives what advantage."
)

NO_CONTEXT_MESSAGE = (
    "לא נמצא מידע רלוונטי במסמכים. ייתכן שאינך מבוטח בנושא זה, "
    "שהפוליסה אינה במאגר, או שחסר נספח רלוונטי."
)


@dataclass
class RetrievedChunk:
    """A single retrieval result, normalized for prompt building."""
    text: str
    company: str
    tier: str | None
    filename: str
    score: float
    raw_metadata: dict[str, Any]


@dataclass
class QueryPlan:
    """The retrieval strategy derived from a single user query."""
    target_policies: list[tuple[str, str | None]]  # Pairs of (company, tier)
    is_comparison: bool
    is_broad: bool


def _get_agent_client() -> Any:
    return bedrock_agent_runtime_client()


def _get_runtime_client() -> Any:
    return bedrock_runtime_client()


# ---------------------------------------------------------------------------
# Query understanding
# ---------------------------------------------------------------------------
def detect_target_policies(query: str, known_policies: list[Any]) -> list[tuple[str, str | None]]:
    """Return specific policies (Company + Tier) referenced in the query."""
    if not known_policies:
        return []

    text = query.lower()
    matched: set[tuple[str, str | None]] = set()
    
    companies_to_tiers: dict[str, set[str | None]] = {}
    for p in known_policies:
        companies_to_tiers.setdefault(p.company, set()).add(p.tier)

    for company, tiers in companies_to_tiers.items():
        candidates = [company.lower(), *(a.lower() for a in COMPANY_ALIASES.get(company, []))]
        
        if any(alias_in_text(text, c) for c in candidates):
            tier_matched = False
            for tier in tiers:
                if tier and alias_in_text(text, tier.lower()):
                    matched.add((company, tier))
                    tier_matched = True
            
            if not tier_matched:
                for tier in tiers:
                    matched.add((company, tier))

    return list(matched)


def _is_comparison_query(query: str, named_count: int) -> bool:
    text = query.lower()
    keyword = any(hint in text for hint in _COMPARISON_HINTS)
    return keyword or named_count >= 2


def _is_broad_query(query: str) -> bool:
    text = query.lower()
    return any(hint in text for hint in _BROAD_HINTS)


def plan_query(query: str) -> QueryPlan:
    """Decide the retrieval scope and volume for a query."""
    known_policies = policy_registry.list_policies()
    named_policies = detect_target_policies(query, known_policies)
    
    is_comparison = _is_comparison_query(query, len(named_policies))
    is_broad = _is_broad_query(query)

    if is_comparison:
        if len(named_policies) >= 2:
            targets = named_policies
        else:
            targets = [(p.company, p.tier) for p in known_policies]
        targets = targets[:MAX_COMPARE_COMPANIES]
    else:
        targets = named_policies[:1]

    plan = QueryPlan(target_policies=targets, is_comparison=is_comparison, is_broad=is_broad)
    logger.info(
        "Query plan: comparison=%s broad=%s policies=%s",
        plan.is_comparison,
        plan.is_broad,
        plan.target_policies or "ALL",
    )
    return plan


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _build_metadata_filter(company: str, tier: str | None) -> dict[str, Any]:
    """Constructs a Bedrock filter for BOTH company and tier."""
    filters: list[dict[str, Any]] = [
        {"equals": {"key": METADATA_KEY, "value": company}}
    ]
    
    if tier and tier != "Unknown":
        filters.append({"equals": {"key": "tier", "value": tier}})
        
    if len(filters) == 1:
        return filters[0]
        
    return {"andAll": filters}


def _retrieve(
    query: str,
    company: str | None,
    tier: str | None,
    top_k: int,
) -> list[RetrievedChunk]:
    """Low-level single retrieval call with an optional metadata filter."""
    top_k = max(1, min(top_k, MAX_RESULTS_PER_CALL))
    vector_config: dict[str, Any] = {
        "numberOfResults": top_k,
        "overrideSearchType": SEARCH_TYPE,
    }
    
    if company:
        vector_config["filter"] = _build_metadata_filter(company, tier)

    try:
        response = _get_agent_client().retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": vector_config},
        )
    except ReadTimeoutError as exc:
        logger.error("Retrieval timed out: %s", exc)
        raise RuntimeError(
            "שליפת המידע מהמאגר ארכה זמן רב מדי. נסו שוב."
        ) from exc
    except ClientError as exc:
        message = exc.response.get("Error", {}).get("Message", str(exc))
        logger.error("Retrieval failed: %s", message)
        raise RuntimeError(f"Retrieval failed: {message}") from exc

    chunks: list[RetrievedChunk] = []
    for result in response.get("retrievalResults", []):
        text = result.get("content", {}).get("text", "")
        if not text:
            continue
        s3_uri = result.get("location", {}).get("s3Location", {}).get("uri", "")
        filename = s3_uri.split("/")[-1].replace(".md", "").replace(".pdf", "")
        metadata = result.get("metadata", {}) or {}
        
        chunk_company = metadata.get(METADATA_KEY) or company or "לא ידוע"
        chunk_tier = metadata.get("tier")
        if chunk_tier == "Unknown":
            chunk_tier = None
            
        chunks.append(
            RetrievedChunk(
                text=text,
                company=chunk_company,
                tier=str(chunk_tier) if chunk_tier is not None else None,
                filename=filename or "לא ידוע",
                score=float(result.get("score", 0.0)),
                raw_metadata=dict(metadata),
            )
        )
    return chunks


def _log_retrieved_chunks(
    groups: "OrderedDict[str, list[RetrievedChunk]]",
    query: str,
) -> None:
    """Debug log: dump every retrieved chunk and its metadata before LLM generation."""
    total = sum(len(chunks) for chunks in groups.values())
    logger.info("=" * 72)
    logger.info("RETRIEVAL DEBUG | query=%r | total_chunks=%d", query, total)
    for group_key, chunks in groups.items():
        logger.info("--- policy group: %s (%d chunks) ---", group_key, len(chunks))
        if not chunks:
            continue
        for idx, chunk in enumerate(chunks, 1):
            logger.info("  chunk #%d | score=%.4f | file=%s", idx, chunk.score, chunk.filename)
    logger.info("=" * 72)


def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[tuple[str, str]] = set()
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        key = (chunk.company, chunk.text)
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def retrieve_grouped(plan: QueryPlan, query: str) -> "OrderedDict[str, list[RetrievedChunk]]":
    """
    Retrieve chunks grouped by company and tier.
    For comparisons we fan out one filtered retrieval per policy.
    """
    groups: "OrderedDict[str, list[RetrievedChunk]]" = OrderedDict()

    if plan.is_comparison and plan.target_policies:
        num = len(plan.target_policies)
        base_k = RESULTS_PER_COMPANY_BROAD if plan.is_broad else RESULTS_PER_COMPANY
        per_policy_k = max(MIN_RESULTS_PER_COMPANY, MAX_TOTAL_RESULTS // num)
        per_policy_k = min(base_k, per_policy_k)

        for company, tier in plan.target_policies:
            chunks = _dedupe(_retrieve(query, company, tier, per_policy_k))
            group_key = f"{company} - {tier}" if tier else company
            groups[group_key] = chunks
        return groups

    # Single-scope retrieval
    top_k = BROAD_NUMBER_OF_RESULTS if plan.is_broad else NARROW_NUMBER_OF_RESULTS
    if plan.target_policies:
        company, tier = plan.target_policies[0]
        chunks = _dedupe(_retrieve(query, company, tier, top_k))
    else:
        chunks = _dedupe(_retrieve(query, None, None, top_k))
        
    for chunk in chunks:
        group_key = f"{chunk.company} - {chunk.tier}" if chunk.tier else chunk.company
        groups.setdefault(group_key, []).append(chunk)
        
    return groups


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def _build_grouped_context(groups: "OrderedDict[str, list[RetrievedChunk]]") -> str:
    """Render sealed, clearly delimited per-policy sections."""
    blocks: list[str] = []
    for group_key, chunks in groups.items():
        header = f"======== התחלת מסמכי פוליסה: {group_key} ========"
        footer = f"======== סוף מסמכי פוליסה: {group_key} ========"
        if not chunks:
            body = f"(לא אותרו קטעים רלוונטיים מתוך מסמכי {group_key} עבור שאלה זו.)"
        else:
            parts = [
                f"[פוליסה: {group_key} | קובץ: {c.filename} | קטע {i}]\n{c.text}"
                for i, c in enumerate(chunks, 1)
            ]
            body = "\n\n---\n\n".join(parts)
        blocks.append(f"{header}\n{body}\n{footer}")
    return "\n\n".join(blocks)


def _build_user_prompt(
    query: str,
    grouped_context: str,
    plan: QueryPlan,
) -> str:
    if plan.is_comparison and plan.target_policies:
        policy_names = [f"{c} {t}" if t else c for c, t in plan.target_policies]
        scope = (
            "המשתמש מבקש השוואה בין הפוליסות הבאות: "
            + ", ".join(policy_names)
            + ".\nמלא כל תא בטבלה המאוחדת אך ורק לפי הסעיף החתום של אותה פוליסה ספציפית.\n\n"
        )
    else:
        scope = ""

    return (
        f"{scope}"
        f"להלן המקורות, מחולקים לפי חברה וסוג כיסוי. כל סעיף חתום ונפרד:\n\n"
        f"{grouped_context}\n\n"
        f"שאלת המשתמש:\n{query}"
    )


def generate_answer(user_prompt: str) -> str:
    """Generate a strictly-formatted, policy-citing Hebrew answer, stripping out the <thinking> tags."""
    try:
        response = _get_runtime_client().converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 7000, "temperature": 0.0},
        )
        raw_output = response["output"]["message"]["content"][0]["text"]
        
        # כירורגיה: מוחק את בלוק המחשבה לפני השליחה ללקוח (כולל ירידות שורה)
        clean_output = re.sub(r"<thinking>.*?</thinking>", "", raw_output, flags=re.DOTALL | re.IGNORECASE).strip()
        
        return clean_output
        
    except ReadTimeoutError as exc:
        logger.error("Generation timed out (read_timeout): %s", exc)
        raise RuntimeError(
            "הניתוח ארך זמן רב מדי ונותק. נסו שוב או צמצמו את השאלה."
        ) from exc
    except ClientError as exc:
        message = exc.response.get("Error", {}).get("Message", str(exc))
        logger.error("Generation failed: %s", message)
        raise RuntimeError(f"Generation failed: {message}") from exc


def retrieve_and_generate(query: str) -> str:
    plan = plan_query(query)
    groups = retrieve_grouped(plan, query)

    has_content = any(chunks for chunks in groups.values())
    if not has_content:
        return NO_CONTEXT_MESSAGE

    _log_retrieved_chunks(groups, query)

    grouped_context = _build_grouped_context(groups)
    user_prompt = _build_user_prompt(query, grouped_context, plan)
    return generate_answer(user_prompt)