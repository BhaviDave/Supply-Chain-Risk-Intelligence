
"""
==============
Multi-hop GraphRAG Q&A engine for supply chain intelligence.


---------------------------------------
1. LLM writes its own Cypher 
2. Iterative graph expansion — up to 3 hops, each informed by previous results.
3. Every answer cites accession_number + document_url (click-through provenance).
4. Specific risk relation types (TARIFF_RISK etc.)


Environment variables (.env or export):
    NEO4J_URI
    NEO4J_USERNAME
    NEO4J_PASSWORD
    ANTHROPIC_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import anthropic
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USER     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
ANTHROPIC_KEY  = os.environ["ANTHROPIC_API_KEY"]

MODEL     = "claude-haiku-4-5-20251001"
MAX_HOPS  = 3
MAX_ROWS  = 30    # max rows returned per Cypher query


# ---------------------------------------------------------------------------
# Graph schema — shown to the LLM so it writes correct Cypher
# ---------------------------------------------------------------------------
GRAPH_SCHEMA = """
NODE LABELS:
  Company, Supplier, Location, Product, Person, Risk,
  RevenueSegment, Regulation, Event, Filing, Market

RELATIONSHIP TYPES (use exact spelling):

  Supply chain:
    SOLE_SOURCE_FROM      — Company depends on one Supplier for a component
    ASSEMBLES_WITH        — Company uses contract Manufacturer for assembly
    MANUFACTURES          — Supplier/Company makes a Product
    SUPPLIES              — Supplier provides components to Company
    PROVIDES              — general provision relationship

  Risk (specific — do NOT use HAS_RISK_FACTOR):
    TARIFF_RISK                    — tariffs, import duties, trade measures
    EXPORT_CONTROL_RISK            — export controls, entity list, sanctions
    GEOGRAPHIC_CONCENTRATION_RISK  — manufacturing concentrated in one region
    SINGLE_SOURCE_RISK             — single-supplier dependency
    COMPONENT_SUPPLY_RISK          — semiconductor / component shortage
    CYBERSECURITY_RISK             — cyber threats, data breaches
    FX_RISK                        — foreign exchange / currency
    REGULATORY_RISK                — regulatory compliance, antitrust
    OPERATIONAL_RISK               — other operational risks

  Revenue:
    GENERATES_REVENUE_FROM  — Company earns from RevenueSegment / Product
    HAS_REVENUE_SEGMENT     — Company has a business segment

  Location / structure:
    LOCATED_IN       — facility or operation in a Location
    HEADQUARTERED_IN — corporate headquarters
    SUBSIDIARY_OF    — Company is subsidiary of parent
    ACQUIRED         — Company acquired another
    PARTNERS_WITH    — partnership relationship

  Events / people:
    APPOINTED_AS     — Person appointed to role
    TRANSITIONS_TO   — Person transitions from one role to another

EDGE PROPERTIES (available on most relationships):
  filing_date, filing_form, filing_company,
  accession_number, document_url, source_text,
  section, severity, countries_affected

COMPANY NAMES (canonical — use exact spelling):
  "Apple Inc.", "Microsoft Corp", "Alphabet Inc.", "Amazon.com Inc.",
  "Meta Platforms, Inc.", "NVIDIA Corp", "Tesla, Inc.",
  "Broadcom Inc.", "Qualcomm Inc."
"""

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_CYPHER_GEN_SYSTEM = f"""You are a Neo4j Cypher expert for a supply chain knowledge graph.
Write a Cypher query that retrieves facts needed to answer the question.

SCHEMA:
{GRAPH_SCHEMA}

CYPHER RULES:
- Use toLower() for case-insensitive text comparison
- Use CONTAINS for partial name matching
- Always use named RETURN columns
- LIMIT to {MAX_ROWS} rows
- Use OPTIONAL MATCH for relationships that may not exist
- Include r.document_url and r.source_text in RETURN when they add value
- For multi-hop: traverse 2-3 relationship hops to reach the answer
- Prefer specific risk types (TARIFF_RISK etc.) over generic ones

Respond with ONLY this JSON (no prose, no fences):
{{
  "cypher": "MATCH ... RETURN ...",
  "intent": "one sentence describing what this query finds",
  "needs_followup": true/false,
  "followup_hint": "what gap remains after this query"
}}"""

_EXPAND_SYSTEM = f"""You are iteratively exploring a supply chain knowledge graph.
You have seen initial query results and must decide if more traversal is needed.

SCHEMA:
{GRAPH_SCHEMA}

Typical expansion patterns:
- Found companies → find their revenue exposure
- Found suppliers → find which companies depend on them
- Found locations → find which suppliers are there
- Found risks → find their financial quantification

Respond with ONLY this JSON:
{{
  "needs_more": true/false,
  "reasoning": "what gap exists in current results",
  "cypher": "MATCH ... RETURN ..." or null,
  "intent": "what this next query finds"
}}"""

_ANSWER_SYSTEM = """You are a supply chain risk analyst synthesizing findings from a knowledge graph.

Structure your response:
1. Direct answer to the question
2. Specific companies and risk factors from the graph data (cite names exactly)
3. Provenance: cite filing form and date for each key claim
4. If document_url is available, note "Source: [SEC filing URL]"
5. Final line: **Key Takeaway:** one sentence

Use **bold** for company names. Use bullet points for lists.
If data is insufficient, say precisely what's missing."""

_NEWS_SYSTEM = """You are a supply chain risk analyst responding to a live news alert.
Use the graph data to assess impact.

Structure:
1. **Immediate Impact** — which companies are directly affected and why
2. **Documented Risk Factors** — what SEC filings already warned about
3. **Revenue at Risk** — quantify from revenue stream data if available
4. **What to Watch** — next indicators to monitor

Cite filing dates and document URLs where available.
Final line: **Key Takeaway:** one sentence."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Hop:
    number:  int
    intent:  str
    cypher:  str
    results: list[dict[str, Any]]

    @property
    def record_count(self) -> int:
        return len(self.results)


@dataclass
class QueryResult:
    question:   str
    is_news:    bool
    answer:     str
    hops:       list[Hop]
    provenance: list[str]   # unique document URLs cited
    timestamp:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_facts(self) -> int:
        return sum(h.record_count for h in self.hops)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_facts"] = self.total_facts
        return d


# ---------------------------------------------------------------------------
# Neo4j helper
# ---------------------------------------------------------------------------
def run_cypher(driver: Driver, cypher: str) -> list[dict[str, Any]]:
    """Execute Cypher and return results as plain dicts."""
    try:
        with driver.session() as session:
            return [dict(r) for r in session.run(cypher)]
    except Exception as exc:
        log.error("Cypher error: %s", str(exc)[:200])
        return [{"error": str(exc)[:200]}]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _call(client: anthropic.Anthropic, system: str, user: str,
          max_tokens: int = 512) -> str:
    """Single LLM call with rate-limit retry."""
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text.strip()
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate_limit" in err.lower() or "overloaded" in err.lower():
                wait = 30 * (attempt + 1)
                log.warning("Rate limit — waiting %ds", wait)
                time.sleep(wait)
            else:
                raise
    return ""


def _parse_json(raw: str) -> dict[str, Any]:
    """Strip fences and parse JSON from LLM response."""
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


def generate_initial_cypher(client: anthropic.Anthropic,
                            question: str) -> dict[str, Any]:
    raw = _call(client, _CYPHER_GEN_SYSTEM, f"Question: {question}")
    return _parse_json(raw)


def decide_expansion(client: anthropic.Anthropic,
                     question: str,
                     all_results: list[dict[str, Any]],
                     hop_num: int) -> dict[str, Any]:
    summary = json.dumps(all_results[-MAX_ROWS:], indent=2, default=str)[:3_000]
    user = (
        f"Original question: {question}\n\n"
        f"Results after hop {hop_num}:\n{summary}\n\n"
        "Do you need more graph traversal to fully answer?"
    )
    raw = _call(client, _EXPAND_SYSTEM, user)
    return _parse_json(raw)


def synthesize(client: anthropic.Anthropic,
               question: str,
               hops: list[Hop],
               is_news: bool) -> str:
    hop_text = ""
    for hop in hops:
        hop_text += f"\n--- Hop {hop.number}: {hop.intent} ({hop.record_count} records) ---\n"
        hop_text += json.dumps(hop.results[:15], indent=2, default=str)[:2_000]
        hop_text += "\n"

    user = (
        f"{'NEWS ALERT' if is_news else 'QUESTION'}: {question}\n\n"
        f"GRAPH DATA ({len(hops)} traversal hops, {sum(h.record_count for h in hops)} total records):\n"
        f"{hop_text}\n\n"
        "Synthesize a complete, grounded answer."
    )
    system = _NEWS_SYSTEM if is_news else _ANSWER_SYSTEM
    return _call(client, system, user, max_tokens=1_500)


# ---------------------------------------------------------------------------
# Main multi-hop pipeline
# ---------------------------------------------------------------------------
def ask(question: str, is_news: bool = False) -> QueryResult:
    """
    Multi-hop GraphRAG pipeline.

    1. LLM generates initial Cypher based on schema + question.
    2. Run query, collect results.
    3. LLM decides if more hops needed; if yes, writes follow-up Cypher.
    4. Repeat up to MAX_HOPS times.
    5. LLM synthesizes answer from all hop results with provenance.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    hops: list[Hop] = []
    all_results: list[dict[str, Any]] = []
    needs_followup = True

    # Hop 1: initial query
    log.info("Hop 1: generating Cypher for: %s", question[:80])
    plan      = generate_initial_cypher(client, question)
    cypher    = plan.get("cypher", "")
    intent    = plan.get("intent", "")
    needs_followup = plan.get("needs_followup", False)

    if cypher:
        results = run_cypher(driver, cypher)
        hop = Hop(number=1, intent=intent, cypher=cypher, results=results)
        hops.append(hop)
        all_results.extend(results)
        log.info("  Hop 1: %d records", hop.record_count)

    # Hops 2–N: iterative expansion
    for hop_num in range(2, MAX_HOPS + 1):
        if not needs_followup or not all_results:
            break

        expansion     = decide_expansion(client, question, all_results, hop_num - 1)
        needs_followup = expansion.get("needs_more", False)
        next_cypher   = expansion.get("cypher")
        next_intent   = expansion.get("intent", "")
        reasoning     = expansion.get("reasoning", "")

        log.info("Hop %d: %s", hop_num, reasoning[:80])

        if not needs_followup or not next_cypher:
            break

        results = run_cypher(driver, next_cypher)
        hop = Hop(number=hop_num, intent=next_intent, cypher=next_cypher, results=results)
        hops.append(hop)
        all_results.extend(results)
        log.info("  Hop %d: %d records", hop_num, hop.record_count)

    driver.close()

    # Synthesis
    log.info("Synthesizing answer from %d hops, %d total records",
             len(hops), sum(h.record_count for h in hops))
    answer = synthesize(client, question, hops, is_news=is_news)

    # Collect unique provenance URLs from all results
    provenance = list({
        r.get("document_url") or r.get("source_url") or ""
        for h in hops for r in h.results
        if (r.get("document_url") or r.get("source_url"))
    } - {""})

    return QueryResult(
        question   = question,
        is_news    = is_news,
        answer     = answer,
        hops       = hops,
        provenance = provenance,
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Intelligence",
        page_icon="🔍",
        layout="wide",
    )
    st.title("🔍 Supply Chain Intelligence — Multi-Hop GraphRAG")
    st.markdown(
        "*LLM writes its own Cypher, traverses the graph iteratively (up to 3 hops), "
        "then synthesizes a grounded answer with SEC filing provenance.*"
    )

    mode    = st.radio("Mode", ["💬 Question", "📰 News Alert"], horizontal=True)
    is_news = mode == "📰 News Alert"

    _EXAMPLES: dict[str, list[str]] = {
        "💬 Question": [
            "If TSMC is disrupted, which companies lose the most revenue?",
            "Which companies have both Taiwan exposure and semiconductor revenue?",
            "Trace: TSMC → Apple → which revenue segments are at risk?",
            "Compare export control risk between NVIDIA and Qualcomm",
            "Which FAANG company has the most diversified supply chain?",
        ],
        "📰 News Alert": [
            "Magnitude 7.4 earthquake strikes Hsinchu — TSMC assessing damage",
            "US expands H20 chip export controls to China, Vietnam, Malaysia",
            "Foxconn Zhengzhou factory suspends production — labor dispute",
        ],
    }

    st.markdown("**Examples:**")
    cols = st.columns(2)
    for i, ex in enumerate(_EXAMPLES[mode]):
        if cols[i % 2].button(ex[:60] + "...", key=f"ex_{i}"):
            st.session_state["qa_input"] = ex

    user_input = st.text_area(
        "Question or news headline",
        value=st.session_state.get("qa_input", ""),
        height=80,
    )

    if st.button("🔍 Analyze", type="primary", disabled=not user_input.strip()):
        with st.spinner(f"Running up to {MAX_HOPS}-hop graph traversal..."):
            result = ask(user_input.strip(), is_news=is_news)

        st.markdown("---")
        (st.error if is_news else st.success)("Analysis complete")
        st.markdown(result.answer)

        st.markdown("---")
        st.subheader(f"🔬 GraphRAG Trace — {len(result.hops)} hops · {result.total_facts} facts")

        for hop in result.hops:
            with st.expander(f"Hop {hop.number}: {hop.intent} → {hop.record_count} records"):
                st.code(hop.cypher, language="cypher")
                if hop.results:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(hop.results[:10]), use_container_width=True)

        if result.provenance:
            st.subheader("📄 Source Documents")
            for url in result.provenance[:5]:
                label = url.split("/")[-1] or url
                st.markdown(f"- [{label}]({url})")

        with st.expander("📋 Raw result (JSON)"):
            st.json(result.to_dict())

        st.caption(f"Completed {result.timestamp}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_cli() -> None:
    is_news = "--news" in sys.argv
    args    = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        question = " ".join(args)
    elif not sys.stdin.isatty():
        question = sys.stdin.read().strip()
    else:
        # Interactive loop
        print(f"Supply Chain GraphRAG  (model={MODEL}, max_hops={MAX_HOPS})")
        print("Type 'news: <headline>' for news mode. Ctrl-C to exit.\n")
        while True:
            try:
                raw = input("Q: ").strip()
                if not raw:
                    continue
                _news = raw.lower().startswith("news:")
                q     = raw[5:].strip() if _news else raw
                res   = ask(q, is_news=_news)
                _print_result(res)
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
        return

    res = ask(question, is_news=is_news)
    _print_result(res)


def _print_result(res: QueryResult) -> None:
    print(f"\n{'='*60}")
    print(res.answer)
    print(f"\n[{len(res.hops)} hops · {res.total_facts} facts]")
    for hop in res.hops:
        print(f"  Hop {hop.number}: {hop.intent} → {hop.record_count} records")
    if res.provenance:
        print("\nSources:")
        for url in res.provenance[:3]:
            print(f"  {url}")
    print(f"\n{res.timestamp}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# REPLACE WITH THIS:
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        run_streamlit()
    else:
        if __name__ == "__main__":
            run_cli()
except ImportError:
    if __name__ == "__main__":
        run_cli()