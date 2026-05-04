"""
SEC filing knowledge graph extractor.

Fetches 10-K, 10-Q, and 8-K filings for FAANG+ companies via edgartools,
extracts supply chain triples using an LLM, and saves to all_triples.json.

Run in Google Colab or locally:
    pip install anthropic edgartools tqdm pydantic python-dotenv
    python curate_knowledge.py

Environment variables (or .env file):
    ANTHROPIC_API_KEY
    EDGAR_IDENTITY   (e.g. "Your Name your@email.com")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Logging — structured, no print() statements in library code
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
EDGAR_IDENTITY: str = os.getenv("EDGAR_IDENTITY", "Research research@example.com")
MODEL: str = "claude-haiku-4-5-20251001"
MAX_TOKENS: int = 8096
CHUNK_CHARS: int = 5_000
MAX_RETRIES: int = 3

OUTPUT_FILE = Path("data/all_triples.json")

TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                       "NVDA", "TSLA", "AVGO", "QCOM"]

# Canonical company name map — fixes fragmented names from SEC filings
CANONICAL_NAMES: dict[str, str] = {
    # Microsoft
    "MICROSOFT CORP":        "Microsoft Corp",
    "Microsoft Corporation": "Microsoft Corp",
    # Amazon
    "AMAZON COM INC":        "Amazon.com Inc.",
    "Amazon Com Inc":        "Amazon.com Inc.",
    "Amazon Com Inc.":       "Amazon.com Inc.",
    "Amazon.com, Inc.":      "Amazon.com Inc.",
    "Amazon.com Inc.":       "Amazon.com Inc.",
    "Amazon.com NV Investment Holdings LLC": "Amazon.com Inc.",
    # Qualcomm
    "QUALCOMM INC/DE":       "Qualcomm Inc.",
    "QUALCOMM Incorporated": "Qualcomm Inc.",
    "QUALCOMM Inc/DE":       "Qualcomm Inc.",
    # NVIDIA
    "NVIDIA CORP":           "NVIDIA Corp",
    "NVIDIA Corporation":    "NVIDIA Corp",
    # Alphabet
    "ALPHABET INC":          "Alphabet Inc.",
    "Google LLC":            "Alphabet Inc.",
    # Meta
    "META PLATFORMS INC":    "Meta Platforms, Inc.",
    "Meta Platforms Inc.":   "Meta Platforms, Inc.",
    # Apple
    "APPLE INC":             "Apple Inc.",
    "Apple Inc":             "Apple Inc.",
    # Tesla
    "TESLA INC":             "Tesla, Inc.",
    "Tesla Inc":             "Tesla, Inc.",
    "Tesla Inc.":            "Tesla, Inc.",
    # Broadcom
    "BROADCOM INC":          "Broadcom Inc.",
    "VMware LLC":            "Broadcom Inc.",
}


def canonical(name: str) -> str:
    """Return canonical company name, or the original if not in map."""
    return CANONICAL_NAMES.get(name, name)


# ---------------------------------------------------------------------------
# Risk relation classifier
# ---------------------------------------------------------------------------
_RISK_PATTERNS: list[tuple[str, str]] = [
    (r"tariff|trade restriction|import duty|customs duty",                       "TARIFF_RISK"),
    (r"export control|entity list|sanction|EAR|ITAR|H20 chip|H100 chip",        "EXPORT_CONTROL_RISK"),
    (r"single.source|sole.source|sole supplier|single supplier|single vendor",   "SINGLE_SOURCE_RISK"),
    (r"geographic concentration|manufacturing concentration",                    "GEOGRAPHIC_CONCENTRATION_RISK"),
    (r"taiwan|hsinchu|tainan|geopolit|asia.pacific|asia pacific",               "GEOGRAPHIC_CONCENTRATION_RISK"),
    (r"cyber|security breach|data breach|unauthorized access|malicious attack",  "CYBERSECURITY_RISK"),
    (r"semiconductor shortage|chip shortage|component shortage|supply constraint","COMPONENT_SUPPLY_RISK"),
    (r"foreign exchange|fx risk|exchange rate|currency fluctuat",                "FX_RISK"),
    (r"regulation|compliance|antitrust|gdpr|privacy law",                       "REGULATORY_RISK"),
]


def classify_risk(target_text: str) -> str:
    """Map risk description to a specific relation type."""
    t = target_text.lower()
    for pattern, label in _RISK_PATTERNS:
        if re.search(pattern, t):
            return label
    return "OPERATIONAL_RISK"


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------
class NodeType(str, Enum):
    COMPANY          = "COMPANY"
    PERSON           = "PERSON"
    LOCATION         = "LOCATION"
    PRODUCT          = "PRODUCT"
    EVENT            = "EVENT"
    REGULATION       = "REGULATION"
    RISK_FACTOR      = "RISK_FACTOR"
    REVENUE_SEGMENT  = "REVENUE_SEGMENT"
    SUPPLIER         = "SUPPLIER"
    MANUFACTURER     = "MANUFACTURER"
    MARKET           = "MARKET"


class EdgeType(str, Enum):
    # Supply chain structure
    SOLE_SOURCE_FROM  = "SOLE_SOURCE_FROM"
    ASSEMBLES_WITH    = "ASSEMBLES_WITH"
    MANUFACTURES      = "MANUFACTURES"
    SUPPLIES          = "SUPPLIES"
    PROVIDES          = "PROVIDES"
    # Risk — specific types (replaces vague HAS_RISK_FACTOR)
    TARIFF_RISK                    = "TARIFF_RISK"
    EXPORT_CONTROL_RISK            = "EXPORT_CONTROL_RISK"
    SINGLE_SOURCE_RISK             = "SINGLE_SOURCE_RISK"
    GEOGRAPHIC_CONCENTRATION_RISK  = "GEOGRAPHIC_CONCENTRATION_RISK"
    COMPONENT_SUPPLY_RISK          = "COMPONENT_SUPPLY_RISK"
    CYBERSECURITY_RISK             = "CYBERSECURITY_RISK"
    FX_RISK                        = "FX_RISK"
    REGULATORY_RISK                = "REGULATORY_RISK"
    OPERATIONAL_RISK               = "OPERATIONAL_RISK"
    # Revenue / financial
    GENERATES_REVENUE_FROM = "GENERATES_REVENUE_FROM"
    HAS_REVENUE_SEGMENT    = "HAS_REVENUE_SEGMENT"
    # Location / structure
    LOCATED_IN       = "LOCATED_IN"
    HEADQUARTERED_IN = "HEADQUARTERED_IN"
    SUBSIDIARY_OF    = "SUBSIDIARY_OF"
    ACQUIRED         = "ACQUIRED"
    PARTNERS_WITH    = "PARTNERS_WITH"
    INVESTS_IN       = "INVESTS_IN"
    SUBJECT_TO       = "SUBJECT_TO"
    IMPACTED_BY      = "IMPACTED_BY"
    # Events / people
    APPOINTED_AS     = "APPOINTED_AS"
    TRANSITIONS_TO   = "TRANSITIONS_TO"


class KGTriple(BaseModel):
    source:      str
    source_type: NodeType
    relation:    EdgeType
    target:      str
    target_type: NodeType
    properties:  dict[str, Any] = Field(default_factory=dict)
    # Provenance — required for line-level traceability
    source_text:      str = ""   # excerpt from filing proving this triple
    accession_number: str = ""   # e.g. "0000320193-26-000005"
    document_url:     str = ""   # direct SEC EDGAR URL
    filing_company:   str = ""
    filing_form:      str = ""
    filing_date:      str = ""
    section:          str = ""


# ---------------------------------------------------------------------------
# SEC filing section extraction (edgartools)
# ---------------------------------------------------------------------------
ITEM_MAP: dict[str, dict[str, list[str]]] = {
    "10-K": {
        "suppliers":                ["item_1a", "item_1"],
        "geographic_concentration": ["item_1a", "item_1"],
        "risk_factors":             ["item_1a"],
        "revenue_segments":         ["item_7", "item_8"],
    },
    "10-Q": {
        "revenue_update":      ["item_2", "item_1"],
        "supply_chain_update": ["item_2", "item_1a"],
        "risk_update":         ["item_1a", "item_2"],
        "guidance":            ["item_2"],
    },
}

SECTIONS_8K: dict[str, list[str]] = {
    "material_event": [
        "material", "definitive agreement", "acquisition", "merger",
        "bankruptcy", "force majeure", "Item 1.01", "Item 2.01",
        "departure", "appointment", "officer", "director",
    ],
    "supply_disruption": [
        "supply", "supplier", "manufacturer", "factory", "facility",
        "production", "disruption", "fire", "earthquake", "flood",
        "shutdown", "closure", "suspend", "force majeure", "halt",
    ],
    "financial_impact": [
        "revenue", "earnings", "guidance", "impact",
        "million", "billion", "quarterly", "annual",
    ],
    "press_release": [
        "Item 7.01", "Item 8.01", "press release", "Regulation FD",
        "announces", "reports results", "reports record",
        "first quarter", "second quarter", "third quarter", "fourth quarter",
        "CEO", "Chief Executive", "transition",
    ],
}

SECTION_PRIORITIES: dict[str, list[str]] = {
    "10-K": ["suppliers", "geographic_concentration", "risk_factors", "revenue_segments"],
    "10-Q": ["supply_chain_update", "risk_update", "revenue_update", "guidance"],
    "8-K":  ["supply_disruption", "material_event", "financial_impact", "press_release"],
}

_ITEM_PATTERN = re.compile(
    r'(?:^|\n)\s{0,4}(Item\s+\d+[A-Z]?\.?\s{0,6}[^\n]{0,80})',
    re.IGNORECASE | re.MULTILINE,
)


def get_section_text(filing: Any, section_key: str) -> str:
    """
    Extract supply-chain-relevant section text from an edgartools filing object.
    Uses Item boundary splitting for 10-K/10-Q; keyword search for 8-K.
    Returns empty string if section not found or too short.
    """
    form_type = filing.form
    base_form = form_type.split("/")[0]

    try:
        full_text: str = filing.text() or ""
    except Exception:
        return ""

    if not full_text:
        return ""

    if base_form in ("10-K", "10-Q"):
        target_prefixes = ITEM_MAP.get(base_form, {}).get(section_key, [])
        if not target_prefixes:
            return ""

        matches = list(_ITEM_PATTERN.finditer(full_text))
        item_sections: dict[str, str] = {}

        for i, m in enumerate(matches):
            header  = m.group(1).strip()
            start   = m.end()
            end     = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            content = full_text[start:end].strip()

            if len(content) < 100:          # skip TOC stubs
                continue

            key = re.sub(r'[^a-z0-9]+', '_', header.lower()).strip('_')
            if key not in item_sections:
                item_sections[key] = content[:5_000]

        found: list[str] = []
        for prefix in target_prefixes:
            for key, text in item_sections.items():
                if key.startswith(prefix) and text not in found:
                    found.append(text)
                    break

        return "\n\n".join(found)[:5_000]

    if base_form == "8-K":
        keywords = SECTIONS_8K.get(section_key, [])
        if not keywords:
            return ""
        sentences = re.split(r'(?<=[.!?]) +', full_text)
        matched = [
            s.strip() for s in sentences
            if any(k.lower() in s.lower() for k in keywords)
        ]
        return "\n".join(matched)

    return ""


def get_filing_provenance(filing: Any) -> tuple[str, str, str]:
    """
    Return (cik, accession_number, document_url) for a filing.
    Falls back to empty strings if edgartools doesn't expose the fields.
    """
    try:
        acc = str(getattr(filing, "accession_no", "") or "")
        cik = str(getattr(filing, "cik", "") or "")
        primary_doc = str(getattr(filing, "primary_document", "") or "")
        if acc and cik:
            acc_clean = acc.replace("-", "")
            cik_int   = int(cik.lstrip("0")) if cik.lstrip("0") else 0
            url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_int}/{acc_clean}/{primary_doc}"
            )
        else:
            url = ""
        return cik, acc, url
    except Exception:
        return "", "", ""


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------
_VALID_NODE_TYPES = ", ".join(f'"{nt.value}"' for nt in NodeType)
_VALID_EDGE_TYPES = ", ".join(f'"{et.value}"' for et in EdgeType)

_SYSTEM_PROMPT = f"""You are a precision supply chain knowledge graph extractor for SEC filings.

Return ONLY valid JSON — no prose, no markdown fences:
{{
  "triples": [
    {{
      "source": "canonical entity name from filing",
      "source_type": one of [{_VALID_NODE_TYPES}],
      "relation": one of [{_VALID_EDGE_TYPES}],
      "target": "specific descriptive name — NOT generic labels",
      "target_type": one of [{_VALID_NODE_TYPES}],
      "properties": {{}},
      "source_text": "REQUIRED: exact quote ≤200 chars proving this triple"
    }}
  ]
}}

RELATION GUIDE (use the most specific):
  Supply chain: SOLE_SOURCE_FROM, ASSEMBLES_WITH, MANUFACTURES, SUPPLIES, PROVIDES
  Risk (specific): TARIFF_RISK, EXPORT_CONTROL_RISK, SINGLE_SOURCE_RISK,
    GEOGRAPHIC_CONCENTRATION_RISK, COMPONENT_SUPPLY_RISK, CYBERSECURITY_RISK,
    FX_RISK, REGULATORY_RISK, OPERATIONAL_RISK
  Revenue: GENERATES_REVENUE_FROM, HAS_REVENUE_SEGMENT
  Location: LOCATED_IN, HEADQUARTERED_IN, SUBSIDIARY_OF
  Events: APPOINTED_AS, TRANSITIONS_TO, ACQUIRED, PARTNERS_WITH

RULES:
1. CANONICAL NAMES: Use exact SEC filing name.
   "Apple Inc." not "Apple", "NVIDIA Corp" not "Nvidia".
2. DEPTH-2: Extract intermediate nodes to enable multi-hop traversal.
   (Apple Inc.)-[SOLE_SOURCE_FROM]->(TSMC) AND (TSMC)-[LOCATED_IN]->(Hsinchu, Taiwan)
3. SPECIFIC TARGETS: Never use "Various risks" or "Market conditions".
   BAD: target="Supply chain risks"
   GOOD: target="Single-source TSMC dependency for A18 chips"
4. SOURCE TEXT: Copy ≤200 chars from the filing proving each triple.
5. SEVERITY property: "HIGH" if filing says "material", "significant", or "substantial".
6. Return {{"triples": []}} if no supply chain content.
7. Extract 5-15 high-quality triples maximum per call."""


def _chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries, respecting max_chars."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in text.split("\n\n"):
        if current_len + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current, current_len = [para], len(para)
        else:
            current.append(para)
            current_len += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _repair_json(raw: str) -> str:
    """Trim truncated JSON to the last complete triple."""
    last = raw.rfind("},\n    {")
    if last == -1:
        last = raw.rfind("}")
    if last == -1 or '"triples"' not in raw:
        return '{"triples": []}'
    return raw[:last + 1] + "\n  ]\n}"


def _parse_response(raw: str) -> list[dict[str, Any]]:
    """Parse and repair LLM JSON response."""
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw).get("triples", [])
    except json.JSONDecodeError:
        try:
            return json.loads(_repair_json(raw)).get("triples", [])
        except json.JSONDecodeError:
            return []


def extract_triples(
    client: Any,
    text: str,
    company_name: str,
    form_type: str,
    filing_date: str,
    section_name: str,
    accession_number: str,
    document_url: str,
) -> list[dict[str, Any]]:
    """
    Call Claude to extract knowledge graph triples from a text chunk.
    Handles chunking, retries, JSON repair, and provenance stamping.
    """
    if not text or len(text.strip()) < 100:
        return []

    chunks   = _chunk_text(text)
    results: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_label = f"chunk {idx + 1}/{len(chunks)}" if len(chunks) > 1 else ""
        prompt = (
            f"Company: {company_name}\n"
            f"Form: {form_type} | Date: {filing_date} | "
            f"Section: {section_name} {chunk_label}\n"
            f"Accession: {accession_number}\n"
            f"Document URL: {document_url}\n\n"
            f"TEXT:\n{chunk}\n\n"
            "Extract supply chain triples. Return only JSON."
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_triples = _parse_response(response.content[0].text)
                break
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "rate_limit" in msg.lower() or "overloaded" in msg.lower():
                    wait = 60 * (attempt + 1)
                    log.warning("Rate limit — waiting %ds (attempt %d)", wait, attempt + 1)
                    time.sleep(wait)
                    raw_triples = []
                else:
                    log.error("API error on %s %s: %s", company_name, form_type, msg[:120])
                    raw_triples = []
                    break
        else:
            raw_triples = []

        for t in raw_triples:
            # Canonicalise company names
            t["source"] = canonical(t.get("source", ""))
            t["target"] = canonical(t.get("target", ""))

            # Upgrade vague risk relations
            if t.get("relation") in ("HAS_RISK_FACTOR", "HAS_SUPPLY_CHAIN_RISK"):
                t["relation"] = classify_risk(t.get("target", ""))

            # Hoist source_text from properties if the LLM put it there
            props = t.get("properties") or {}
            if isinstance(props, dict) and "source_text" in props:
                t["source_text"] = props.pop("source_text", "")

            # Stamp provenance
            t.setdefault("source_text",      "")
            t["accession_number"] = accession_number
            t["document_url"]     = document_url
            t["filing_company"]   = canonical(company_name)
            t["filing_form"]      = form_type
            t["filing_date"]      = filing_date
            t["section"]          = section_name

            # Validate with Pydantic; keep even if validation fails (don't lose data)
            try:
                validated = KGTriple(**t)
                results.append(validated.model_dump())
            except ValidationError:
                results.append(t)

        time.sleep(0.3)

    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first occurrence of each (source, relation, target) triple."""
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for t in triples:
        key = (
            str(t.get("source", "")).lower().strip(),
            str(t.get("relation", "")),
            str(t.get("target",  "")).lower().strip(),
        )
        if not all(key):
            continue
        if key not in seen:
            seen[key] = t
    return list(seen.values())


# ---------------------------------------------------------------------------
# Fetch filings
# ---------------------------------------------------------------------------
def fetch_filings(tickers: list[str]) -> list[Any]:
    """Fetch 10-K, 10-Q, 8-K filings for each ticker via edgartools."""
    from edgar import Company  # imported here so module works without edgartools installed

    os.environ["EDGAR_IDENTITY"] = EDGAR_IDENTITY
    all_filings: list[Any] = []

    for ticker in tickers:
        try:
            company = Company(ticker)
            filings = company.get_filings(form=["10-K", "10-Q", "8-K"]).latest(10)
            if filings is None:
                log.warning("%s: get_filings() returned None — skipping", ticker)
                continue
            batch = list(filings)
            if not batch:
                log.warning("%s: 0 filings found — skipping", ticker)
                continue
            all_filings.extend(batch)
            log.info("%s: %d filings", ticker, len(batch))
        except Exception as exc:
            log.error("%s: fetch failed — %s", ticker, exc)

    return all_filings


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------
def run_extraction() -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    log.info("Fetching filings for %d tickers...", len(TICKERS))
    filings = fetch_filings(TICKERS)
    log.info("Total filings: %d", len(filings))

    all_raw: list[dict[str, Any]] = []

    for filing in tqdm(filings, desc="Extracting"):
        try:
            company_name = canonical(filing.company)
            form_type    = filing.form
            filing_date  = str(filing.filing_date)
            base_form    = form_type.split("/")[0]

            _cik, accession, doc_url = get_filing_provenance(filing)

            section_keys = SECTION_PRIORITIES.get(base_form, ["risk_factors"])
            filing_triples: list[dict[str, Any]] = []

            for section_key in section_keys:
                text = get_section_text(filing, section_key)
                if not text or len(text.strip()) < 100:
                    continue

                triples = extract_triples(
                    client           = client,
                    text             = text,
                    company_name     = company_name,
                    form_type        = form_type,
                    filing_date      = filing_date,
                    section_name     = section_key,
                    accession_number = accession,
                    document_url     = doc_url,
                )
                filing_triples.extend(triples)

            all_raw.extend(filing_triples)

            if filing_triples:
                tqdm.write(f"  ✓ {company_name:<28} {form_type:<6} {filing_date} → {len(filing_triples)}")
            else:
                tqdm.write(f"  · {company_name:<28} {form_type:<6} {filing_date} → 0")

        except Exception as exc:
            tqdm.write(f"  ✗ {exc}")

    unique = deduplicate(all_raw)
    log.info("Raw: %d  Unique: %d", len(all_raw), len(unique))

    output = {
        "extracted_at":  datetime.now(timezone.utc).isoformat(),
        "model":         MODEL,
        "total_raw":     len(all_raw),
        "total_unique":  len(unique),
        "triples":       unique,
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    log.info("Saved → %s", OUTPUT_FILE)

    # Quality report
    rels = Counter(t.get("relation") for t in unique)
    print("\nRelation distribution:")
    for rel, n in rels.most_common():
        print(f"  {rel:<45} {n:>4}  {'█' * min(n // 3, 20)}")

    provenance_ok  = sum(1 for t in unique if t.get("accession_number"))
    source_text_ok = sum(1 for t in unique if t.get("source_text"))
    print(f"\nProvenance coverage:")
    print(f"  accession_number : {provenance_ok}/{len(unique)}")
    print(f"  source_text      : {source_text_ok}/{len(unique)}")


if __name__ == "__main__":
    run_extraction()