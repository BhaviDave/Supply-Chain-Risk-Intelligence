"""
==============
Loads all_triples.json into Neo4j AuraDB (or local Neo4j).

Features
--------
- Canonical company name normalisation (no duplicate nodes)
- Specific risk relation types (TARIFF_RISK etc.) upgraded from vague ones
- Full edge provenance: accession_number, document_url, source_text
- Filing provenance nodes (:Filing) for click-through to exact document
- Idempotent MERGE queries (safe to re-run)
- No APOC dependency


Environment variables (.env or export):
    NEO4J_URI
    NEO4J_USERNAME
    NEO4J_PASSWORD
    TRIPLES_FILE   (default: all_triples.json)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
# Configuration — all from environment, nothing hard-coded
# ---------------------------------------------------------------------------
NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USER     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
TRIPLES_FILE   = Path(os.getenv("TRIPLES_FILE", "all_triples.json"))


# ---------------------------------------------------------------------------
# Canonical company name map (same source-of-truth as curate_knowledge.py)
# ---------------------------------------------------------------------------
CANONICAL_NAMES: dict[str, str] = {
    "MICROSOFT CORP":        "Microsoft Corp",
    "Microsoft Corporation": "Microsoft Corp",
    "AMAZON COM INC":        "Amazon.com Inc.",
    "Amazon Com Inc":        "Amazon.com Inc.",
    "Amazon Com Inc.":       "Amazon.com Inc.",
    "Amazon.com, Inc.":      "Amazon.com Inc.",
    "Amazon.com Inc.":       "Amazon.com Inc.",
    "Amazon.com NV Investment Holdings LLC": "Amazon.com Inc.",
    "QUALCOMM INC/DE":       "Qualcomm Inc.",
    "QUALCOMM Incorporated": "Qualcomm Inc.",
    "QUALCOMM Inc/DE":       "Qualcomm Inc.",
    "NVIDIA CORP":           "NVIDIA Corp",
    "NVIDIA Corporation":    "NVIDIA Corp",
    "ALPHABET INC":          "Alphabet Inc.",
    "Google LLC":            "Alphabet Inc.",
    "META PLATFORMS INC":    "Meta Platforms, Inc.",
    "Meta Platforms Inc.":   "Meta Platforms, Inc.",
    "APPLE INC":             "Apple Inc.",
    "Apple Inc":             "Apple Inc.",
    "TESLA INC":             "Tesla, Inc.",
    "Tesla Inc":             "Tesla, Inc.",
    "Tesla Inc.":            "Tesla, Inc.",
    "BROADCOM INC":          "Broadcom Inc.",
    "VMware LLC":            "Broadcom Inc.",
}


def canonical(name: str) -> str:
    return CANONICAL_NAMES.get(name, name)


# ---------------------------------------------------------------------------
# Risk relation classifier (same as curate_knowledge.py)
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
    t = target_text.lower()
    for pattern, label in _RISK_PATTERNS:
        if re.search(pattern, t):
            return label
    return "OPERATIONAL_RISK"


# ---------------------------------------------------------------------------
# Node label mapping
# ---------------------------------------------------------------------------
_NODE_LABELS: dict[str, str] = {
    "COMPANY":          "Company",
    "SUPPLIER":         "Supplier",
    "MANUFACTURER":     "Supplier",
    "CUSTOMER":         "Company",
    "LOCATION":         "Location",
    "PRODUCT":          "Product",
    "SERVICE":          "Product",
    "TECHNOLOGY":       "Product",
    "PERSON":           "Person",
    "RISK_FACTOR":      "Risk",
    "REVENUE_SEGMENT":  "RevenueSegment",
    "MARKET":           "Market",
    "REGULATION":       "Regulation",
    "EVENT":            "Event",
    "FINANCIAL_FILING": "Filing",
    "DOCUMENT":         "Document",
    "INVESTMENT":       "Investment",
    "ACQUISITION":      "Acquisition",
    "AGREEMENT":        "Agreement",
    "CONTRACT":         "Agreement",
    "ROLE":             "Role",
}


def node_label(node_type: str) -> str:
    return _NODE_LABELS.get(node_type, "Entity")


# ---------------------------------------------------------------------------
# Schema — constraints and indexes
# ---------------------------------------------------------------------------
_SCHEMA: list[str] = [
    "CREATE CONSTRAINT company_name     IF NOT EXISTS FOR (n:Company)       REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT supplier_name    IF NOT EXISTS FOR (n:Supplier)      REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT location_name    IF NOT EXISTS FOR (n:Location)      REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT person_name      IF NOT EXISTS FOR (n:Person)        REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT product_name     IF NOT EXISTS FOR (n:Product)       REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT risk_name        IF NOT EXISTS FOR (n:Risk)          REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT regulation_name  IF NOT EXISTS FOR (n:Regulation)    REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT event_name       IF NOT EXISTS FOR (n:Event)         REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT filing_accession IF NOT EXISTS FOR (n:Filing)        REQUIRE n.accession_number IS UNIQUE",
    "CREATE CONSTRAINT role_name        IF NOT EXISTS FOR (n:Role)          REQUIRE n.name IS UNIQUE",
    """CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
       FOR (n:Company|Supplier|Location|Product|Risk|Regulation|Event)
       ON EACH [n.name, n.description]""",
]

_FILING_MERGE = """
MERGE (f:Filing {accession_number: $accession_number})
  ON CREATE SET
    f.company      = $company,
    f.form_type    = $form_type,
    f.filing_date  = $filing_date,
    f.document_url = $document_url,
    f.loaded_at    = $now
"""


# ---------------------------------------------------------------------------
# Triple → Cypher MERGE
# ---------------------------------------------------------------------------
def _build_merge(triple: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Build a parameterised Cypher MERGE for one triple.
    Upgrades vague risk relations and applies canonical names.
    """
    src_label = node_label(triple.get("source_type", "COMPANY"))
    tgt_label = node_label(triple.get("target_type", "COMPANY"))

    relation = triple.get("relation", "RELATED_TO")
    if relation in ("HAS_RISK_FACTOR", "HAS_SUPPLY_CHAIN_RISK"):
        relation = classify_risk(triple.get("target", ""))

    rel_type = re.sub(r'[^A-Z0-9_]', '_', relation.upper())

    src_name = canonical(triple.get("source", "Unknown"))
    tgt_name = canonical(triple.get("target", "Unknown"))
    props    = triple.get("properties") or {}
    now      = datetime.now(timezone.utc).isoformat()

    cypher = f"""
MERGE (src:{src_label} {{name: $src_name}})
  ON CREATE SET src.node_type = $src_type, src.created_at = $now
MERGE (tgt:{tgt_label} {{name: $tgt_name}})
  ON CREATE SET tgt.node_type = $tgt_type, tgt.created_at = $now
MERGE (src)-[r:{rel_type}]->(tgt)
  ON CREATE SET
    r.filing_company   = $filing_company,
    r.filing_form      = $filing_form,
    r.filing_date      = $filing_date,
    r.section          = $section,
    r.accession_number = $accession_number,
    r.document_url     = $document_url,
    r.source_text      = $source_text,
    r.properties_json  = $props_json,
    r.created_at       = $now
  ON MATCH SET
    r.last_seen = $filing_date
"""

    params: dict[str, Any] = {
        "src_name":        src_name,
        "src_type":        triple.get("source_type", ""),
        "tgt_name":        tgt_name,
        "tgt_type":        triple.get("target_type", ""),
        "filing_company":  canonical(triple.get("filing_company", "")),
        "filing_form":     triple.get("filing_form", ""),
        "filing_date":     str(triple.get("filing_date", "")),
        "section":         triple.get("section", ""),
        "accession_number": triple.get("accession_number", ""),
        "document_url":    triple.get("document_url", ""),
        "source_text":     (triple.get("source_text") or "")[:500],
        "props_json":      json.dumps(props),
        "now":             now,
    }

    # Flatten scalar properties onto the edge for direct Cypher querying
    for k, v in props.items():
        if isinstance(v, (str, int, float, bool)):
            safe_k = re.sub(r'[^a-z0-9_]', '_', k.lower())
            cypher += f"\n  ON MATCH SET r.{safe_k} = ${safe_k}_p"
            cypher += f"\n  ON CREATE SET r.{safe_k} = ${safe_k}_p"
            params[f"{safe_k}_p"] = v

    return cypher.strip(), params


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_triples(driver: Driver, triples: list[dict[str, Any]]) -> dict[str, int]:
    stats = {"loaded": 0, "skipped": 0, "failed": 0, "upgraded": 0}
    seen_accessions: set[str] = set()

    with driver.session() as session:
        for i, triple in enumerate(triples):
            src = canonical(triple.get("source", ""))
            tgt = canonical(triple.get("target", ""))

            # Skip degenerate triples
            if not src or not tgt or src.lower() == tgt.lower():
                stats["skipped"] += 1
                continue

            # Track relation upgrades for reporting
            if triple.get("relation") in ("HAS_RISK_FACTOR", "HAS_SUPPLY_CHAIN_RISK"):
                stats["upgraded"] += 1

            # Upsert Filing provenance node once per accession
            acc = triple.get("accession_number", "")
            if acc and acc not in seen_accessions:
                try:
                    session.run(_FILING_MERGE, {
                        "accession_number": acc,
                        "company":    canonical(triple.get("filing_company", "")),
                        "form_type":  triple.get("filing_form", ""),
                        "filing_date": str(triple.get("filing_date", "")),
                        "document_url": triple.get("document_url", ""),
                        "now": datetime.now(timezone.utc).isoformat(),
                    })
                    seen_accessions.add(acc)
                except Exception:
                    pass

            try:
                cypher, params = _build_merge(triple)
                session.run(cypher, **params)
                stats["loaded"] += 1
            except Exception as exc:
                stats["failed"] += 1
                if stats["failed"] <= 5:
                    log.error(
                        "Failed (%s)-[%s]->(%s): %s",
                        src, triple.get("relation"), tgt, str(exc)[:100],
                    )

            if (i + 1) % 100 == 0:
                log.info(
                    "Progress %d/%d  loaded=%d  failed=%d  upgraded=%d",
                    i + 1, len(triples),
                    stats["loaded"], stats["failed"], stats["upgraded"],
                )

    return stats


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------
def graph_stats(driver: Driver) -> None:
    log.info("Graph statistics:")
    with driver.session() as session:
        for label in ["Company", "Supplier", "Location", "Product",
                      "Person", "Risk", "RevenueSegment", "Regulation",
                      "Event", "Filing", "Role", "Market"]:
            try:
                c = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
                if c:
                    log.info("  %-20s %d", label, c)
            except Exception:
                pass

        log.info("  Relationships:")
        rows = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC LIMIT 20"
        )
        for row in rows:
            log.info("    %-40s %d", row["t"], row["c"])


# ---------------------------------------------------------------------------
# Provenance verification queries (run after loading to confirm traceability)
# ---------------------------------------------------------------------------
_PROVENANCE_CHECKS: list[dict[str, str]] = [
    {
        "name": "Risk edges with full provenance",
        "cypher": """
MATCH (c:Company)-[r]->(risk:Risk)
WHERE r.accession_number IS NOT NULL
  AND r.document_url IS NOT NULL
  AND r.source_text IS NOT NULL
  AND r.source_text <> ''
RETURN c.name, type(r), risk.name,
       r.filing_date, r.document_url, left(r.source_text, 80) AS excerpt
ORDER BY r.filing_date DESC LIMIT 5
""",
    },
    {
        "name": "Click-through provenance (Filing nodes)",
        "cypher": """
MATCH (f:Filing)
RETURN f.accession_number, f.company, f.form_type,
       f.filing_date, f.document_url
ORDER BY f.filing_date DESC LIMIT 5
""",
    },
    {
        "name": "Upgraded risk relations",
        "cypher": """
MATCH (c:Company)-[r:TARIFF_RISK|EXPORT_CONTROL_RISK|GEOGRAPHIC_CONCENTRATION_RISK
                     |SINGLE_SOURCE_RISK|COMPONENT_SUPPLY_RISK]->(risk)
RETURN type(r) AS relation, count(*) AS count
ORDER BY count DESC
""",
    },
]


def verify_provenance(driver: Driver) -> None:
    log.info("Verifying provenance...")
    with driver.session() as session:
        for check in _PROVENANCE_CHECKS:
            log.info("  %s", check["name"])
            try:
                rows = list(session.run(check["cypher"]))
                if rows:
                    for row in rows:
                        log.info("    %s", dict(row))
                else:
                    log.info("    (no results)")
            except Exception as exc:
                log.error("    Query error: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    stats_only = "--stats-only" in sys.argv

    if not TRIPLES_FILE.exists():
        log.error("Triples file not found: %s", TRIPLES_FILE)
        log.error("Run curate_knowledge.py first to generate it.")
        sys.exit(1)

    with TRIPLES_FILE.open() as f:
        data = json.load(f)

    triples = data["triples"]
    log.info("Loaded %d triples from %s", len(triples), TRIPLES_FILE)
    log.info("Extracted: %s  Model: %s", data.get("extracted_at"), data.get("model"))

    rels = Counter(t.get("relation") for t in triples)
    log.info("Relation distribution:")
    for rel, n in rels.most_common(10):
        log.info("  %-45s %d", rel, n)

    log.info("Connecting to Neo4j: %s", NEO4J_URI)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as exc:
        log.error("Connection failed: %s", exc)
        sys.exit(1)
    log.info("Connected")

    if stats_only:
        graph_stats(driver)
        verify_provenance(driver)
        driver.close()
        return

    # Apply schema
    log.info("Applying schema constraints and indexes...")
    with driver.session() as session:
        for q in _SCHEMA:
            try:
                session.run(q)
            except Exception as exc:
                log.warning("Schema: %s", str(exc)[:80])

    # Load
    log.info("Loading %d triples...", len(triples))
    stats = load_triples(driver, triples)
    log.info(
        "Done — loaded=%d  upgraded=%d  failed=%d  skipped=%d",
        stats["loaded"], stats["upgraded"], stats["failed"], stats["skipped"],
    )

    graph_stats(driver)
    verify_provenance(driver)
    driver.close()


if __name__ == "__main__":
    main()