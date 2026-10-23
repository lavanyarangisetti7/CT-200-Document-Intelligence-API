#!/usr/bin/env python3
"""
End-to-end demo script — runs the full v1→v2 versioning + staleness flow.

Usage:
    python demo.py

Requirements:
    - MongoDB running locally on port 27017  (or set MONGO_URI in .env)
    - GROQ_API_KEY set in .env
    - .venv activated  (or run via .venv/Scripts/python demo.py)
    - API server running: uvicorn app.main:app --reload

This script will:
 1. Ingest ct200_manual.pdf as v1
 2. List top-level sections (v1)
 3. Ingest ct200_manual_v2.pdf as v2
 4. Show the diff between v1 and v2
 5. Create a selection from specific nodes in v1
 6. Generate QA test cases for the selection (calls LLM)
 7. Retrieve the generation and check staleness against v2
 8. Print the full staleness report
"""

import json
import os
import sys

import requests

BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


def pp(label: str, data):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print('─'*60)
    print(json.dumps(data, indent=2, default=str))


def post(path, body=None):
    resp = requests.post(f"{BASE}{path}", json=body or {})
    resp.raise_for_status()
    return resp.json()


def get(path):
    resp = requests.get(f"{BASE}{path}")
    resp.raise_for_status()
    return resp.json()


def main():
    print("🩺  CT-200 Document Intelligence — End-to-End Demo")
    print("=" * 60)

    # ── 1. Ingest v1 ────────────────────────────────────────────────────────
    print("\n[1/7] Ingesting ct200_manual.pdf as v1 ...")
    v1 = post("/ingest", {"filename": "ct200_manual.pdf", "document_name": "CT-200 Manual"})
    pp("Ingestion result (v1)", v1)
    version_id_v1 = v1["version_id"]

    # ── 2. List sections ─────────────────────────────────────────────────────
    print("\n[2/7] Listing top-level sections (v1) ...")
    sections = get("/sections?version=1")
    pp("Top-level sections (v1)", sections)

    # ── 3. Ingest v2 ────────────────────────────────────────────────────────
    print("\n[3/7] Ingesting ct200_manual_v2.pdf as v2 ...")
    v2 = post("/ingest", {"filename": "ct200_manual_v2.pdf", "document_name": "CT-200 Manual"})
    pp("Ingestion result (v2)", v2)
    version_id_v2 = v2["version_id"]

    # ── 4. Diff v1 vs v2 ────────────────────────────────────────────────────
    print("\n[4/7] Diffing v1 vs v2 ...")
    diff = get(f"/versions/{version_id_v1}/diff/{version_id_v2}")
    pp("Version diff summary", {
        "unchanged_count": diff["unchanged_count"],
        "modified": diff["modified"],
        "added": diff["added"],
        "deleted": diff["deleted"],
    })

    # ── 5. Create a selection from a v1 node ─────────────────────────────────
    # Use the first two nodes from the v1 sections listing
    print("\n[5/7] Creating a selection from v1 nodes ...")
    v1_sections = sections["sections"]
    if not v1_sections:
        print("ERROR: No sections found — was v1 ingested?")
        sys.exit(1)

    # Pick the first section and query its children to get a richer selection
    first_section_id = v1_sections[0]["id"]
    first_section_detail = get(f"/sections/{first_section_id}")
    node_ids = [first_section_id]
    if first_section_detail.get("children"):
        node_ids.append(first_section_detail["children"][0]["id"])

    selection = post("/selections", {
        "name": "Safety-critical nodes from v1",
        "node_ids": node_ids,
        "version_id": version_id_v1,
    })
    pp("Selection created", selection)
    selection_id = selection["id"]

    # ── 6. Generate QA test cases ─────────────────────────────────────────────
    print("\n[6/7] Generating QA test cases (calls LLM) ...")
    generation = post(f"/selections/{selection_id}/generate", {"force": False})
    mongo_id = generation.get("_id")
    pp("Generated test cases", {
        "is_valid": generation.get("is_valid"),
        "parse_error": generation.get("parse_error"),
        "test_cases": generation.get("test_cases"),
    })

    # ── 7. Retrieve generation with staleness check against v2 ───────────────
    print(f"\n[7/7] Retrieving generation {mongo_id} and checking staleness vs v2 ...")
    if mongo_id:
        retrieved = get(f"/generations/{mongo_id}?check_version_id={version_id_v2}")
        pp("Generation + staleness report", {
            "is_valid": retrieved.get("is_valid"),
            "staleness": retrieved.get("staleness"),
            "test_case_titles": [tc["title"] for tc in retrieved.get("test_cases", [])],
        })
    else:
        print("WARNING: No mongo_id returned — skipping retrieval step")

    print("\n✅  Demo complete!")


if __name__ == "__main__":
    main()
