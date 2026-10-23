"""
Extended test suite for CT-200 Document Intelligence System.

Tests cover:
  - Parser edge cases (the 3+ explicit unit tests required by the spec)
  - Hierarchy edge cases discovered in real CT-200 document
  - Content hash determinism
  - Version diffing / staleness logic
  - LLM output parsing & validation
"""

from __future__ import annotations

import pytest

from app.parser import (
    build_hierarchy,
    content_hash,
    get_heading,
    normalize_text,
)
from app.versioning import check_staleness, compare_versions
from app.llm import parse_and_validate


# ===========================================================================
# Parser unit tests (original)
# ===========================================================================

def test_normalize_text():
    assert normalize_text("  Hello   World  ") == "Hello World"


def test_content_hash_is_consistent():
    """Normalisation must make extra whitespace transparent."""
    assert content_hash("Hello World") == content_hash("Hello   World")


def test_detect_section_heading():
    assert get_heading("1. Device Overview") == ("1", "Device Overview")


def test_detect_multi_level_heading():
    """Deeply nested section numbers must be parsed correctly."""
    result = get_heading("2.1.1.1 Battery Life Under Typical Use")
    assert result == ("2.1.1.1", "Battery Life Under Typical Use")


def test_ignore_numbered_blood_pressure_category():
    """Lines like '1. Normal: systolic < 120' are list items, not headings."""
    assert get_heading("1. Normal: systolic < 120 and diastolic < 80") is None


def test_ignore_numbered_list_with_parenthesis():
    """e.g. '2) Press the button' is a list item."""
    assert get_heading("2) Press the button") is None


def test_ignore_heading_with_colon():
    """A heading-like line containing ':' is classified as a list item."""
    assert get_heading("3. Hypertension Stage 1: systolic 130–139") is None


def test_ignore_lowercase_start():
    """Headings must begin with an uppercase letter."""
    assert get_heading("1.1 intended use") is None


def test_build_hierarchy():
    pages = [
        {
            "page_number": 1,
            "text": (
                "1. Device Overview\n"
                "This is the overview.\n"
                "2. Device Operation\n"
                "This describes operation."
            ),
        }
    ]

    tree = build_hierarchy(pages)

    assert len(tree) == 2
    assert tree[0]["heading"] == "Device Overview"
    assert tree[1]["heading"] == "Device Operation"
    assert "This is the overview." in tree[0]["body_text"]


# ===========================================================================
# Explicit edge-case unit tests (required by spec)
# ===========================================================================

def test_duplicate_heading_produces_distinct_ids_and_correct_parents():
    """
    Edge case: two sections with the same heading text (e.g. 'Error Codes' appears
    at 4.2 and 7.1 in the CT-200 manual).  Both must produce distinct node IDs
    and be parented to the correct top-level section.
    """
    pages = [
        {
            "page_number": 1,
            "text": (
                "4. Alarms\n"
                "Some alarm info.\n"
                "4.2 Error Codes\n"
                "First error code table.\n"
                "7. Troubleshooting\n"
                "Some troubleshooting info.\n"
                "7.1 Error Codes\n"
                "Second error code reference.\n"
            ),
        }
    ]
    tree = build_hierarchy(pages)

    # Both headings are 'Error Codes' — find all nodes with that heading
    all_error_code_nodes = []

    def collect(nodes):
        for n in nodes:
            if n["heading"] == "Error Codes":
                all_error_code_nodes.append(n)
            collect(n["children"])

    collect(tree)

    assert len(all_error_code_nodes) == 2, (
        f"Expected 2 'Error Codes' nodes, got {len(all_error_code_nodes)}"
    )

    ids = {n["id"] for n in all_error_code_nodes}
    assert len(ids) == 2, "Duplicate-heading nodes must have distinct IDs"

    # Check parents
    section4 = next(n for n in tree if n["section_number"] == "4")
    section7 = next(n for n in tree if n["section_number"] == "7")

    child_sections_of_4 = {c["section_number"] for c in section4["children"]}
    child_sections_of_7 = {c["section_number"] for c in section7["children"]}

    assert "4.2" in child_sections_of_4, "4.2 should be a child of section 4"
    assert "7.1" in child_sections_of_7, "7.1 should be a child of section 7"


def test_skipped_heading_level_is_accepted():
    """
    Edge case: the CT-200 manual skips from level 2 (2.1) directly to level 4
    (2.1.1.1) without a 2.1.1 intermediate.  The parser must not crash and must
    parent 2.1.1.1 under 2.1.
    """
    pages = [
        {
            "page_number": 2,
            "text": (
                "2. Physical Specifications\n"
                "2.1 General Specifications\n"
                "Oscillometric method.\n"
                "2.1.1.1 Battery Life Under Typical Use\n"
                "300 cycles.\n"
            ),
        }
    ]
    tree = build_hierarchy(pages)

    section2 = next(n for n in tree if n["section_number"] == "2")
    section21 = next(n for n in section2["children"] if n["section_number"] == "2.1")
    assert len(section21["children"]) == 1
    assert section21["children"][0]["section_number"] == "2.1.1.1"


def test_out_of_order_heading_at_same_level():
    """
    Edge case: section 3.4 appears in the PDF before 3.3 (as in the real CT-200
    manual due to page-flow rendering order). The parser must produce two distinct
    child nodes of 3 with section_numbers '3.4' and '3.3', in text order.
    """
    pages = [
        {
            "page_number": 3,
            "text": (
                "3. Device Operation\n"
                "3.2 Cuff Inflation Sequence\n"
                "Inflate to 180 mmHg.\n"
                "3.4 Auto Shutoff\n"
                "Powers off after 60 seconds.\n"
                "3.3 Result Display\n"
                "Shows readings.\n"
            ),
        }
    ]
    tree = build_hierarchy(pages)

    section3 = next(n for n in tree if n["section_number"] == "3")
    child_numbers = [c["section_number"] for c in section3["children"]]

    assert "3.2" in child_numbers
    assert "3.4" in child_numbers
    assert "3.3" in child_numbers
    assert len(child_numbers) == 3


def test_body_text_includes_numbered_list_entries():
    """
    Numbered lines that are blood-pressure categories (contain ':') must be
    stored as body_text under the containing section, not parsed as headings.
    """
    pages = [
        {
            "page_number": 3,
            "text": (
                "3.3 Result Display and Classification\n"
                "After measurement:\n"
                "1. Normal: systolic < 120 and diastolic < 80\n"
                "2. Elevated: systolic 120-129\n"
            ),
        }
    ]
    tree = build_hierarchy(pages)

    # tree[0] should be 3.3
    node = tree[0]
    assert node["section_number"] == "3.3"
    assert "Normal: systolic" in node["body_text"]
    assert "Elevated" in node["body_text"]
    assert len(node["children"]) == 0, "Category lines must not create child nodes"


# ===========================================================================
# Versioning / diff tests
# ===========================================================================

class _FakeNode:
    """Minimal stand-in for DocumentNode in staleness tests."""
    def __init__(self, stable_path, content_hash):
        self.stable_path = stable_path
        self.content_hash = content_hash
        self.heading = f"Heading {stable_path}"
        self.body_text = f"Body for {stable_path}"


class _FakeDB:
    """Minimal in-memory DB stand-in for staleness tests."""
    def __init__(self, nodes):
        self._nodes = {n.stable_path: n for n in nodes}

    def query(self, model):
        return self

    def filter_by(self, **kwargs):
        version_id = kwargs.get("version_id")
        stable_path = kwargs.get("stable_path")
        node = self._nodes.get(stable_path)
        self._result = node
        return self

    def first(self):
        return self._result


def test_staleness_all_current():
    pinned = {"4.1": "hash_abc", "4.2": "hash_def"}
    nodes = [
        _FakeNode("4.1", "hash_abc"),
        _FakeNode("4.2", "hash_def"),
    ]
    db = _FakeDB(nodes)
    result = check_staleness(db, pinned, current_version_id=2)
    assert result["overall"] == "current"


def test_staleness_detects_stale_node():
    pinned = {"4.1": "hash_abc", "4.2": "hash_def"}
    nodes = [
        _FakeNode("4.1", "hash_NEW"),   # hash changed!
        _FakeNode("4.2", "hash_def"),
    ]
    db = _FakeDB(nodes)
    result = check_staleness(db, pinned, current_version_id=2)
    assert result["overall"] == "stale"
    stale_detail = next(d for d in result["details"] if d["stable_path"] == "4.1")
    assert stale_detail["status"] == "stale"


def test_staleness_detects_missing_node():
    pinned = {"4.1": "hash_abc", "9.9": "hash_xyz"}
    nodes = [_FakeNode("4.1", "hash_abc")]    # 9.9 not present
    db = _FakeDB(nodes)
    result = check_staleness(db, pinned, current_version_id=2)
    assert result["overall"] == "missing"


# ===========================================================================
# LLM output parsing tests
# ===========================================================================

VALID_LLM_JSON = '''{
  "test_cases": [
    {
      "id": "tc_overpressure",
      "title": "Overpressure auto-deflate",
      "steps": ["Simulate cuff pressure exceeding 299 mmHg", "Observe device behavior"],
      "expected_result": "Device displays E3 and auto-deflates within 2 seconds",
      "traceability": ["4.1", "4.2"]
    }
  ]
}'''

def test_llm_valid_json_parses_correctly():
    cases, error = parse_and_validate(VALID_LLM_JSON)
    assert error is None
    assert cases is not None
    assert len(cases) == 1
    assert cases[0]["id"] == "tc_overpressure"
    assert "4.1" in cases[0]["traceability"]


def test_llm_malformed_json_returns_error():
    cases, error = parse_and_validate("this is not json at all")
    assert cases is None
    assert error is not None


def test_llm_missing_required_field_returns_error():
    # 'expected_result' is missing
    bad_json = '{"test_cases": [{"id": "x", "title": "T", "steps": ["step"]}]}'
    cases, error = parse_and_validate(bad_json)
    assert cases is None
    assert error is not None


def test_llm_json_wrapped_in_markdown_fences():
    wrapped = "```json\n" + VALID_LLM_JSON + "\n```"
    cases, error = parse_and_validate(wrapped)
    assert error is None
    assert cases is not None
