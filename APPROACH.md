# CT-200 Document Intelligence — Approach Document

## 1. OCR & Document Parsing Approach

### Tool Selected: PyMuPDF (`pymupdf`)

**Why PyMuPDF, not an OCR tool:**  
The CT-200 manual is a born-digital PDF — the text layer is embedded as actual characters, not scanned image pixels. Running an OCR engine (Tesseract, AWS Textract) on a digital PDF is redundant; it would introduce recognition errors that don't exist in the source. PyMuPDF's `page.get_text("text")` extracts the embedded font glyphs directly with 100% fidelity.

If the document were a scan, the appropriate tool would be Tesseract for simple cases or a cloud-based layout-aware model (AWS Textract, Google Document AI) for complex tables and multi-column layouts. This is noted in the "What I'd do differently" section below.

**What `pymupdf4llm` offers (and why I didn't use it as the primary extractor):**  
`pymupdf4llm` converts a PDF to Markdown, which is convenient but trades control for convenience. Its markdown hierarchy inference can silently merge or split sections when heading styles are visually inconsistent — exactly the failure mode we need to catch explicitly. I chose raw text extraction + my own heading detection so every decision is auditable.

---

## 2. Hierarchy Reconstruction Strategy

### Heading Detection

The CT-200 manual uses a consistent but imperfect numbered heading system:

```
1. Device Overview
1.1 Intended Use
2.1.1.1 Battery Life Under Typical Use
```

The `get_heading()` function uses a regex to match lines that:
- Start with a dot-separated numeric prefix (`\d+(?:\.\d+)*\.?`)
- Followed by a title starting with an uppercase letter
- Shorter than 150 characters (excludes wrapped body text)

**Exclusions added after inspection:**
- Lines matching `^\d+\s*[:)]` are list items (e.g. `1. Normal: systolic < 120`)
- Lines where the title contains `:` after the first word are classified as list items (the most common false-positive in the CT-200 document)

### Tree Construction

A stack-based approach: we maintain a "current ancestors" stack keyed by heading level (number of dots + 1). When a new heading is detected:
1. Pop all stack entries at the same or deeper level
2. Append the new node as a child of the new stack top
3. Push the new node

Body text lines (non-headings) accumulate into the current top-of-stack node.

---

## 3. Structural Inconsistencies Discovered in the PDF

| # | Issue | Location | How Identified |
|---|-------|----------|---------------|
| 1 | **Duplicate heading text**: "Error Codes" appears at §4.2 and §7.1 | Pages 4, 6 | Manual inspection of parsed_tree.json |
| 2 | **Skipped heading level**: §2.1.1.1 follows §2.1 directly (no §2.1.1) | Page 2 | Level-depth assertion in test |
| 3 | **Out-of-order sections**: §3.4 appears before §3.3 in the PDF text stream | Page 3 | Comparing section numbers in order of appearance vs sorted order |
| 4 | **Numbered list entries in body**: §3.3 uses numbered lines for classification categories | Page 3 | Visual comparison of parsed_tree.json vs PDF |
| 5 | **Unicode hyphens**: Em-dash (‐) and en-dash (–) in cuff size ranges | Throughout | Python `unicodedata.normalize("NFKC", ...)` canonicalisation |

---

## 4. Initial Implementation Failures

1. **Numbered list items parsed as headings**: The initial regex `^\d+\.\s+\w` matched `1. Normal: systolic < 120`. Fixed by adding a `:` exclusion check.
2. **Out-of-order sections became orphans**: When §3.4 was encountered before §3.3 at the same level, the stack logic popped §3.2 and then correctly placed §3.4 under §3. But §3.3 was then also placed under §3 — which is correct behavior — but initially I had a bug where §3.4 was re-parented under §3.3 because of an off-by-one in the `>=` comparison. Fixed by using `>= level` (pop same-or-deeper levels).
3. **Section 2.1.1.1 orphaned under root**: Skipping levels (2.1 → 2.1.1.1) left no stack entry at level 3. Fixed by the "pop same-or-deeper" rule — the stack top becomes §2.1 at level 2, which correctly parents §2.1.1.1 at level 4.

---

## 5. How Failures Were Identified

- **Manual inspection**: Compared `parsed_tree.json` against the PDF side-by-side
- **Explicit unit tests**: `test_duplicate_heading_produces_distinct_ids_and_correct_parents`, `test_skipped_heading_level_is_accepted`, `test_out_of_order_heading_at_same_level`, `test_body_text_includes_numbered_list_entries`
- **Assertion-driven debugging**: Added `assert` statements during development that checked parent counts at each level

---

## 6. Data Model

### SQLite (SQLAlchemy)

```
documents (id, name)
  └── document_versions (id, document_id, version_number, source_filename, file_hash)
        └── document_nodes (id, version_id, node_ref_id, stable_path, heading,
                            level, body_text, content_hash, page_number,
                            parent_stable_path)
                └── selection_items.node_id → document_nodes.id

selections (id, name, version_id, generation_mongo_id)
  └── selection_items (id, selection_id, node_id, pinned_content_hash)
```

**Key design choice — `stable_path`**: The section-number string (e.g. `4.1`) is stored separately from the parser-generated `node_ref_id` (e.g. `section-4.1-4-2`). `stable_path` is the human-stable identifier used for cross-version matching and staleness keying. `node_ref_id` is the parser's internal identifier.

### MongoDB

```javascript
{
  _id: ObjectId,
  selection_id: int,             // FK to SQLite selections.id
  selection_name: string,
  version_id: int,
  version_number: int,
  pinned_hashes: {               // staleness anchor
    "4.1": "<sha256>",
    "4.2": "<sha256>"
  },
  node_summaries: [...],
  test_cases: [                  // validated LLM output
    {
      id: "tc_overpressure",
      title: "...",
      steps: [...],
      expected_result: "...",
      traceability: ["4.1", "4.2"]
    }
  ],
  raw_llm_response: "...",
  prompt_used: "...",
  is_valid: true,
  parse_error: null,
  generated_at: ISODate
}
```

---

## 7. Version Matching Strategy

**Approach**: Path-based matching using `stable_path` (section number).

**Justification**: The CT-200 manual uses a structured numbering system that is stable across minor revisions. Matching by section number is deterministic, fast (O(n)), and auditable. The diff result shows exactly which sections changed.

**Where it breaks**:
1. **Renumbering**: If a new section is inserted (e.g. old §3 becomes §4), every renumbered section appears as deleted/added rather than moved. This is a real risk in major revisions.
2. **Duplicate stable paths** from PDF formatting errors: only the last parser-seen node survives.
3. **Fuzzy title match** would be better for case 1 but worse for case 3 — not a clear win with limited time.

---

## 8. LLM Prompt Design

### System prompt strategy
- Roles and context: "Senior QA engineer for IEC 62304-compliant medical device software"
- Strict JSON-only output instruction (no prose, no markdown fences)
- Field specification with types and constraints
- Requirement for concrete values from the source text (e.g. "299 mmHg", "2 seconds")
- Prohibition on invented requirements

### Structured-output validation
- Response requested with `response_format={"type": "json_object"}` (Groq supports this)
- Pydantic model `TestCaseBatch` validates the response before storage
- Three-layer JSON extraction: direct parse → regex fence extraction → brute-force first `{...}`
- On validation failure: store `is_valid=False`, `parse_error=<reason>`, empty `test_cases` — **no retry**

### Why no retry?
Automatic retry on a bad Pydantic model risks an infinite retry loop on a consistently malformed prompt response. The engineer should fix the prompt and regenerate. The system exposes `parse_error` to make the failure transparent.

### Duplicate generation policy
- Default: idempotent — return the existing generation for a selection
- `force=True`: create a new generation unconditionally
- Rationale: the same document text should produce the same (or equivalent) test cases; calling the LLM twice wastes API quota and risks returning inconsistent test case IDs in the same selection's history

---

## 9. Staleness Detection Design

At generation time, `pinned_hashes: {stable_path: sha256}` is stored in MongoDB alongside the test cases. This is the staleness anchor.

At retrieval time (`GET /generations/{id}`), the API:
1. Reads `pinned_hashes` from MongoDB
2. Queries the specified (or latest) version's nodes for each `stable_path`
3. Compares hashes; classifies each node as `current`, `stale`, or `missing`
4. Returns overall status: `current` / `stale` / `missing` / `mixed`

**Honest limit**: A one-word wording change (e.g. "2 seconds" → "3 seconds" for an overpressure threshold) gets the same `stale` flag as a completely rewritten section. The system treats all content changes identically — it cannot determine whether a change is safety-critical without LLM-assisted semantic diff, which is out of scope.

---

## 10. Decision Log (Required)

### Q1: What's the one part of this system most likely to silently give wrong results without erroring?

**The version-matching diff for renumbered sections.** If a document revision inserts a new top-level section (pushing §3 → §4), every section below the insertion point will appear as "deleted" in v1 and "added" in v2 — with no error, just incorrect classification. The diff JSON looks valid and parseable; only a human reviewing it against the PDF would spot the problem. I would catch this with a "fuzzy title similarity" audit pass that counts how many "deleted" nodes have a close title match in the "added" set, and flags those pairs for human review.

### Q2: Where did you choose simplicity over correctness because of time, and what would break first?

**The search endpoint uses Python-level substring matching** (`q_lower in body_text.lower()`) rather than a SQL `LIKE` query or a proper full-text index. This loads *all* nodes into memory before filtering — fine for a 30-node document, catastrophic for a 3,000-node corpus. In production, I'd add a `LIKE` filter in the SQLAlchemy query or use SQLite's FTS5 virtual table. The current implementation would also fail silently on very large documents if it OOM-killed the process.

### Q3: Name one input to your parser, versioning matcher, or LLM call that you did not handle.

**A PDF with scanned images instead of a text layer.** `page.get_text("text")` returns an empty string for a scanned page. The parser would produce a tree with all zero-body nodes and no error — the caller would see an apparently successful parse with no content. I detect this partially (`node_count` in the ingest response would be 0), but the API returns 200 OK either way. The correct fix is to detect zero-content pages and either raise a clear error or fall back to OCR (Tesseract/Document AI), documented in the README.

---

## 11. What I'd Do Differently With More Time

1. **Fallback OCR pipeline**: Detect empty-text pages and route them through Tesseract or Google Document AI
2. **Fuzzy title matching for version diff**: Add Levenshtein/cosine similarity to handle renumbered sections
3. **SQLite FTS5 for search**: Full-text index with ranking instead of substring scan
4. **LLM retry with exponential backoff**: One retry on transient errors (rate limit, timeout), but not on validation errors
5. **Semantic staleness**: Feed the changed diff to the LLM and ask "is this change safety-critical?" as a second-pass enrichment
6. **Structured PDF table extraction**: Use `pymupdf`'s table detection API to parse the Error Codes and Specifications tables into typed structured data rather than flat text
