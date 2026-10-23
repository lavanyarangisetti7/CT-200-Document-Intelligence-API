# CT-200 Document Intelligence API

A FastAPI backend that parses the **CardioTrack CT-200** Home Blood Pressure Monitor manual (PDF), builds a versioned hierarchical document tree, and generates QA test cases from selected sections using an LLM — with staleness detection when the document changes.

---

## Quick Start

### Prerequisites

| Tool | Notes |
|------|-------|
| Python 3.11+ | Must match the `.venv` |
| MongoDB | Local (`localhost:27017`) or Atlas free tier |
| Groq API key | Free tier at [console.groq.com](https://console.groq.com) |

### 1. Clone & activate virtual environment

```bash
# Windows (PowerShell)
cd "Tri9T AI"
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install pymongo groq python-dotenv
```
> All other dependencies (FastAPI, SQLAlchemy, pymupdf, pydantic, pytest, uvicorn) are already in `.venv`.

### 3. Configure environment

```bash
copy .env.example .env
# Edit .env and set GROQ_API_KEY and MONGO_URI
```

### 4. Start MongoDB

```bash
# If installed locally:
mongod --dbpath C:\data\db

# Or use Docker:
docker run -d -p 27017:27017 mongo
```

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

API docs available at **http://127.0.0.1:8000/docs**

---

## Triggering the v1 → v2 Re-ingestion Flow

This is the key flow demonstrating versioning + staleness detection.

```bash
# 1. Ingest v1
curl -X POST http://127.0.0.1:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"filename": "ct200_manual.pdf"}'

# 2. List sections (v1)
curl http://127.0.0.1:8000/sections?version=1

# 3. Ingest v2. This creates version_number=2 WITHOUT destroying v1.
curl -X POST http://127.0.0.1:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"filename": "ct200_manual_v2.pdf"}'

# 4. Diff v1 (id=1) vs v2 (id=2)
curl http://127.0.0.1:8000/versions/1/diff/2

# 5. Create a version-pinned selection (use real node IDs from step 2)
curl -X POST http://127.0.0.1:8000/selections \
     -H "Content-Type: application/json" \
     -d '{"name": "Safety nodes", "node_ids": [5, 6], "version_id": 1}'

# 6. Generate QA test cases from the selection
curl -X POST http://127.0.0.1:8000/selections/1/generate \
     -H "Content-Type: application/json" \
     -d '{"force": false}'

# 7. Retrieve generation + staleness check against v2
curl "http://127.0.0.1:8000/generations/<mongo_id>?check_version_id=2"
```

Or run the automated demo (requires API server to be running):

```bash
python demo.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite includes:
- **3+ explicit edge-case unit tests** for the parser (duplicate headings with distinct IDs, skipped heading levels, out-of-order headings, numbered list body preservation)
- Staleness detection logic with fake DB stubs
- LLM output parsing (valid JSON, malformed JSON, missing fields, markdown-wrapped JSON)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required** for LLM generation |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `ct200_intelligence` | MongoDB database name |

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest` | Parse & ingest a PDF as a new version |
| GET | `/documents` | List all documents and versions |
| GET | `/sections?version=N` | Top-level sections (default: latest) |
| GET | `/sections/{id}` | Node + children + hash |
| GET | `/search?q=<term>` | Full-text search |
| GET | `/versions/{v1}/diff/{v2}` | Version diff |
| GET | `/nodes/{id}/changes` | Node change status |
| POST | `/selections` | Create a version-pinned selection |
| GET | `/selections/{id}` | Get selection |
| POST | `/selections/{id}/generate` | Generate QA test cases |
| GET | `/selections/{id}/generations` | List generations for selection |
| GET | `/generations/{mongo_id}` | Retrieve generation + staleness |
| GET | `/nodes/{id}/generations` | Generations referencing a node |

---

## Architecture

```
PDF (v1 / v2)
     │
     ▼
app/parser.py       ← pymupdf extraction + heading detection + hierarchy build
     │
     ▼
app/ingestion.py    ← version management, node persistence
     │
     ▼
SQLite (ct200.db)   ← documents, versions, nodes, selections (SQLAlchemy)
     │
     ├── app/versioning.py  ← path-based diff, staleness detection
     │
     └── app/llm.py         ← Groq call, structured-output validation
           │
           ▼
      MongoDB              ← generation documents (test cases + pinned hashes)
```

### Version Matching Strategy

**Path-based matching** using the section's dot-notation number (e.g. `4.1`, `3.2`).

- **Unchanged**: same `stable_path`, same `content_hash`
- **Modified**: same `stable_path`, different `content_hash`
- **Added**: `stable_path` only in new version
- **Deleted**: `stable_path` only in old version

**Known failure modes** (honest):
1. Section renumbering (e.g. new section 2 inserted, pushing old section 3 → 4) causes false add/delete pairs
2. Duplicate section numbers in the PDF only keep the last-seen node
3. Heading-only changes with same number are reported as modified (correct behavior)

### Staleness Detection

Each generation stores `pinned_hashes: {stable_path: sha256}` at creation time. At retrieval, the API compares these against the current version's hashes. **Limitations**: a one-word wording change gets the same "stale" flag as a complete rewrite — the system does not distinguish safety-critical numeric changes from editorial tweaks.

### Duplicate Generation Policy

`POST /selections/{id}/generate` is idempotent by default: if a generation already exists for the selection, it is returned unchanged. Pass `{"force": true}` to create a new generation.

---

## Data Files

| File | Description |
|------|-------------|
| `data/ct200_manual.pdf` | Version 1 of the CT-200 manual |
| `data/ct200_manual_v2.pdf` | Version 2 (with deliberate changes for staleness demo) |
| `ct200.db` | SQLite database (created on first run) |
