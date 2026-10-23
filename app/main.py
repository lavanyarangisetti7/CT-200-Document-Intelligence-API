"""
CT-200 Document Intelligence API — main FastAPI application.

Endpoints
---------
Ingestion
  POST /ingest                            Ingest a PDF (v1 or v2)

Browse
  GET  /documents                         List all documents and their versions
  GET  /sections?version=N               Top-level sections (default: latest)
  GET  /sections/{node_id}               Single node with children + hash
  GET  /search?q=<term>&version=N        Full-text / heading search
  GET  /versions/{vid}/diff/{vid2}       Diff between two versions
  GET  /nodes/{node_id}/changes          Node change status across versions

Selection
  POST /selections                       Create a named, version-pinned selection
  GET  /selections/{selection_id}        Get a selection

Generation
  POST /selections/{selection_id}/generate     Generate test cases
  GET  /selections/{selection_id}/generations  List all generations for selection

Retrieval (with staleness)
  GET  /generations/{mongo_id}            Fetch generation + staleness status
  GET  /nodes/{node_id}/generations       All generations referencing a node
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import llm as llm_module
from app import versioning
from app.database import (
    Document,
    DocumentNode,
    DocumentVersion,
    Selection,
    SelectionItem,
    get_db,
    init_db,
)
from app.ingestion import ingest_pdf
from app import mongo_store

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="CT-200 Document Intelligence API",
    description=(
        "Parse, version, browse, and generate QA test cases from "
        "the CardioTrack CT-200 Home Blood Pressure Monitor manual."
    ),
    version="2.0.0",
)


@app.on_event("startup")
def startup_event():
    init_db()


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    filename: str = Field(
        default="ct200_manual.pdf",
        description="Filename inside the data/ folder",
    )
    document_name: str = Field(default="CT-200 Manual")


class SelectionCreateRequest(BaseModel):
    name: str
    node_ids: list[int] = Field(..., description="SQLite DocumentNode IDs to include")
    version_id: Optional[int] = Field(
        default=None,
        description="Version to pin to. Defaults to the latest version of the document.",
    )


class GenerateRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="If True, create a new generation even if one already exists for this selection.",
    )


# ---------------------------------------------------------------------------
# Helper: node_to_dict
# ---------------------------------------------------------------------------

def node_to_dict(node: DocumentNode, include_children: bool = False, db: Session = None) -> dict:
    result = {
        "id": node.id,
        "node_ref_id": node.node_ref_id,
        "stable_path": node.stable_path,
        "heading": node.heading,
        "level": node.level,
        "body_text": node.body_text,
        "content_hash": node.content_hash,
        "page_number": node.page_number,
        "parent_stable_path": node.parent_stable_path,
        "version_id": node.version_id,
    }
    if include_children and db is not None:
        children = (
            db.query(DocumentNode)
            .filter_by(version_id=node.version_id, parent_stable_path=node.stable_path)
            .order_by(DocumentNode.stable_path)
            .all()
        )
        result["children"] = [node_to_dict(c, include_children=True, db=db) for c in children]
    return result


def _latest_version(db: Session, document_name: str = "CT-200 Manual") -> DocumentVersion | None:
    doc = db.query(Document).filter_by(name=document_name).first()
    if doc is None:
        return None
    return (
        db.query(DocumentVersion)
        .filter_by(document_id=doc.id)
        .order_by(DocumentVersion.version_number.desc())
        .first()
    )


# ---------------------------------------------------------------------------
# Root / health
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "CT-200 Document Intelligence API",
        "docs": "/docs",
        "version": "2.0.0",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@app.post("/ingest", tags=["Ingestion"])
def ingest(req: IngestRequest, db: Session = Depends(get_db)):
    """
    Parse a PDF from data/ and store it as a new document version.
    Re-running with the same file will create v2, v3, etc.
    """
    pdf_path = DATA_DIR / req.filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")

    version = ingest_pdf(db, pdf_path, document_name=req.document_name)
    node_count = db.query(DocumentNode).filter_by(version_id=version.id).count()

    return {
        "message": "Ingestion complete",
        "document": req.document_name,
        "version_number": version.version_number,
        "version_id": version.id,
        "source_filename": version.source_filename,
        "file_hash": version.file_hash,
        "node_count": node_count,
    }


# ---------------------------------------------------------------------------
# Documents & versions
# ---------------------------------------------------------------------------

@app.get("/documents", tags=["Browse"])
def list_documents(db: Session = Depends(get_db)):
    """List all ingested documents and their versions."""
    docs = db.query(Document).all()
    result = []
    for doc in docs:
        versions = (
            db.query(DocumentVersion)
            .filter_by(document_id=doc.id)
            .order_by(DocumentVersion.version_number)
            .all()
        )
        result.append({
            "id": doc.id,
            "name": doc.name,
            "versions": [
                {
                    "id": v.id,
                    "version_number": v.version_number,
                    "source_filename": v.source_filename,
                    "file_hash": v.file_hash,
                    "ingested_at": v.ingested_at.isoformat() if v.ingested_at else None,
                }
                for v in versions
            ],
        })
    return result


# ---------------------------------------------------------------------------
# Browse API
# ---------------------------------------------------------------------------

@app.get("/sections", tags=["Browse"])
def list_sections(
    version: Optional[int] = Query(default=None, description="Version number (default: latest)"),
    document: str = Query(default="CT-200 Manual"),
    db: Session = Depends(get_db),
):
    """List top-level sections for a given version (default: latest)."""
    if version is not None:
        doc = db.query(Document).filter_by(name=document).first()
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        ver = (
            db.query(DocumentVersion)
            .filter_by(document_id=doc.id, version_number=version)
            .first()
        )
        if ver is None:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
    else:
        ver = _latest_version(db, document)
        if ver is None:
            raise HTTPException(status_code=404, detail="No versions ingested yet. POST /ingest first.")

    top_nodes = (
        db.query(DocumentNode)
        .filter_by(version_id=ver.id, parent_stable_path=None)
        .order_by(DocumentNode.stable_path)
        .all()
    )
    return {
        "version_number": ver.version_number,
        "version_id": ver.id,
        "sections": [
            {
                "id": n.id,
                "stable_path": n.stable_path,
                "heading": n.heading,
                "level": n.level,
                "page_number": n.page_number,
                "content_hash": n.content_hash,
                "has_children": db.query(DocumentNode)
                    .filter_by(version_id=ver.id, parent_stable_path=n.stable_path)
                    .count() > 0,
            }
            for n in top_nodes
        ],
    }


@app.get("/sections/{node_id}", tags=["Browse"])
def get_section(
    node_id: int,
    db: Session = Depends(get_db),
):
    """Get a node by its SQLite ID, including its children, full text, and hash."""
    node = db.query(DocumentNode).filter_by(id=node_id).first()
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node_to_dict(node, include_children=True, db=db)


@app.get("/search", tags=["Browse"])
def search(
    q: str = Query(..., description="Search term"),
    version: Optional[int] = Query(default=None),
    document: str = Query(default="CT-200 Manual"),
    db: Session = Depends(get_db),
):
    """Full-text search across node headings and body text."""
    if version is not None:
        doc = db.query(Document).filter_by(name=document).first()
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        ver = db.query(DocumentVersion).filter_by(document_id=doc.id, version_number=version).first()
        if ver is None:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
        version_id = ver.id
    else:
        ver = _latest_version(db, document)
        if ver is None:
            raise HTTPException(status_code=404, detail="No versions found")
        version_id = ver.id

    q_lower = q.lower()
    nodes = db.query(DocumentNode).filter_by(version_id=version_id).all()
    results = []
    for n in nodes:
        if q_lower in n.heading.lower() or q_lower in n.body_text.lower():
            results.append({
                "id": n.id,
                "stable_path": n.stable_path,
                "heading": n.heading,
                "level": n.level,
                "body_text": n.body_text,
                "content_hash": n.content_hash,
            })
    return {"query": q, "version_id": version_id, "count": len(results), "results": results}


# ---------------------------------------------------------------------------
# Versioning & diff
# ---------------------------------------------------------------------------

@app.get("/versions/{vid1}/diff/{vid2}", tags=["Versioning"])
def version_diff(
    vid1: int,
    vid2: int,
    full_diff: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    """Compare two version IDs and return classified changes."""
    v1 = db.query(DocumentVersion).filter_by(id=vid1).first()
    v2 = db.query(DocumentVersion).filter_by(id=vid2).first()
    if v1 is None:
        raise HTTPException(status_code=404, detail=f"Version {vid1} not found")
    if v2 is None:
        raise HTTPException(status_code=404, detail=f"Version {vid2} not found")

    return versioning.compare_versions(db, vid1, vid2, include_full_diff=full_diff)


@app.get("/nodes/{node_id}/changes", tags=["Versioning"])
def node_changes(
    node_id: int,
    compare_version_id: Optional[int] = Query(
        default=None,
        description="Compare this node against a different version. Defaults to latest.",
    ),
    db: Session = Depends(get_db),
):
    """Return whether a node has changed across versions."""
    node = db.query(DocumentNode).filter_by(id=node_id).first()
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")

    if compare_version_id is None:
        lat = _latest_version(db)
        if lat is None:
            raise HTTPException(status_code=400, detail="No versions found")
        compare_version_id = lat.id

    return versioning.node_change_summary(
        db,
        stable_path=node.stable_path,
        version_id_old=node.version_id,
        version_id_new=compare_version_id,
    )


# ---------------------------------------------------------------------------
# Selections
# ---------------------------------------------------------------------------

@app.post("/selections", tags=["Selections"])
def create_selection(req: SelectionCreateRequest, db: Session = Depends(get_db)):
    """
    Create a version-pinned selection of nodes.

    The selection records the content_hash of each node at creation time.
    This snapshot survives re-ingestion — old selections still resolve to
    the exact text they were created against.
    """
    # Resolve version
    if req.version_id is not None:
        ver = db.query(DocumentVersion).filter_by(id=req.version_id).first()
        if ver is None:
            raise HTTPException(status_code=404, detail=f"Version {req.version_id} not found")
    else:
        ver = _latest_version(db)
        if ver is None:
            raise HTTPException(status_code=400, detail="No versions found — ingest first")

    # Validate node IDs
    nodes = []
    for nid in req.node_ids:
        node = db.query(DocumentNode).filter_by(id=nid, version_id=ver.id).first()
        if node is None:
            raise HTTPException(
                status_code=404,
                detail=f"Node {nid} not found in version {ver.id}",
            )
        nodes.append(node)

    selection = Selection(name=req.name, version_id=ver.id)
    db.add(selection)
    db.flush()

    for node in nodes:
        item = SelectionItem(
            selection_id=selection.id,
            node_id=node.id,
            pinned_content_hash=node.content_hash,
        )
        db.add(item)

    db.commit()
    db.refresh(selection)

    return {
        "id": selection.id,
        "name": selection.name,
        "version_id": selection.version_id,
        "created_at": selection.created_at.isoformat(),
        "node_count": len(nodes),
        "nodes": [{"id": n.id, "stable_path": n.stable_path, "heading": n.heading} for n in nodes],
    }


@app.get("/selections/{selection_id}", tags=["Selections"])
def get_selection(selection_id: int, db: Session = Depends(get_db)):
    """Get a selection by ID, including its pinned nodes."""
    sel = db.query(Selection).filter_by(id=selection_id).first()
    if sel is None:
        raise HTTPException(status_code=404, detail="Selection not found")

    items = db.query(SelectionItem).filter_by(selection_id=sel.id).all()
    nodes_info = []
    for item in items:
        node = db.query(DocumentNode).filter_by(id=item.node_id).first()
        nodes_info.append({
            "node_id": item.node_id,
            "stable_path": node.stable_path if node else None,
            "heading": node.heading if node else None,
            "pinned_content_hash": item.pinned_content_hash,
        })

    return {
        "id": sel.id,
        "name": sel.name,
        "version_id": sel.version_id,
        "created_at": sel.created_at.isoformat(),
        "generation_mongo_id": sel.generation_mongo_id,
        "nodes": nodes_info,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@app.post("/selections/{selection_id}/generate", tags=["Generation"])
def generate(
    selection_id: int,
    req: GenerateRequest = GenerateRequest(),
    db: Session = Depends(get_db),
):
    """
    Generate 3–5 QA test cases from the selected sections using an LLM.

    Duplicate policy: if a generation already exists for this selection and
    force=False, the existing generation is returned (idempotent).
    Set force=True to create a new generation regardless.
    """
    sel = db.query(Selection).filter_by(id=selection_id).first()
    if sel is None:
        raise HTTPException(status_code=404, detail="Selection not found")

    # Idempotency check
    if sel.generation_mongo_id and not req.force:
        existing = mongo_store.get_generation_by_id(sel.generation_mongo_id)
        if existing:
            existing["duplicate_policy"] = "returned_existing — use force=true to regenerate"
            return existing

    # Resolve pinned nodes
    items = db.query(SelectionItem).filter_by(selection_id=sel.id).all()
    if not items:
        raise HTTPException(status_code=400, detail="Selection has no nodes")

    sections = []
    pinned_hashes = {}
    node_summaries = []

    for item in items:
        node = db.query(DocumentNode).filter_by(id=item.node_id).first()
        if node is None:
            continue
        sections.append({
            "section_number": node.stable_path,
            "heading": node.heading,
            "body_text": node.body_text,
        })
        pinned_hashes[node.stable_path] = item.pinned_content_hash
        node_summaries.append({
            "stable_path": node.stable_path,
            "heading": node.heading,
            "body_preview": node.body_text[:200],
        })

    # Call LLM
    result = llm_module.generate_test_cases(sections)

    # Persist to MongoDB
    ver = db.query(DocumentVersion).filter_by(id=sel.version_id).first()
    mongo_doc = {
        "selection_id": sel.id,
        "selection_name": sel.name,
        "version_id": sel.version_id,
        "version_number": ver.version_number if ver else None,
        "pinned_hashes": pinned_hashes,
        "node_summaries": node_summaries,
        **result,
    }
    mongo_id = mongo_store.insert_generation(mongo_doc)

    # Update SQLite with the mongo ID (link the two stores)
    sel.generation_mongo_id = mongo_id
    db.commit()

    mongo_doc["_id"] = mongo_id
    return mongo_doc


@app.get("/selections/{selection_id}/generations", tags=["Generation"])
def list_generations_for_selection(selection_id: int, db: Session = Depends(get_db)):
    """List all generations linked to a selection."""
    sel = db.query(Selection).filter_by(id=selection_id).first()
    if sel is None:
        raise HTTPException(status_code=404, detail="Selection not found")

    generations = mongo_store.get_generations_by_selection(selection_id)
    return {"selection_id": selection_id, "count": len(generations), "generations": generations}


# ---------------------------------------------------------------------------
# Retrieval + staleness
# ---------------------------------------------------------------------------

@app.get("/generations/{mongo_id}", tags=["Retrieval"])
def get_generation(
    mongo_id: str,
    check_version_id: Optional[int] = Query(
        default=None,
        description=(
            "Version ID to check staleness against. "
            "Defaults to the latest version of the document the generation came from."
        ),
    ),
    db: Session = Depends(get_db),
):
    """
    Retrieve a previously generated test-case set with staleness status.

    The staleness check compares the hashes pinned at generation time against
    the specified (or latest) version.  If any source section has changed,
    the overall status is 'stale'.
    """
    gen = mongo_store.get_generation_by_id(mongo_id)
    if gen is None:
        raise HTTPException(status_code=404, detail="Generation not found")

    # Determine staleness reference version
    if check_version_id is not None:
        ref_ver = db.query(DocumentVersion).filter_by(id=check_version_id).first()
        if ref_ver is None:
            raise HTTPException(status_code=404, detail=f"Version {check_version_id} not found")
    else:
        ref_ver = _latest_version(db)

    staleness = {"overall": "unknown", "details": [], "note": "No reference version available"}
    if ref_ver is not None:
        pinned_hashes = gen.get("pinned_hashes", {})
        staleness = versioning.check_staleness(db, pinned_hashes, current_version_id=ref_ver.id)
        staleness["checked_against_version"] = ref_ver.version_number

    gen["staleness"] = staleness
    return gen


@app.get("/nodes/{node_id}/generations", tags=["Retrieval"])
def node_generations(
    node_id: int,
    db: Session = Depends(get_db),
):
    """Fetch all generations that reference a given node (by stable_path)."""
    node = db.query(DocumentNode).filter_by(id=node_id).first()
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")

    gens = mongo_store.get_generations_by_node_path(node.stable_path)
    return {
        "node_id": node_id,
        "stable_path": node.stable_path,
        "generation_count": len(gens),
        "generations": gens,
    }
