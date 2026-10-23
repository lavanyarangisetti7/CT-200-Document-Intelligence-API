"""
Ingestion service — converts a parsed PDF tree into versioned DB records.

Version-matching strategy
--------------------------
Path-based matching using the section-number chain (stable_path, e.g. "3.2.1").

Rationale:
  The CT-200 manual uses structured numbered headings throughout. The section
  number is the most stable identifier across minor revisions — an author
  editing the *content* of section 4.1 will rarely renumber it. It is also
  human-readable and auditable.

Known failure modes (honest):
  1. If a section is renumbered (e.g. a new section 2 is inserted, pushing
     old section 3 → 4) every renumbered node will be flagged as
     added/deleted rather than moved — false alarm.
  2. Duplicate section numbers in the PDF (a formatting error we detect in
     parsing) will only keep the last-seen node under that path.
  3. If a section heading changes completely but the number stays the same,
     we'll report it as a changed node (body diff), not as a new node.
     This is arguably correct for a versioned document.

We document these in the approach doc and accept them given the time
constraint.  A fuzzy-title fallback matcher would improve case 1 at the cost
of more false positives in case 3 — not a clear win.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app import parser as pdf_parser
from app.database import Document, DocumentNode, DocumentVersion

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_pdf(
    db: Session,
    pdf_path: Path,
    document_name: str = "CT-200 Manual",
) -> DocumentVersion:
    """
    Parse `pdf_path` and persist it as a new version under `document_name`.

    Returns the newly created DocumentVersion ORM object.
    """
    # 1. Parse PDF into tree
    pages = pdf_parser.extract_pdf_text(pdf_path)
    tree = pdf_parser.build_hierarchy(pages)
    file_hash = pdf_parser.calculate_file_hash(pdf_path)

    # 2. Get or create the logical document record
    doc = db.query(Document).filter_by(name=document_name).first()
    if doc is None:
        doc = Document(name=document_name)
        db.add(doc)
        db.flush()

    # 3. Determine next version number
    existing_versions = (
        db.query(DocumentVersion)
        .filter_by(document_id=doc.id)
        .order_by(DocumentVersion.version_number.desc())
        .all()
    )
    next_version = (existing_versions[0].version_number + 1) if existing_versions else 1

    # 4. Create version record
    version = DocumentVersion(
        document_id=doc.id,
        version_number=next_version,
        source_filename=pdf_path.name,
        file_hash=file_hash,
    )
    db.add(version)
    db.flush()

    # 5. Flatten tree and persist nodes
    _persist_nodes(db, tree, version_id=version.id, parent_stable_path=None)

    db.commit()
    db.refresh(version)
    return version


def _persist_nodes(
    db: Session,
    nodes: list[dict],
    version_id: int,
    parent_stable_path: str | None,
) -> None:
    """Recursively persist nodes from the parser tree."""
    for node_dict in nodes:
        stable_path = node_dict["section_number"] or "root"

        db_node = DocumentNode(
            version_id=version_id,
            node_ref_id=node_dict["id"],
            stable_path=stable_path,
            heading=node_dict["heading"],
            level=node_dict["level"],
            body_text=node_dict["body_text"],
            content_hash=node_dict["content_hash"],
            page_number=node_dict.get("page_number"),
            parent_stable_path=parent_stable_path,
        )
        db.add(db_node)
        db.flush()

        _persist_nodes(db, node_dict.get("children", []), version_id, stable_path)
