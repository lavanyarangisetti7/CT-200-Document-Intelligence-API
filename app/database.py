"""
SQLAlchemy database layer for CT-200 document intelligence system.

Stores: documents, versions, nodes (hierarchical tree), selections, and generation links.
NoSQL (MongoDB) is used separately for LLM-generated test-case output.
"""

from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'ct200.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Enable WAL mode for better concurrency
@event.listens_for(engine, "connect")
def set_wal_mode(dbapi_connection, connection_record):
    dbapi_connection.execute("PRAGMA journal_mode=WAL")
    dbapi_connection.execute("PRAGMA foreign_keys=ON")


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Document(Base):
    """A logical document (e.g. CT-200 Manual). Multiple versions belong to it."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")


class DocumentVersion(Base):
    """A specific ingested version of a document."""
    __tablename__ = "document_versions"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    version_number = Column(Integer, nullable=False)   # 1, 2, 3, …
    source_filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False)     # SHA-256 of the PDF file
    ingested_at = Column(DateTime, default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="versions")
    nodes = relationship("DocumentNode", back_populates="version", cascade="all, delete-orphan")


class DocumentNode(Base):
    """
    A single node in the document tree (section / subsection / paragraph).

    Matching between versions
    -------------------------
    We use a "stable path" strategy: the stable_path is the dot-separated
    chain of section_numbers from the root (e.g. "3.2.1").  This is the
    primary key for recognising the *same logical node* across versions.

    When stable_path AND content_hash are identical across versions, the node
    is considered unchanged.  When only the path matches but the hash differs,
    the node body has been edited. When a path exists in v1 but not v2, the
    node was deleted.  When a path exists in v2 but not v1, the node is new.

    Known failure modes of this strategy (documented in Approach Doc):
    - If a section is renumbered (e.g. 3.1 becomes 4.1 due to an insertion
      above it) the matcher will treat old 3.1 as deleted and new 4.1 as
      added — a false negative.
    - If two independent nodes share the same numbering on different pages
      (document formatting error) only the later one survives the path index.
    """
    __tablename__ = "document_nodes"

    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey("document_versions.id"), nullable=False)
    node_ref_id = Column(String(128), nullable=False)   # e.g. "section-1.1-1-1" (parser ID)
    stable_path = Column(String(128), nullable=False)   # e.g. "1.1" (section number chain)
    heading = Column(String(512), nullable=False)
    level = Column(Integer, nullable=False)
    body_text = Column(Text, nullable=False, default="")
    content_hash = Column(String(64), nullable=False)
    page_number = Column(Integer, nullable=True)
    parent_stable_path = Column(String(128), nullable=True)   # None for top-level

    version = relationship("DocumentVersion", back_populates="nodes")
    selection_items = relationship("SelectionItem", back_populates="node")


class Selection(Base):
    """
    A named, version-pinned set of nodes submitted by the user.

    Version-pinning: each SelectionItem records the exact (node_id, content_hash)
    pair, so the selection always resolves to the original text even after
    re-ingestion.
    """
    __tablename__ = "selections"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    version_id = Column(Integer, ForeignKey("document_versions.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # MongoDB ObjectId of the linked generation (nullable until generated)
    generation_mongo_id = Column(String(64), nullable=True)

    items = relationship("SelectionItem", back_populates="selection", cascade="all, delete-orphan")
    version = relationship("DocumentVersion")


class SelectionItem(Base):
    """One node pinned inside a selection."""
    __tablename__ = "selection_items"

    id = Column(Integer, primary_key=True, index=True)
    selection_id = Column(Integer, ForeignKey("selections.id"), nullable=False)
    node_id = Column(Integer, ForeignKey("document_nodes.id"), nullable=False)
    pinned_content_hash = Column(String(64), nullable=False)   # hash at selection time

    selection = relationship("Selection", back_populates="items")
    node = relationship("DocumentNode", back_populates="selection_items")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db():
    """FastAPI dependency — yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
