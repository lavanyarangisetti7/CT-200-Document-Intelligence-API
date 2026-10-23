"""
MongoDB store for LLM-generated test-case output.

Design decisions
----------------
- We store each generation as a document in the `generations` collection.
- Each generation is linked to a SQLite selection_id AND carries a snapshot
  of the node content_hashes it was generated from.  This snapshot is what
  enables staleness detection: at retrieval time we compare stored hashes
  against the current hashes for the same stable paths.
- Connection is lazy (first use), so the app starts without MongoDB running
  and raises a clear error only when a generation endpoint is actually called.
- pymongo (sync) is used deliberately over motor (async) because our
  FastAPI routes run in a thread pool via run_in_executor at the call site,
  keeping the dependency surface small.
"""

from __future__ import annotations

import datetime
import os
from typing import Any

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "ct200_intelligence")

_client: MongoClient | None = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client


def _generations() -> Collection:
    return _get_client()[MONGO_DB_NAME]["generations"]


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def insert_generation(doc: dict) -> str:
    """
    Insert a generation document and return its string ID.

    Expected fields in `doc`:
        selection_id: int
        selection_name: str
        version_id: int
        pinned_hashes: dict[stable_path -> content_hash]  # staleness anchor
        node_summaries: list[{stable_path, heading, body_preview}]
        test_cases: list[{id, title, steps, expected_result}]
        raw_llm_response: str
        prompt_used: str
        generated_at: datetime (optional, added here if missing)
        is_valid: bool  (True when LLM output parsed successfully)
        parse_error: str | None
    """
    doc.setdefault("generated_at", datetime.datetime.utcnow())
    result = _generations().insert_one(doc)
    return str(result.inserted_id)


def get_generation_by_id(mongo_id: str) -> dict | None:
    """Retrieve a generation by its MongoDB ObjectId string."""
    try:
        obj = _generations().find_one({"_id": ObjectId(mongo_id)})
    except Exception:
        return None
    if obj is None:
        return None
    obj["_id"] = str(obj["_id"])
    return obj


def get_generations_by_selection(selection_id: int) -> list[dict]:
    """Return all generations for a selection, newest first."""
    cursor = _generations().find(
        {"selection_id": selection_id},
        sort=[("generated_at", -1)],
    )
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results


def get_generations_by_node_path(stable_path: str) -> list[dict]:
    """Return generations that include a specific stable_path."""
    cursor = _generations().find(
        {f"pinned_hashes.{stable_path}": {"$exists": True}},
        sort=[("generated_at", -1)],
    )
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results


def update_generation(mongo_id: str, update: dict) -> bool:
    """Partial update on a generation document."""
    result = _generations().update_one(
        {"_id": ObjectId(mongo_id)},
        {"$set": update},
    )
    return result.modified_count > 0
