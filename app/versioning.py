"""
Versioning helpers — compare two document versions and detect changes.

Diff strategy
------------
Given v_old and v_new (by version_id), we index all nodes by stable_path.
Then we classify each node as:
  - unchanged  : same stable_path, same content_hash
  - modified   : same stable_path, different content_hash
  - added      : stable_path present only in v_new
  - deleted    : stable_path present only in v_old

The lightweight "diff summary" we expose is the first 200 chars of old vs
new body_text.  We deliberately do not compute a line-level diff (difflib)
by default — that would be expensive for large documents and the caller can
request it via `include_full_diff=True`.

Staleness detection
-------------------
A generation stores `pinned_hashes: {stable_path: content_hash}` at the time
it was created.  Staleness is determined by comparing each pinned hash against
the *current latest version's* hash for that same stable_path.

Three outcomes:
  - "current"  : all pinned hashes match the latest version
  - "stale"    : at least one node has changed (hash mismatch)
  - "missing"  : a node's stable_path no longer exists in the latest version

Honest limits:
  A one-word wording change (e.g. "2 seconds" → "3 seconds" for an overpressure
  threshold) gets the same "stale" flag as a completely rewritten section.
  The system does NOT distinguish safety-critical numeric changes from editorial
  tweaks — that requires NLP/LLM parsing on top of hash detection, which is
  out of scope.  We document this clearly.
"""

from __future__ import annotations

import difflib
from typing import Optional

from sqlalchemy.orm import Session

from app.database import DocumentNode, DocumentVersion


# ---------------------------------------------------------------------------
# Core diff
# ---------------------------------------------------------------------------

def compare_versions(
    db: Session,
    version_id_old: int,
    version_id_new: int,
    include_full_diff: bool = False,
) -> dict:
    """
    Return a structured diff between two document versions.

    Returns:
        {
          "version_old": int,
          "version_new": int,
          "unchanged": [stable_path, ...],
          "modified": [{stable_path, heading, old_hash, new_hash, summary}],
          "added": [stable_path, ...],
          "deleted": [stable_path, ...],
        }
    """
    old_nodes = {
        n.stable_path: n
        for n in db.query(DocumentNode).filter_by(version_id=version_id_old).all()
    }
    new_nodes = {
        n.stable_path: n
        for n in db.query(DocumentNode).filter_by(version_id=version_id_new).all()
    }

    all_paths = set(old_nodes) | set(new_nodes)

    unchanged, modified, added, deleted = [], [], [], []

    for path in sorted(all_paths):
        old = old_nodes.get(path)
        new = new_nodes.get(path)

        if old and new:
            if old.content_hash == new.content_hash:
                unchanged.append(path)
            else:
                entry = {
                    "stable_path": path,
                    "heading": new.heading,
                    "old_hash": old.content_hash,
                    "new_hash": new.content_hash,
                    "old_preview": old.body_text[:200],
                    "new_preview": new.body_text[:200],
                }
                if include_full_diff:
                    entry["full_diff"] = _text_diff(old.body_text, new.body_text)
                modified.append(entry)
        elif new and not old:
            added.append(path)
        else:
            deleted.append(path)

    return {
        "version_old": version_id_old,
        "version_new": version_id_new,
        "unchanged_count": len(unchanged),
        "unchanged": unchanged,
        "modified": modified,
        "added": added,
        "deleted": deleted,
    }


def node_change_summary(
    db: Session,
    stable_path: str,
    version_id_old: int,
    version_id_new: int,
) -> dict:
    """Return change status for a single node across two versions."""
    old_node = (
        db.query(DocumentNode)
        .filter_by(version_id=version_id_old, stable_path=stable_path)
        .first()
    )
    new_node = (
        db.query(DocumentNode)
        .filter_by(version_id=version_id_new, stable_path=stable_path)
        .first()
    )

    if old_node is None and new_node is None:
        return {"status": "not_found"}
    if old_node is None:
        return {"status": "added", "stable_path": stable_path}
    if new_node is None:
        return {"status": "deleted", "stable_path": stable_path}
    if old_node.content_hash == new_node.content_hash:
        return {"status": "unchanged", "stable_path": stable_path}

    return {
        "status": "modified",
        "stable_path": stable_path,
        "old_hash": old_node.content_hash,
        "new_hash": new_node.content_hash,
        "old_preview": old_node.body_text[:300],
        "new_preview": new_node.body_text[:300],
        "full_diff": _text_diff(old_node.body_text, new_node.body_text),
    }


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

def check_staleness(
    db: Session,
    pinned_hashes: dict,   # {stable_path: content_hash}
    current_version_id: int,
) -> dict:
    """
    Compare pinned node hashes against a specific document version.

    Returns:
        {
          "overall": "current" | "stale" | "mixed" | "error",
          "details": [{stable_path, status: "current"|"stale"|"missing", ...}]
        }
    """
    details = []
    any_stale = False
    any_missing = False

    for stable_path, pinned_hash in pinned_hashes.items():
        node = (
            db.query(DocumentNode)
            .filter_by(version_id=current_version_id, stable_path=stable_path)
            .first()
        )
        if node is None:
            details.append({
                "stable_path": stable_path,
                "status": "missing",
                "note": "Section no longer exists in the referenced version",
            })
            any_missing = True
        elif node.content_hash == pinned_hash:
            details.append({
                "stable_path": stable_path,
                "status": "current",
                "heading": node.heading,
            })
        else:
            details.append({
                "stable_path": stable_path,
                "status": "stale",
                "heading": node.heading,
                "pinned_hash": pinned_hash,
                "current_hash": node.content_hash,
                "current_preview": node.body_text[:300],
                "note": (
                    "The section body has changed since this test case was generated. "
                    "A one-word change is treated identically to a major rewrite — "
                    "manual review is required to determine impact."
                ),
            })
            any_stale = True

    if any_stale and any_missing:
        overall = "mixed"
    elif any_stale:
        overall = "stale"
    elif any_missing:
        overall = "missing"
    else:
        overall = "current"

    return {"overall": overall, "details": details}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _text_diff(old: str, new: str) -> list[str]:
    """Unified diff between two text blocks."""
    return list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="v_old",
            tofile="v_new",
        )
    )
