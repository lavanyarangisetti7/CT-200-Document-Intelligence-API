
from pathlib import Path
import hashlib
import json
import re
import unicodedata

import pymupdf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "ct200_manual.pdf"
OUTPUT_PATH = BASE_DIR / "parsed_tree.json"


# --------------------------------------------------
# Text Utilities
# --------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize Unicode characters and whitespace."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    """Generate a SHA-256 hash of normalized text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# --------------------------------------------------
# PDF Text Extraction
# --------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> list[dict]:
    """Extract text from each page of a PDF."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            pages.append({
                "page_number": page_number,
                "text": page.get_text("text").strip()
            })

    return pages


def calculate_file_hash(pdf_path: Path) -> str:
    """Calculate a SHA-256 hash of the complete PDF file."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    sha256 = hashlib.sha256()

    with pdf_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

# --------------------------------------------------
# Heading Detection
# --------------------------------------------------

def get_heading(line: str):
    """
    Recognize numbered section headings.

    Examples:
        1. Device Overview
        1.1 Intended Use
        2.1.1.1 Battery Life Under Typical Use

    Excludes numbered list entries such as:
        1. Normal: systolic < 120
        2. Elevated: systolic 120-129
    """
    line = normalize_text(line)

    # Ignore numbered list entries with a colon or closing parenthesis.
    if re.match(r"^\d+\s*[:)]\s*", line):
        return None

    # Match numbered headings.
    match = re.match(
        r"^(\d+(?:\.\d+)*\.?)\s+(.+)$",
        line
    )

    if not match:
        return None

    number = match.group(1).rstrip(".")
    title = match.group(2).strip()

    if not title or len(title) > 150:
        return None

    # A heading should begin with an uppercase letter.
    if not title[0].isupper():
        return None

    # Exclude numbered category/list lines ending in a colon-based
    # description, such as "1. Normal: systolic < 120".
    if ":" in title:
        return None

    return number, title


# --------------------------------------------------
# Hierarchy Construction
# --------------------------------------------------

def build_hierarchy(pages: list[dict]) -> list[dict]:
    """Build a tree from numbered PDF headings."""
    root = {
        "id": "document-root",
        "heading": "Document",
        "section_number": None,
        "level": 0,
        "body_text": "",
        "content_hash": "",
        "page_number": None,
        "children": []
    }

    # Each entry contains (heading level, node).
    stack = [(0, root)]

    for page in pages:
        page_number = page["page_number"]

        for raw_line in page["text"].splitlines():
            line = normalize_text(raw_line)

            if not line:
                continue

            heading = get_heading(line)

            if heading is not None:
                section_number, title = heading
                level = section_number.count(".") + 1

                # Pop the current section and any deeper sections.
                while len(stack) > 1 and stack[-1][0] >= level:
                    stack.pop()

                parent = stack[-1][1]

                node = {
                    "id": (
                        f"section-{section_number}-"
                        f"{page_number}-"
                        f"{len(parent['children']) + 1}"
                    ),
                    "heading": title,
                    "section_number": section_number,
                    "level": level,
                    "body_text": "",
                    "content_hash": "",
                    "page_number": page_number,
                    "children": []
                }

                parent["children"].append(node)
                stack.append((level, node))

            else:
                # Preserve ordinary text, tables extracted as text,
                # and numbered list entries in the current section.
                current_node = stack[-1][1]
                current_node["body_text"] += line + "\n"

    def finalize(node: dict) -> None:
        """Normalize body text and calculate content hashes."""
        node["body_text"] = node["body_text"].strip()
        node["content_hash"] = content_hash(node["body_text"])

        for child in node["children"]:
            finalize(child)

    finalize(root)

    return root["children"]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    """Run PDF extraction and save the hierarchy as JSON."""
    pages = extract_pdf_text(PDF_PATH)
    tree = build_hierarchy(pages)

    OUTPUT_PATH.write_text(
        json.dumps(tree, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"PDF: {PDF_PATH.name}")
    print(f"Total pages: {len(pages)}")
    print(f"Top-level sections: {len(tree)}")
    print(f"Saved hierarchy to: {OUTPUT_PATH}")

    print("\nTop-level headings:")
    for node in tree:
        print(f"{node['section_number']}: {node['heading']}")


if __name__ == "__main__":
    main()
