"""
LLM-powered test-case generation.

Provider: Groq (free tier, llama-3.3-70b-versatile)
Fallback: Returns a structured error response — never crashes the API.

Prompt design
-------------
The prompt instructs the model to act as a QA engineer for a medical device
and to return a JSON array of 3–5 test cases.  Each test case has:
  - id: string slug
  - title: short human label
  - steps: list[str] — the exact test procedure
  - expected_result: str — what the device should do
  - traceability: list[str] — section numbers this covers

We request JSON mode (response_format={"type": "json_object"}) where the
provider supports it; otherwise we attempt to extract JSON from the raw text.

Structured-output validation
-----------------------------
We use pydantic models to validate the LLM response before storing it.
If validation fails (malformed JSON, missing required fields, wrong types):
  - We store the raw response with is_valid=False and a parse_error string
  - We do NOT surface a 500 error to the caller; instead we return the stored
    raw text alongside a "parse_failed" flag
  - We do NOT auto-retry (retrying on a bad pydantic model risks infinite
    loops on consistently malformed responses; the engineer should fix the
    prompt instead)

Duplicate selection policy
--------------------------
If the same selection is submitted twice for generation:
  - We return the most recent existing generation (idempotent read)
  - A `force=True` query param allows creating a fresh generation
  This avoids wasting LLM credits on unchanged content while still
  allowing re-generation when needed.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, ValidationError

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class TestCase(BaseModel):
    id: str
    title: str
    steps: list[str]
    expected_result: str
    traceability: list[str] = []


class TestCaseBatch(BaseModel):
    test_cases: list[TestCase]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior QA engineer specializing in IEC 62304-compliant software testing \
for medical devices.  Your job is to generate thorough, concrete QA test cases from \
device manual sections.

RULES:
1. Output ONLY valid JSON — no prose, no markdown code fences.
2. The JSON must be a single object with a key "test_cases" containing an array of \
3 to 5 test case objects.
3. Each test case object must have exactly these fields:
   - "id": a unique snake_case slug (e.g. "tc_overpressure_auto_deflate")
   - "title": a short, specific human-readable label
   - "steps": a list of strings, each a numbered action to perform
   - "expected_result": a single string stating what should happen
   - "traceability": a list of section number strings that this test covers \
(e.g. ["4.1", "4.3"])
4. Test cases must be specific enough that a different engineer could execute them \
without reading the source section.  Include concrete values (e.g. "299 mmHg", \
"2 seconds") taken directly from the text.
5. Do not invent requirements not present in the provided text.
"""

def build_user_prompt(sections: list[dict]) -> str:
    """
    Build the user portion of the prompt from a list of section dicts.
    Each dict: {section_number, heading, body_text}
    """
    parts = ["DOCUMENT SECTIONS TO COVER:\n"]
    for s in sections:
        parts.append(f"Section {s['section_number']}: {s['heading']}")
        parts.append(s["body_text"])
        parts.append("")
    parts.append("Generate the test cases JSON now.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_groq(system: str, user: str) -> tuple[str, str | None]:
    """
    Call the Groq API and return (raw_text, error_message).

    We do NOT raise exceptions — all errors are returned as (raw, error).
    This keeps the calling code's error path simple.
    """
    if not GROQ_API_KEY:
        return "", "GROQ_API_KEY not set — cannot call LLM"

    try:
        from groq import Groq  # import here to fail gracefully if not installed
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        return raw, None
    except Exception as exc:
        return "", str(exc)


# ---------------------------------------------------------------------------
# JSON extraction & validation
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict | None:
    """
    Try to extract a JSON object from the raw LLM string.
    Handles cases where the model adds markdown fences or prose.
    """
    raw = raw.strip()

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block wrapped in ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Brute-force: find first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_and_validate(raw: str) -> tuple[list[dict] | None, str | None]:
    """
    Parse and validate LLM output into a list of TestCase dicts.

    Returns (test_cases, error_message).
    On success: (list_of_dicts, None)
    On failure: (None, error_string)
    """
    blob = _extract_json(raw)
    if blob is None:
        return None, f"Could not extract JSON from LLM response (first 200 chars): {raw[:200]}"

    try:
        batch = TestCaseBatch.model_validate(blob)
        return [tc.model_dump() for tc in batch.test_cases], None
    except ValidationError as exc:
        return None, f"Pydantic validation failed: {exc.json(indent=2)}"


# ---------------------------------------------------------------------------
# High-level generate function
# ---------------------------------------------------------------------------

def generate_test_cases(sections: list[dict]) -> dict:
    """
    Given resolved section dicts, call the LLM and return a generation result.

    Returns:
        {
          test_cases: list[dict] | [],
          raw_llm_response: str,
          prompt_used: str,
          is_valid: bool,
          parse_error: str | None,
        }
    """
    user_prompt = build_user_prompt(sections)
    raw, call_error = call_groq(SYSTEM_PROMPT, user_prompt)

    if call_error:
        return {
            "test_cases": [],
            "raw_llm_response": "",
            "prompt_used": user_prompt,
            "is_valid": False,
            "parse_error": f"LLM call failed: {call_error}",
        }

    test_cases, parse_error = parse_and_validate(raw)

    return {
        "test_cases": test_cases or [],
        "raw_llm_response": raw,
        "prompt_used": user_prompt,
        "is_valid": parse_error is None,
        "parse_error": parse_error,
    }
