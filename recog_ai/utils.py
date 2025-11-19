"""Utility functions for JSON parsing and metadata extraction."""

import json
import isodate
import logging

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict:
    """
    Extract JSON from text, attempting direct parse then bracketed fallback.

    Args:
        text: Raw text potentially containing JSON.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        json.JSONDecodeError: If no valid JSON is found.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise


def parse_workload(metadata: dict) -> str:
    """
    Extract and format workload from module metadata.

    Attempts to parse duration in ISO format (e.g., "P0Y1M0DT0H0M0S"),
    falls back to credits-based estimate, or returns empty string.

    Args:
        metadata: Module metadata dictionary.

    Returns:
        Formatted workload string (e.g., "90 Stunden") or empty string.
    """
    try:
        workload_iso = metadata.get("duration") or metadata.get("workload")
        if workload_iso:
            duration = isodate.parse_duration(workload_iso)
            hours = int(duration.total_seconds() / 3600)
            return str(hours) + " Stunden"
    except Exception:
        pass

    credits = metadata.get("credits")
    if credits:
        return "~" + str(int(credits) * 30) + " Stunden"

    return ""


def collect_programs(metadata: dict) -> str:
    """
    Extract and format program list from module metadata.

    Args:
        metadata: Module metadata dictionary.

    Returns:
        Comma-separated program names or empty string.
    """
    programs = metadata.get("programs") or metadata.get("program")
    if isinstance(programs, (list, tuple)):
        return ", ".join(programs)
    return programs or ""
