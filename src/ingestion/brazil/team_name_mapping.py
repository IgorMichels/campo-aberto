"""Maps raw club names scraped from CBF dockets to the normalized spelling
used across the dataset -- a "de-para" (raw -> normalized) table.

The mapping is entirely manually curated (see build_treated_dataset.py's
docstring and data/processed/brazil/README.md's "Resolving unmapped team
names" section): a raw name with no entry yet is left as-is in the treated
output and logged, with rapidfuzz suggestions (see suggest_matches), to
build_treated_dataset.py's unmapped-name log for a human to review and add.
"""

import csv
import os
import re
from typing import Optional

from rapidfuzz import fuzz, process, utils

from src.ingestion.brazil.constants import MAPPING_PATH, SUGGESTION_COUNT

STATE_PATTERN = re.compile(r"^(.*?)\s*/\s*([A-Za-z]{2})$")


def _state(team_name: str) -> Optional[str]:
    match = STATE_PATTERN.match(team_name.strip())
    return match.group(2).upper() if match else None


def load_mapping(path: str = MAPPING_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["raw_name"]: row["normalized_name"] for row in csv.DictReader(f)}


def save_mapping(mapping: dict, path: str = MAPPING_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_name", "normalized_name"])
        for raw_name in sorted(mapping):
            writer.writerow([raw_name, mapping[raw_name]])


def resolve_team_name(raw_name: str, lower_mapping: dict, lower_known_names: dict) -> tuple:
    """Resolves a raw club name to its normalized spelling, matching
    case-insensitively since CBF's own docket generator isn't consistent
    about it (e.g. "Santos FC / SP" vs "Santos Fc / SP").

    Returns (resolved_name, was_resolved). was_resolved is False when the raw
    name isn't in the mapping and doesn't already exactly match (up to case) a
    known canonical name -- that's the case that should be logged for review.
    """
    lowered = raw_name.lower()
    if lowered in lower_mapping:
        return lower_mapping[lowered], True
    if lowered in lower_known_names:
        return lower_known_names[lowered], True
    return raw_name, False


def build_lookup_tables(mapping: dict) -> tuple:
    """Precomputes case-insensitive lookup tables for resolve_team_name."""
    lower_mapping = {raw_name.lower(): normalized for raw_name, normalized in mapping.items()}
    lower_known_names = {name.lower(): name for name in mapping.values()}
    return lower_mapping, lower_known_names


def suggest_matches(raw_name: str, known_normalized_names: set) -> list:
    """Returns the top rapidfuzz suggestions for a raw name with no mapping
    yet, scoped to clubs from the same state when possible (plain full-string
    matching produces false positives across different states/clubs).
    """
    state = _state(raw_name)
    candidates = known_normalized_names
    if state is not None:
        same_state = {name for name in known_normalized_names if _state(name) == state}
        if same_state:
            candidates = same_state

    if not candidates:
        return []

    matches = process.extract(
        raw_name,
        candidates,
        scorer=fuzz.token_set_ratio,
        processor=utils.default_process,
        limit=SUGGESTION_COUNT,
    )
    return [(name, score) for name, score, _ in matches]
