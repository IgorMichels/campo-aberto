"""Fetches and parses official CBF match dockets (súmulas) directly from cbf.com.br.

This module only scrapes: it returns fields exactly as they appear on the
docket, with no team-name treatment. Name normalization is a separate,
downstream concern (see team_name_mapping.py).
"""

import io
import re
import time
from typing import Optional

import requests
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from src.ingestion.brazil.constants import (
    COMPETITION_CODES,
    DOCKET_URL,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
)

TEAM_PATTERN = re.compile(
    r"Jogo:\s*([a-zA-Z0-9À-ÿ\s\.\-]+/\s*[A-Z]{2})\s*X\s*([a-zA-Z0-9À-ÿ\s\.\-]+/\s*[A-Z]{2})"
)
DATE_PATTERN = re.compile(r"Data:\s*(\d{2}/\d{2}/\d{4})")
TIME_PATTERN = re.compile(r"Horário:\s*(\d{2}:\d{2})")
STADIUM_PATTERN = re.compile(r"Estádio:\s*(.+?)(?=\n|$)")
RESULT_PATTERN = re.compile(r"Resultado\s*Final:\s*(\d+\s*[xX]\s*\d+)")


def _get_with_retries(url: str) -> Optional[requests.Response]:
    """GETs a URL, retrying on transient network errors (CBF's server is
    occasionally slow or drops connections) instead of letting one flaky
    request kill an entire scraping run. Returns None if every attempt fails.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return requests.get(url, timeout=30)
        except requests.exceptions.RequestException:
            if attempt == RETRY_ATTEMPTS - 1:
                return None
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    return None


def _download_docket_content(competition: str, year: int, game: int) -> Optional[bytes]:
    url = DOCKET_URL.format(code=COMPETITION_CODES[competition], year=year, game=game)
    response = _get_with_retries(url)
    if response is None or response.status_code != 200:
        return None

    content = response.content
    # The CBF server sometimes appends garbage bytes after the real PDF trailer,
    # which breaks strict PDF parsers. Truncate right after the first %%EOF marker.
    eof_index = content.find(b"%%EOF")
    if eof_index != -1:
        content = content[: eof_index + len(b"%%EOF")]
    return content


def _parse_docket(content: bytes) -> Optional[dict]:
    try:
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() for page in reader.pages)
    except (PdfReadError, ValueError):
        # PyPDF2 raises assorted errors (not just PdfReadError) on the malformed
        # PDFs CBF occasionally serves; treat any of them as "unparseable".
        return None

    team_match = TEAM_PATTERN.search(text)
    result_match = RESULT_PATTERN.search(text)
    date_match = DATE_PATTERN.search(text)
    time_match = TIME_PATTERN.search(text)
    stadium_match = STADIUM_PATTERN.search(text)
    if not (team_match and result_match and date_match and time_match and stadium_match):
        return None

    home_team, away_team = team_match.groups()
    home_goals, away_goals = result_match.group(1).upper().split(" X ")

    return {
        "Date": date_match.group(1),
        "Time": time_match.group(1),
        "Stadium": stadium_match.group(1).strip(),
        "Home": home_team.strip(),
        "Away": away_team.strip(),
        "Result": f"{home_goals} X {away_goals}",
    }


def try_fetch_docket(competition: str, year: int, game: int) -> Optional[dict]:
    """Fetches and parses a single game's docket.

    Returns None if the docket doesn't exist yet (game not played) or is
    incomplete, instead of raising, so callers can use it to probe for games
    whose existence isn't known in advance.
    """
    content = _download_docket_content(competition, year, game)
    if content is None:
        return None
    return _parse_docket(content)
