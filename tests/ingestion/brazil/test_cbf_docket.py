"""Unit tests for cbf_docket.py's CBF PDF docket fetch/parse. Both requests.get
and pypdf's PdfReader are mocked -- these tests never hit the network or
parse a real PDF.
"""

from unittest import mock

import requests
from pypdf.errors import PdfReadError

from src.ingestion.brazil import cbf_docket

VALID_DOCKET_TEXT = """
Jogo: Santos / SP X Flamengo / RJ
Data: 08/08/2020
Horário: 19:00
Estádio: Vila Belmiro
Resultado Final: 2 X 1
"""


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, stream, text=VALID_DOCKET_TEXT):
        self.pages = [_FakePage(text)]


def _fake_reader(text):
    return lambda stream: _FakeReader(stream, text=text)


def test_parse_docket_extracts_every_field():
    with mock.patch.object(cbf_docket, "PdfReader", _fake_reader(VALID_DOCKET_TEXT)):
        docket = cbf_docket._parse_docket(b"irrelevant pdf bytes")

    assert docket == {
        "Date": "08/08/2020",
        "Time": "19:00",
        "Stadium": "Vila Belmiro",
        "Home": "Santos / SP",
        "Away": "Flamengo / RJ",
        "Result": "2 X 1",
    }


def test_parse_docket_returns_none_when_a_required_field_is_missing():
    with mock.patch.object(cbf_docket, "PdfReader", _fake_reader("Data: 08/08/2020")):
        assert cbf_docket._parse_docket(b"x") is None


def test_parse_docket_returns_none_on_an_unparseable_pdf():
    def raising_reader(stream):
        raise PdfReadError("malformed pdf")

    with mock.patch.object(cbf_docket, "PdfReader", raising_reader):
        assert cbf_docket._parse_docket(b"garbage") is None


def test_download_docket_content_truncates_after_the_first_eof_marker():
    """CBF sometimes appends garbage bytes after the real PDF trailer."""

    class _Response:
        status_code = 200
        content = b"PDF-CONTENT%%EOFGARBAGE-AFTER"

    with mock.patch.object(cbf_docket, "_get_with_retries", lambda url: _Response()):
        content = cbf_docket._download_docket_content("Serie_A", 2020, 1)

    assert content == b"PDF-CONTENT%%EOF"


def test_download_docket_content_returns_none_on_a_non_200_status():
    class _Response:
        status_code = 404
        content = b"not found"

    with mock.patch.object(cbf_docket, "_get_with_retries", lambda url: _Response()):
        assert cbf_docket._download_docket_content("Serie_A", 2020, 1) is None


def test_download_docket_content_returns_none_when_every_retry_fails():
    with mock.patch.object(cbf_docket, "_get_with_retries", lambda url: None):
        assert cbf_docket._download_docket_content("Serie_A", 2020, 1) is None


def test_get_with_retries_retries_transient_network_errors(monkeypatch):
    monkeypatch.setattr(cbf_docket.time, "sleep", lambda seconds: None)
    calls = {"count": 0}

    def flaky_get(url, timeout):
        calls["count"] += 1
        if calls["count"] < cbf_docket.RETRY_ATTEMPTS:
            raise requests.exceptions.ConnectionError("boom")

        class _Response:
            status_code = 200
            content = b"ok"

        return _Response()

    monkeypatch.setattr(cbf_docket.requests, "get", flaky_get)

    response = cbf_docket._get_with_retries("http://example.test")

    assert response.content == b"ok"
    assert calls["count"] == cbf_docket.RETRY_ATTEMPTS


def test_get_with_retries_gives_up_after_every_attempt_fails(monkeypatch):
    monkeypatch.setattr(cbf_docket.time, "sleep", lambda seconds: None)
    calls = {"count": 0}

    def always_fail(url, timeout):
        calls["count"] += 1
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(cbf_docket.requests, "get", always_fail)

    assert cbf_docket._get_with_retries("http://example.test") is None
    assert calls["count"] == cbf_docket.RETRY_ATTEMPTS


def test_try_fetch_docket_returns_none_when_the_docket_does_not_exist_yet():
    """A game not yet played (or not yet published) has no docket -- this is
    the expected way scrape_season probes for the season's live edge, not an
    error."""
    with mock.patch.object(cbf_docket, "_download_docket_content", lambda *a: None):
        assert cbf_docket.try_fetch_docket("Serie_A", 2020, 999) is None


def test_try_fetch_docket_parses_a_successful_download():
    with (
        mock.patch.object(cbf_docket, "_download_docket_content", lambda *a: b"bytes"),
        mock.patch.object(cbf_docket, "PdfReader", _fake_reader(VALID_DOCKET_TEXT)),
    ):
        docket = cbf_docket.try_fetch_docket("Serie_A", 2020, 1)

    assert docket["Home"] == "Santos / SP"
