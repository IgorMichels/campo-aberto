"""Unit tests for team_name_mapping.py's raw -> normalized club name resolution.

The mapping itself is entirely manually curated (no external source to mock
or hit here) -- see team_name_mapping.py's docstring.
"""

from src.ingestion.brazil import team_name_mapping as tnm


def test_state_extracts_the_trailing_two_letter_state_code():
    assert tnm._state("Santos / SP") == "SP"
    assert tnm._state("Santos/SP") == "SP"
    assert tnm._state("  Flamengo  /  RJ  ") == "RJ"


def test_state_returns_none_when_there_is_no_state_suffix():
    assert tnm._state("Santos") is None


def test_build_lookup_tables_lowercases_both_directions():
    mapping = {"Santos Fc": "Santos / SP", "flamengo": "Flamengo / RJ"}

    lower_mapping, lower_known_names = tnm.build_lookup_tables(mapping)

    assert lower_mapping == {"santos fc": "Santos / SP", "flamengo": "Flamengo / RJ"}
    assert lower_known_names == {"santos / sp": "Santos / SP", "flamengo / rj": "Flamengo / RJ"}


def test_resolve_team_name_matches_a_known_raw_alias_case_insensitively():
    lower_mapping, lower_known_names = tnm.build_lookup_tables({"Santos Fc": "Santos / SP"})

    assert tnm.resolve_team_name("SANTOS FC", lower_mapping, lower_known_names) == (
        "Santos / SP",
        True,
    )


def test_resolve_team_name_matches_an_already_canonical_name_case_insensitively():
    """CBF's own docket generator isn't consistent about case (e.g. 'Santos Fc /
    SP' vs 'Santos FC / SP') even for names that are already canonical."""
    lower_mapping, lower_known_names = tnm.build_lookup_tables({"Santos Fc": "Santos / SP"})

    assert tnm.resolve_team_name("santos / sp", lower_mapping, lower_known_names) == (
        "Santos / SP",
        True,
    )


def test_resolve_team_name_reports_unresolved_names():
    lower_mapping, lower_known_names = tnm.build_lookup_tables({"Santos Fc": "Santos / SP"})

    assert tnm.resolve_team_name("Unknown FC / XX", lower_mapping, lower_known_names) == (
        "Unknown FC / XX",
        False,
    )


def test_suggest_matches_prefers_same_state_candidates():
    known = {"Santos / SP", "Santos Futebol Clube / SP", "Flamengo / RJ"}

    suggestions = tnm.suggest_matches("Santoss / SP", known)

    # Both SP candidates are suggested, ranked by similarity; the RJ one (a
    # different state) is excluded even though it's still string-similar.
    names = [name for name, _score in suggestions]
    assert names == ["Santos / SP", "Santos Futebol Clube / SP"]
    assert "Flamengo / RJ" not in names


def test_suggest_matches_falls_back_to_all_candidates_with_no_same_state_match():
    known = {"Santos Futebol Clube / SP", "Flamengo / RJ"}

    suggestions = tnm.suggest_matches("Totally Unrelated Name / XX", known)

    assert {name for name, _score in suggestions} == known


def test_suggest_matches_returns_empty_for_no_candidates():
    assert tnm.suggest_matches("Anything / SP", set()) == []


def test_load_and_save_mapping_round_trip(tmp_path):
    path = tmp_path / "team_name_mapping.csv"
    mapping = {"Santos Fc": "Santos / SP", "Flamengo Rj": "Flamengo / RJ"}

    tnm.save_mapping(mapping, path=str(path))
    loaded = tnm.load_mapping(path=str(path))

    assert loaded == mapping


def test_load_mapping_returns_empty_dict_when_file_does_not_exist(tmp_path):
    assert tnm.load_mapping(path=str(tmp_path / "missing.csv")) == {}
