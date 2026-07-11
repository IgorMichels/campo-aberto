"""Unit + integration tests for src.site.export_matches_data.

Uses tiny fixture CSVs/crests under tmp_path for results_dir/matches_path/
club_infos_path/site_dir, but reuses the real configs/serie_*.yaml (via
load_configs_by_season, same as src.simulation.run_rounds and
src.site.export_site_data) for competition/season discovery -- mirrors
tests/site/test_export_site_data.py's tmp_path-fixture-CSV style.
"""

import json

import pandas as pd
import pytest

from src.site.export_matches_data import (
    _latest_results_csv,
    _latest_results_csv_by_competition,
    _load_params,
    _played_cards,
    _read_snapshot_params,
    _upcoming_cards,
    export_matches_data,
)


def _write_results_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_crest(path, content: bytes = b"crest-bytes") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _write_matches_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _matches_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    return df


def _strengths_row(team, attack, defense, eta=0.2, beta_home=0.3, rho=-0.05):
    return {
        "team": team,
        "expected_position": 1.0,
        "prob_title": 0.5,
        "attack": attack,
        "defense": defense,
        "eta": eta,
        "beta_home": beta_home,
        "rho": rho,
    }


NOW = pd.Timestamp("2026-07-09 12:00")


# ---------------------------------------------------------------------------
# _latest_results_csv / _load_params
# ---------------------------------------------------------------------------


def test_latest_results_csv_picks_globally_latest_across_competitions(tmp_path):
    _write_results_csv(
        tmp_path / "serie_a" / "2026" / "2026_01_01.csv", [_strengths_row("Team A", 0.1, 0.2)]
    )
    later = tmp_path / "serie_b" / "2026" / "2026_02_01.csv"
    _write_results_csv(later, [_strengths_row("Team B", 0.3, 0.4)])

    assert _latest_results_csv(str(tmp_path)) == str(later)


def test_latest_results_csv_raises_when_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        _latest_results_csv(str(tmp_path))


def test_latest_results_csv_by_competition_picks_one_per_slug(tmp_path):
    # serie_a's own latest round predates serie_b's -- both should still
    # surface, each pointing at ITS OWN latest file, not the globally-latest
    # one (this is the real, currently-occurring scenario -- see
    # _load_params's docstring).
    serie_a_older = tmp_path / "serie_a" / "2026" / "2026_06_01.csv"
    serie_a_newer_than_a_but_older_than_b = tmp_path / "serie_a" / "2026" / "2026_07_01.csv"
    serie_b_latest = tmp_path / "serie_b" / "2026" / "2026_07_09.csv"
    for path in (serie_a_older, serie_a_newer_than_a_but_older_than_b, serie_b_latest):
        _write_results_csv(path, [_strengths_row("Team X", 0.1, 0.2)])

    by_slug = _latest_results_csv_by_competition(str(tmp_path))

    assert by_slug == {
        "serie_a": str(serie_a_newer_than_a_but_older_than_b),
        "serie_b": str(serie_b_latest),
    }


def test_load_params_reads_shared_scalars_and_every_team(tmp_path):
    _write_results_csv(
        tmp_path / "serie_a" / "2026" / "2026_01_01.csv",
        [_strengths_row("Team A", 0.1, 0.2)],
    )
    later = tmp_path / "serie_b" / "2026" / "2026_02_01.csv"
    _write_results_csv(
        later,
        [
            _strengths_row("Team B", 0.3, 0.4, eta=0.21, beta_home=0.31, rho=-0.04),
            _strengths_row("Team C", -0.1, 0.05, eta=0.21, beta_home=0.31, rho=-0.04),
        ],
    )

    params = _load_params(str(tmp_path))

    # eta/beta_home/rho/reference_date come from the single globally-latest
    # file (serie_b's), but `teams` is the union across every competition's
    # OWN latest file -- Team A (serie_a's only, older, file) must still be
    # present, not silently dropped just because serie_a's own latest round
    # predates serie_b's. This is the real, currently-occurring scenario
    # (Serie A paused for a World Cup break while Serie B kept playing) that
    # motivated this design -- see _latest_results_csv_by_competition.
    assert params == {
        "reference_date": "2026-02-01",
        "eta": 0.21,
        "beta_home": 0.31,
        "rho": -0.04,
        "teams": {
            "Team A": {"attack": 0.1, "defense": 0.2},
            "Team B": {"attack": 0.3, "defense": 0.4},
            "Team C": {"attack": -0.1, "defense": 0.05},
        },
    }


# ---------------------------------------------------------------------------
# _upcoming_cards
# ---------------------------------------------------------------------------


def _card_rows(*, scheduled_dates=(), postponed_teams=(), played_teams=()):
    rows = []
    for i, date in enumerate(scheduled_dates):
        rows.append(
            {
                "competition": "Serie A",
                "season": 2026,
                "match_datetime": date,
                "home_team": f"Home {i}",
                "away_team": f"Away {i}",
                "home_goals": None,
                "away_goals": None,
                "status": "scheduled",
            }
        )
    for i, (home, away) in enumerate(postponed_teams):
        rows.append(
            {
                "competition": "Serie A",
                "season": 2026,
                "match_datetime": "2026-03-01 16:00",
                "home_team": home,
                "away_team": away,
                "home_goals": None,
                "away_goals": None,
                "status": "postponed",
            }
        )
    for i, (home, away) in enumerate(played_teams):
        rows.append(
            {
                "competition": "Serie A",
                "season": 2026,
                "match_datetime": "2026-01-01 16:00",
                "home_team": home,
                "away_team": away,
                "home_goals": 1,
                "away_goals": 0,
                "status": "played",
            }
        )
    return rows


def _crests_colors(teams):
    crest_by_team = {t: f"assets/crests/{t.lower().replace(' ', '_')}.png" for t in teams}
    color_by_team = {t: "#111111" for t in teams}
    return crest_by_team, color_by_team


def test_upcoming_cards_windowed_scheduled_rows_sorted_soonest_first():
    rows = _card_rows(scheduled_dates=["2026-07-15 16:00", "2026-07-10 16:00", "2026-07-20 16:00"])
    df = _matches_df(rows)
    teams = {f"Home {i}" for i in range(3)} | {f"Away {i}" for i in range(3)}
    crest_by_team, color_by_team = _crests_colors(teams)

    cards = _upcoming_cards(df, "Serie A", 2026, NOW, crest_by_team, color_by_team, teams)

    assert [c["home_team"] for c in cards] == ["Home 1", "Home 0", "Home 2"]
    assert all(c["status"] == "scheduled" for c in cards)
    assert all(c["date"] is not None for c in cards)


def test_upcoming_cards_includes_every_remaining_scheduled_row_regardless_of_distance():
    # 20 scheduled matches, every one far in the future -- no window/cap any
    # more (the site paginates client-side instead), so all 20 must come
    # back, still soonest-first.
    far_dates = [(NOW + pd.Timedelta(days=30 + i)).strftime("%Y-%m-%d %H:%M") for i in range(20)]
    rows = _card_rows(scheduled_dates=far_dates)
    df = _matches_df(rows)
    teams = {f"Home {i}" for i in range(20)} | {f"Away {i}" for i in range(20)}
    crest_by_team, color_by_team = _crests_colors(teams)

    cards = _upcoming_cards(df, "Serie A", 2026, NOW, crest_by_team, color_by_team, teams)

    assert len(cards) == 20
    assert [c["home_team"] for c in cards] == [f"Home {i}" for i in range(20)]


def test_upcoming_cards_excludes_scheduled_row_dated_before_now():
    rows = _card_rows(scheduled_dates=["2026-01-01 16:00", "2026-07-15 16:00"])
    df = _matches_df(rows)
    teams = {f"Home {i}" for i in range(2)} | {f"Away {i}" for i in range(2)}
    crest_by_team, color_by_team = _crests_colors(teams)

    cards = _upcoming_cards(df, "Serie A", 2026, NOW, crest_by_team, color_by_team, teams)

    assert [c["home_team"] for c in cards] == ["Home 1"]


def test_upcoming_cards_postponed_rows_always_included_alphabetically_after_dated():
    rows = _card_rows(
        scheduled_dates=["2026-07-15 16:00"],
        postponed_teams=[("Zeta", "Alpha"), ("Beta", "Gamma")],
    )
    df = _matches_df(rows)
    teams = {"Home 0", "Away 0", "Zeta", "Alpha", "Beta", "Gamma"}
    crest_by_team, color_by_team = _crests_colors(teams)

    cards = _upcoming_cards(df, "Serie A", 2026, NOW, crest_by_team, color_by_team, teams)

    assert [c["home_team"] for c in cards] == ["Home 0", "Beta", "Zeta"]
    postponed_cards = [c for c in cards if c["status"] == "postponed"]
    assert all(c["date"] is None for c in postponed_cards)


def test_upcoming_cards_excludes_played_rows():
    rows = _card_rows(
        scheduled_dates=["2026-07-15 16:00"], played_teams=[("Played Home", "Played Away")]
    )
    df = _matches_df(rows)
    teams = {"Home 0", "Away 0", "Played Home", "Played Away"}
    crest_by_team, color_by_team = _crests_colors(teams)

    cards = _upcoming_cards(df, "Serie A", 2026, NOW, crest_by_team, color_by_team, teams)

    assert len(cards) == 1
    assert cards[0]["home_team"] == "Home 0"


def test_upcoming_cards_skips_row_with_team_missing_from_known_teams(capsys):
    rows = _card_rows(scheduled_dates=["2026-07-15 16:00"])
    df = _matches_df(rows)
    crest_by_team, color_by_team = _crests_colors({"Home 0", "Away 0"})
    known_teams = {"Home 0"}  # "Away 0" missing

    cards = _upcoming_cards(df, "Serie A", 2026, NOW, crest_by_team, color_by_team, known_teams)

    assert cards == []
    assert "Home 0 x Away 0" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _read_snapshot_params / _played_cards
# ---------------------------------------------------------------------------


def _played_rows(*, matches, competition="Serie A", season=2026):
    """matches: list of (home, away, home_goals, away_goals, match_datetime)."""
    return [
        {
            "competition": competition,
            "season": season,
            "match_datetime": dt,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "status": "played",
        }
        for home, away, hg, ag, dt in matches
    ]


def test_read_snapshot_params_reads_scalars_and_every_team(tmp_path):
    csv_path = tmp_path / "2026_05_01.csv"
    _write_results_csv(
        csv_path,
        [
            _strengths_row("Team A", 0.1, 0.2, eta=0.02, beta_home=0.3, rho=-0.01),
            _strengths_row("Team B", -0.1, 0.05, eta=0.02, beta_home=0.3, rho=-0.01),
        ],
    )

    params = _read_snapshot_params(str(csv_path))

    assert params == {
        "eta": 0.02,
        "beta_home": 0.3,
        "rho": -0.01,
        "teams": {
            "Team A": {"attack": 0.1, "defense": 0.2},
            "Team B": {"attack": -0.1, "defense": 0.05},
        },
    }


def test_played_cards_has_model_true_with_two_team_params_slice(tmp_path):
    results_dir = tmp_path / "results"
    _write_results_csv(
        results_dir / "serie_a" / "2026" / "2026_05_01.csv",
        [
            _strengths_row("Home 0", 0.1, 0.2, eta=0.02, beta_home=0.3, rho=-0.01),
            _strengths_row("Away 0", -0.1, 0.05, eta=0.02, beta_home=0.3, rho=-0.01),
            _strengths_row("Other Team", 0.5, 0.5, eta=0.02, beta_home=0.3, rho=-0.01),
        ],
    )
    rows = _played_rows(matches=[("Home 0", "Away 0", 2, 1, "2026-05-10 16:00")])
    df = _matches_df(rows)
    crest_by_team, color_by_team = _crests_colors({"Home 0", "Away 0"})

    cards = _played_cards(
        df, "Serie A", 2026, crest_by_team, color_by_team, "serie_a", str(results_dir)
    )

    assert len(cards) == 1
    card = cards[0]
    assert card["home_team"] == "Home 0" and card["away_team"] == "Away 0"
    assert card["home_goals"] == 2 and card["away_goals"] == 1
    assert card["has_model"] is True
    assert card["reference_date"] == "2026-05-01"
    # Only the two relevant teams travel with the card, not the full roster.
    assert card["params"] == {
        "eta": 0.02,
        "beta_home": 0.3,
        "rho": -0.01,
        "teams": {
            "Home 0": {"attack": 0.1, "defense": 0.2},
            "Away 0": {"attack": -0.1, "defense": 0.05},
        },
    }


def test_played_cards_has_model_false_when_no_prior_snapshot_exists(tmp_path):
    results_dir = tmp_path / "results"
    # Only snapshot AFTER the match -- no valid "prior" snapshot exists yet
    # (the real, confirmed edge case for a season's earliest played matches).
    _write_results_csv(
        results_dir / "serie_a" / "2026" / "2026_05_01.csv",
        [
            _strengths_row("Home 0", 0.1, 0.2),
            _strengths_row("Away 0", -0.1, 0.05),
        ],
    )
    rows = _played_rows(matches=[("Home 0", "Away 0", 2, 1, "2026-01-01 16:00")])
    df = _matches_df(rows)
    crest_by_team, color_by_team = _crests_colors({"Home 0", "Away 0"})

    cards = _played_cards(
        df, "Serie A", 2026, crest_by_team, color_by_team, "serie_a", str(results_dir)
    )

    assert len(cards) == 1
    assert cards[0]["has_model"] is False
    assert cards[0]["reference_date"] is None
    assert cards[0]["params"] is None


def test_played_cards_has_model_false_when_team_missing_from_snapshot_roster(tmp_path):
    results_dir = tmp_path / "results"
    _write_results_csv(
        results_dir / "serie_a" / "2026" / "2026_05_01.csv",
        [_strengths_row("Home 0", 0.1, 0.2)],  # "Away 0" never fit
    )
    rows = _played_rows(matches=[("Home 0", "Away 0", 2, 1, "2026-05-10 16:00")])
    df = _matches_df(rows)
    crest_by_team, color_by_team = _crests_colors({"Home 0", "Away 0"})

    cards = _played_cards(
        df, "Serie A", 2026, crest_by_team, color_by_team, "serie_a", str(results_dir)
    )

    assert cards[0]["has_model"] is False


def test_played_cards_mixed_season_some_with_and_without_a_snapshot(tmp_path):
    results_dir = tmp_path / "results"
    _write_results_csv(
        results_dir / "serie_a" / "2026" / "2026_05_01.csv",
        [_strengths_row("Home 0", 0.1, 0.2), _strengths_row("Away 0", -0.1, 0.05)],
    )
    rows = _played_rows(
        matches=[
            ("Home 0", "Away 0", 2, 1, "2026-01-01 16:00"),  # before the only snapshot
            ("Home 0", "Away 0", 1, 1, "2026-05-10 16:00"),  # after it
        ]
    )
    df = _matches_df(rows)
    crest_by_team, color_by_team = _crests_colors({"Home 0", "Away 0"})

    cards = _played_cards(
        df, "Serie A", 2026, crest_by_team, color_by_team, "serie_a", str(results_dir)
    )

    has_model_by_date = {c["date"][:10]: c["has_model"] for c in cards}
    assert has_model_by_date == {"2026-01-01": False, "2026-05-10": True}


def test_played_cards_most_recent_first():
    rows = _played_rows(
        matches=[
            ("Home 0", "Away 0", 1, 0, "2026-01-01 16:00"),
            ("Home 0", "Away 0", 2, 0, "2026-06-01 16:00"),
            ("Home 0", "Away 0", 0, 0, "2026-03-01 16:00"),
        ]
    )
    df = _matches_df(rows)
    crest_by_team, color_by_team = _crests_colors({"Home 0", "Away 0"})

    cards = _played_cards(
        df, "Serie A", 2026, crest_by_team, color_by_team, "serie_a", "/nonexistent"
    )

    assert [c["date"][:10] for c in cards] == ["2026-06-01", "2026-03-01", "2026-01-01"]


def test_played_cards_skips_row_with_missing_crest(capsys):
    rows = _played_rows(matches=[("Home 0", "Away 0", 1, 0, "2026-01-01 16:00")])
    df = _matches_df(rows)
    crest_by_team, color_by_team = _crests_colors({"Home 0"})  # "Away 0" missing

    cards = _played_cards(
        df, "Serie A", 2026, crest_by_team, color_by_team, "serie_a", "/nonexistent"
    )

    assert cards == []
    assert "Home 0 x Away 0" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# export_matches_data (integration)
# ---------------------------------------------------------------------------


def _setup_export(tmp_path, *, matches_rows, results_rows_by_path, club_infos_rows):
    matches_path = tmp_path / "matches.csv"
    _write_matches_csv(matches_path, matches_rows)

    results_dir = tmp_path / "results"
    for rel_path, rows in results_rows_by_path.items():
        _write_results_csv(results_dir / rel_path, rows)

    crest_src = _write_crest(tmp_path / "crests_src" / "team.png")
    club_infos_path = tmp_path / "club_infos.csv"
    pd.DataFrame(
        [
            {"club": club, "crest_path": crest_src, "primary_color": color}
            for club, color in club_infos_rows.items()
        ]
    ).to_csv(club_infos_path, index=False)

    site_dir = tmp_path / "site"
    return matches_path, results_dir, club_infos_path, site_dir


def test_export_matches_data_writes_manifest_matches_and_params(tmp_path):
    matches_rows = _card_rows(
        scheduled_dates=["2026-07-15 16:00", "2026-07-11 16:00"],
        postponed_teams=[("Zeta", "Alpha")],
    )
    teams = {"Home 0", "Away 0", "Home 1", "Away 1", "Zeta", "Alpha"}
    results_rows_by_path = {
        "serie_a/2026/2026_07_01.csv": [_strengths_row(t, 0.1, 0.2) for t in teams],
    }
    club_infos_rows = {t: "#123456" for t in teams}

    matches_path, results_dir, club_infos_path, site_dir = _setup_export(
        tmp_path,
        matches_rows=matches_rows,
        results_rows_by_path=results_rows_by_path,
        club_infos_rows=club_infos_rows,
    )

    export_matches_data(
        seasons=[2026],
        results_dir=str(results_dir),
        matches_path=str(matches_path),
        club_infos_path=str(club_infos_path),
        site_dir=str(site_dir),
        now=NOW,
    )

    manifest = json.loads((site_dir / "data" / "matches_manifest.json").read_text())
    assert manifest == {
        "competitions": [{"competition": "Serie A", "slug": "serie_a", "seasons": [2026]}]
    }

    matches_json = json.loads((site_dir / "data" / "serie_a" / "matches_2026.json").read_text())
    home_teams = [m["home_team"] for m in matches_json["matches"]]
    assert home_teams == ["Home 1", "Home 0", "Zeta"]  # soonest-first, then postponed
    assert all("home_win" not in m and "scores" not in m for m in matches_json["matches"])
    postponed = [m for m in matches_json["matches"] if m["status"] == "postponed"][0]
    assert postponed == {
        "home_team": "Zeta",
        "away_team": "Alpha",
        "home_crest": "assets/crests/team.png",
        "away_crest": "assets/crests/team.png",
        "home_color": "#123456",
        "away_color": "#123456",
        "date": None,
        "status": "postponed",
    }

    params = json.loads((site_dir / "data" / "params.json").read_text())
    assert params["reference_date"] == "2026-07-01"
    assert set(params["teams"]) == teams

    assert (site_dir / "assets" / "crests" / "team.png").read_bytes() == b"crest-bytes"


def test_export_matches_data_finished_season_produces_no_upcoming_file_or_manifest_entry(
    tmp_path,
):
    # All matches for this competition/season are "played" -- no upcoming
    # cards/manifest entry, but a played_2026.json + played_manifest.json
    # entry IS produced (played_teams' fixed match date, 2026-01-01, predates
    # the only results snapshot, 2026-07-01, so has_model is False here).
    matches_rows = _card_rows(played_teams=[("Home 0", "Away 0")])
    results_rows_by_path = {
        "serie_a/2026/2026_07_01.csv": [
            _strengths_row("Home 0", 0.1, 0.2),
            _strengths_row("Away 0", 0.1, 0.2),
        ],
    }
    club_infos_rows = {"Home 0": "#123456", "Away 0": "#654321"}

    matches_path, results_dir, club_infos_path, site_dir = _setup_export(
        tmp_path,
        matches_rows=matches_rows,
        results_rows_by_path=results_rows_by_path,
        club_infos_rows=club_infos_rows,
    )

    export_matches_data(
        seasons=[2026],
        results_dir=str(results_dir),
        matches_path=str(matches_path),
        club_infos_path=str(club_infos_path),
        site_dir=str(site_dir),
        now=NOW,
    )

    manifest = json.loads((site_dir / "data" / "matches_manifest.json").read_text())
    assert manifest == {"competitions": []}
    assert not (site_dir / "data" / "serie_a" / "matches_2026.json").exists()

    played_manifest = json.loads((site_dir / "data" / "played_manifest.json").read_text())
    assert played_manifest == {
        "competitions": [{"competition": "Serie A", "slug": "serie_a", "seasons": [2026]}]
    }
    played_json = json.loads((site_dir / "data" / "serie_a" / "played_2026.json").read_text())
    assert len(played_json["matches"]) == 1
    assert played_json["matches"][0]["has_model"] is False


def test_export_matches_data_writes_played_cards_with_embedded_params(tmp_path):
    matches_rows = _card_rows(played_teams=[("Home 0", "Away 0")])  # played on 2026-01-01
    results_rows_by_path = {
        # A snapshot BEFORE the match's own date (still filed under the
        # 2026 season directory -- _snapshot_csv_before looks under
        # results_dir/<slug>/<season>/, not by the filename's own year), so
        # has_model is True here.
        "serie_a/2026/2025_12_01.csv": [
            _strengths_row("Home 0", 0.1, 0.2),
            _strengths_row("Away 0", -0.1, 0.05),
        ],
    }
    club_infos_rows = {"Home 0": "#123456", "Away 0": "#654321"}

    matches_path, results_dir, club_infos_path, site_dir = _setup_export(
        tmp_path,
        matches_rows=matches_rows,
        results_rows_by_path=results_rows_by_path,
        club_infos_rows=club_infos_rows,
    )

    export_matches_data(
        seasons=[2026],
        results_dir=str(results_dir),
        matches_path=str(matches_path),
        club_infos_path=str(club_infos_path),
        site_dir=str(site_dir),
        now=NOW,
    )

    played_json = json.loads((site_dir / "data" / "serie_a" / "played_2026.json").read_text())
    card = played_json["matches"][0]
    assert card["home_goals"] == 1 and card["away_goals"] == 0
    assert card["has_model"] is True
    assert card["reference_date"] == "2025-12-01"
    assert card["params"]["teams"] == {
        "Home 0": {"attack": 0.1, "defense": 0.2},
        "Away 0": {"attack": -0.1, "defense": 0.05},
    }
