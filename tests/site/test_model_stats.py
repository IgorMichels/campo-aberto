"""Unit + integration tests for src.site.model_stats.

compute_scoreline_grid is checked against hand-derivable cases (a symmetric
matchup, home-advantage direction, the Dixon-Coles rho direction) rather than
a golden number from elsewhere, since it's a from-scratch Python port of
site/assets/js/poisson_home.js -- there's no other Python implementation to
diff against. load_played_records/aggregate_metrics use tiny on-disk fixture
JSON under tmp_path, mirroring tests/site/test_export_matches_data.py's
tmp_path-fixture style.
"""

import json

import pytest

from src.site.model_stats import (
    aggregate_metrics,
    compute_scoreline_grid,
    export_model_stats,
    load_played_records,
)

# --- compute_scoreline_grid ---


def _shared(eta=0.0, beta_home=0.0, rho=0.0):
    return {"eta": eta, "beta_home": beta_home, "rho": rho}


def _team(attack=0.0, defense=0.0):
    return {"attack": attack, "defense": defense}


def test_probabilities_sum_to_one():
    grid = compute_scoreline_grid(
        _shared(beta_home=0.3, rho=0.05), _team(0.4, -0.2), _team(-0.1, 0.2)
    )

    assert grid["home_win"] + grid["draw"] + grid["away_win"] == pytest.approx(1.0)


def test_symmetric_matchup_has_equal_home_and_away_win_and_best_is_zero_zero():
    """Zero attack/defense/eta/beta_home/rho on both sides -> mu_home ==
    mu_away == 1, a perfectly symmetric Poisson(1) vs Poisson(1) game: home
    and away win probabilities must be identical, and the argmax tie among
    (0,0)/(1,0)/(0,1)/(1,1) (all equal probability at mu=1, rho=0) resolves
    to (0,0) deterministically, since compute_scoreline_grid's argmax only
    overwrites on a STRICT improvement and (0,0) is scanned first."""
    grid = compute_scoreline_grid(_shared(), _team(), _team())

    assert grid["home_win"] == pytest.approx(grid["away_win"])
    assert grid["best"]["home"] == 0
    assert grid["best"]["away"] == 0


def test_home_advantage_shifts_win_probability_toward_the_home_team():
    grid = compute_scoreline_grid(_shared(beta_home=0.3), _team(), _team())

    assert grid["home_win"] > grid["away_win"]


def test_positive_rho_reduces_draw_probability_in_a_symmetric_matchup():
    """At mu_home == mu_away == 1: tau(0,0) = 1 - rho and tau(1,1) = 1 - rho
    (both draw cells shrink as rho grows past 0), while tau(1,0) = tau(0,1) =
    1 + rho (both non-draw cells grow) -- so a positive rho must strictly
    reduce the symmetric matchup's draw probability relative to rho=0."""
    no_correction = compute_scoreline_grid(_shared(rho=0.0), _team(), _team())
    with_correction = compute_scoreline_grid(_shared(rho=0.2), _team(), _team())

    assert with_correction["draw"] < no_correction["draw"]


def test_lopsided_matchup_predicts_more_home_goals_than_away_in_best_score():
    grid = compute_scoreline_grid(_shared(), _team(attack=1.5), _team(defense=-1.5))

    assert grid["best"]["home"] > grid["best"]["away"]


# --- load_played_records / aggregate_metrics (tmp_path fixture site tree) ---


def _write_played_season(site_dir, slug, season, matches):
    data_dir = site_dir / "data" / slug
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"played_{season}.json").write_text(json.dumps({"matches": matches}))


def _match(
    home_team="Team A",
    away_team="Team B",
    home_goals=1,
    away_goals=0,
    has_model=True,
    model="poisson_home",
    teams=None,
):
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "has_model": has_model,
        "params": {
            "model": model,
            "shared": {"eta": 0.0, "beta_home": 0.2, "rho": 0.0},
            "teams": teams if teams is not None else {home_team: _team(), away_team: _team()},
        },
    }


def _write_manifest(site_dir, competitions):
    (site_dir / "data").mkdir(parents=True, exist_ok=True)
    (site_dir / "data" / "played_manifest.json").write_text(
        json.dumps({"competitions": competitions})
    )


def test_load_played_records_skips_matches_without_a_model_snapshot(tmp_path):
    _write_manifest(tmp_path, [{"competition": "Serie A", "slug": "serie_a", "seasons": [2026]}])
    _write_played_season(
        tmp_path,
        "serie_a",
        2026,
        [_match(has_model=True), _match(has_model=False)],
    )

    records = load_played_records(str(tmp_path))

    assert len(records) == 1


def test_load_played_records_skips_a_team_missing_from_params(tmp_path):
    _write_manifest(tmp_path, [{"competition": "Serie A", "slug": "serie_a", "seasons": [2026]}])
    _write_played_season(
        tmp_path,
        "serie_a",
        2026,
        [_match(teams={"Team A": _team()})],  # away team missing
    )

    records = load_played_records(str(tmp_path))

    assert records == []


def test_load_played_records_raises_for_an_unimplemented_model(tmp_path):
    _write_manifest(tmp_path, [{"competition": "Serie A", "slug": "serie_a", "seasons": [2026]}])
    _write_played_season(tmp_path, "serie_a", 2026, [_match(model="hierarchical_home")])

    with pytest.raises(NotImplementedError):
        load_played_records(str(tmp_path))


def test_load_played_records_reports_the_real_outcome_and_exact_correctness(tmp_path):
    _write_manifest(tmp_path, [{"competition": "Serie A", "slug": "serie_a", "seasons": [2026]}])
    # eta=0, beta_home=0.2, rho=0, attack/defense=0 on both sides -> a
    # home-favored symmetric-strength matchup, real result 1-0 (a home win).
    _write_played_season(tmp_path, "serie_a", 2026, [_match(home_goals=1, away_goals=0)])

    [record] = load_played_records(str(tmp_path))

    assert record["actual_outcome"] == "home"
    assert record["competition"] == "Serie A"
    assert record["season"] == 2026
    assert record["home"] + record["draw"] + record["away"] == pytest.approx(1.0)


def test_aggregate_metrics_exact_pct(tmp_path):
    _write_manifest(tmp_path, [{"competition": "Serie A", "slug": "serie_a", "seasons": [2026]}])
    # beta_home=0.2 on both matches makes 1-0 the best score for a
    # symmetric-strength home-favored matchup (verified above) -- one match's
    # real result matches it (exact_correct=True), the other doesn't.
    _write_played_season(
        tmp_path,
        "serie_a",
        2026,
        [_match(home_goals=1, away_goals=0), _match(home_goals=4, away_goals=4)],
    )

    metrics = aggregate_metrics(load_played_records(str(tmp_path)))

    assert metrics["all"]["n"] == 2
    assert metrics["all"]["exact_pct"] == pytest.approx(0.5)


# --- export_model_stats (end to end) ---


def test_export_model_stats_writes_breakdown_and_calibration(tmp_path):
    _write_manifest(tmp_path, [{"competition": "Serie A", "slug": "serie_a", "seasons": [2026]}])
    _write_played_season(
        tmp_path,
        "serie_a",
        2026,
        [_match(home_goals=1, away_goals=0), _match(home_goals=0, away_goals=1)],
    )

    export_model_stats(str(tmp_path))

    payload = json.loads((tmp_path / "data" / "model_stats.json").read_text())
    assert payload["breakdown"]["all"]["n"] == 2
    assert len(payload["calibration"]) == 10
