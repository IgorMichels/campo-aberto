"""Unit tests for fit.py's pure-Python helpers (summarize_teams, samples_long,
reference_date, save_samples) -- everything except fit()/fit_stan_data(), which
actually compiles and samples poisson_home.stan and isn't exercised here.

A fake stand-in for CmdStanMCMC (only `.draws_pd()` is used by these helpers)
keeps these tests fast and independent of cmdstan being installed. Team/country
names are deliberately fictional, to prove there's nothing Brazil-specific in
this module -- see its docstring: "Generic over the matches CSV passed in --
pass any competition/country's data/processed/.../matches.csv".
"""

import pandas as pd
import pytest

from src.models.fit import reference_date, samples_long, save_samples, summarize_teams


class _FakeMCMC:
    def __init__(self, draws_df: pd.DataFrame):
        self._draws_df = draws_df

    def draws_pd(self) -> pd.DataFrame:
        return self._draws_df


def _fake_mcmc(values: dict[int, tuple[list[float], list[float]]]) -> _FakeMCMC:
    """values: {1-indexed stan team index: (attack draws, defense draws)}."""
    data = {}
    for i, (attack_draws, defense_draws) in values.items():
        data[f"attack[{i}]"] = attack_draws
        data[f"defense[{i}]"] = defense_draws
    return _FakeMCMC(pd.DataFrame(data))


def test_summarize_teams_maps_1_indexed_stan_columns_and_sorts_by_attack():
    teams = ["Alpha FC", "Beta FC"]
    mcmc = _fake_mcmc({1: ([0.1, 0.3], [0.0, 0.0]), 2: ([0.5, 0.7], [0.1, 0.1])})

    summary = summarize_teams(mcmc, teams).set_index("team")

    assert list(summary.index) == ["Beta FC", "Alpha FC"]  # higher mean attack first
    assert summary.loc["Alpha FC", "attack"] == pytest.approx(0.2)
    assert summary.loc["Beta FC", "attack"] == pytest.approx(0.6)
    assert summary.loc["Beta FC", "defense"] == pytest.approx(0.1)


def test_samples_long_is_one_row_per_team_per_draw():
    teams = ["Alpha FC", "Beta FC"]
    mcmc = _fake_mcmc({1: ([0.1, 0.3], [0.0, 0.0]), 2: ([0.5, 0.7], [0.1, 0.1])})

    long_df = samples_long(mcmc, teams)

    assert len(long_df) == 4  # 2 teams x 2 draws
    assert set(long_df["team"]) == set(teams)
    alpha = long_df[long_df["team"] == "Alpha FC"].sort_values("draw")
    assert list(alpha["attack"]) == pytest.approx([0.1, 0.3])
    assert list(alpha["draw"]) == [0, 1]


def test_reference_date_returns_the_latest_match_date(tmp_path):
    csv_path = tmp_path / "matches.csv"
    pd.DataFrame({"match_datetime": ["2026-01-01", "2026-03-15", "2026-02-01"]}).to_csv(
        csv_path, index=False
    )

    assert reference_date(str(csv_path)) == "2026_03_15"


def test_save_samples_infers_country_from_the_matches_path_parent_directory(tmp_path):
    """matches_path following the data/processed/<country>/matches.csv convention
    -- here a fictional "atlantis" -- should land in samples_dir/atlantis/, not
    anything Brazil-specific."""
    country_dir = tmp_path / "processed" / "atlantis"
    country_dir.mkdir(parents=True)
    matches_path = country_dir / "matches.csv"
    pd.DataFrame({"match_datetime": ["2026-05-01"]}).to_csv(matches_path, index=False)

    teams = ["Alpha FC", "Beta FC"]
    mcmc = _fake_mcmc({1: ([0.1], [0.0]), 2: ([0.2], [0.0])})
    samples_dir = tmp_path / "samples"

    out_path = save_samples(mcmc, teams, str(matches_path), samples_dir=str(samples_dir))

    assert out_path == str(samples_dir / "atlantis" / "2026_05_01.csv")
    saved = pd.read_csv(out_path)
    assert set(saved["team"]) == set(teams)
    assert len(saved) == 2  # 2 teams x 1 draw
