"""Unit + integration tests for src.site.export_site_data.

Uses tiny fixture CSVs/crests under tmp_path for results_dir/club_infos/
matches_path/site_dir, but reuses the real configs/serie_*.yaml (via
load_configs_by_season, same as src.simulation.run_rounds) for
competition/season discovery -- the fixture results CSVs only need column
names covered by SPOT_LABELS, not every spot a given config actually
declares, since the exporter never looks at the config's spots directly.
"""

import json

import pandas as pd
import pytest

from src.simulation.config import AggregateConfig
from src.site.export_site_data import (
    _all_results_csvs,
    _build_columns,
    _competition_slug,
    _copy_crest,
    _export_season,
    _export_snapshot,
    export_site_data,
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


# Team A beats Team B 2-1 on 2024-12-15, then a 0-0 draw (B at home) on
# 2025-03-01 -- deliberately *not* on the same calendar day as either
# snapshot's reference_date below, so the standings split is unambiguous
# regardless of time-of-day (reference_date is midnight-normalized, matching
# src.simulation.fixtures.split_fixtures' own convention -- see that module).
TWO_MATCH_ROWS = [
    {
        "competition": "Serie A",
        "season": 2025,
        "match_datetime": "2024-12-15 16:00",
        "home_team": "Team A",
        "away_team": "Team B",
        "home_goals": 2,
        "away_goals": 1,
    },
    {
        "competition": "Serie A",
        "season": 2025,
        "match_datetime": "2025-03-01 16:00",
        "home_team": "Team B",
        "away_team": "Team A",
        "home_goals": 0,
        "away_goals": 0,
    },
]


def test_competition_slug_lowercases_and_replaces_spaces():
    assert _competition_slug("Serie A") == "serie_a"
    assert _competition_slug("Serie B") == "serie_b"


def test_all_results_csvs_returns_every_dated_file_oldest_first(tmp_path):
    season_dir = tmp_path / "serie_a" / "2025"
    for name in ["2025_03_01.csv", "2025_01_01.csv", "2025_02_01.csv"]:
        _write_results_csv(season_dir / name, [{"team": "A", "prob_title": 0.1}])

    csvs = _all_results_csvs("serie_a", 2025, str(tmp_path))

    assert csvs == [str(season_dir / f"2025_{m}_01.csv") for m in ("01", "02", "03")]


def test_all_results_csvs_returns_empty_list_when_missing(tmp_path):
    assert _all_results_csvs("serie_a", 2025, str(tmp_path)) == []


def test_copy_crest_writes_once_and_skips_identical_rewrite(tmp_path):
    src = _write_crest(tmp_path / "src" / "team.png")
    crests_dir = tmp_path / "crests"

    relative_path = _copy_crest(src, str(crests_dir))
    dest = crests_dir / "team.png"

    assert relative_path == "assets/crests/team.png"
    assert dest.read_bytes() == b"crest-bytes"

    dest.write_bytes(b"stale-marker")  # prove a second identical-content copy overwrites it
    _write_crest(tmp_path / "src" / "team.png")  # same content, same mtime-independent bytes
    _copy_crest(src, str(crests_dir))
    assert dest.read_bytes() == b"crest-bytes"


def test_export_snapshot_drops_expected_position_renames_attaches_crest_and_standings(tmp_path):
    csv_path = tmp_path / "2025_01_01.csv"
    _write_results_csv(
        csv_path,
        [
            {
                "team": "Team A",
                "expected_position": 1.5,
                "prob_title": 0.4321,
                "prob_rebaixamento": 0.0,
            },
            {
                "team": "Team B",
                "expected_position": 2.5,
                "prob_title": 0.1,
                "prob_rebaixamento": 0.2,
            },
        ],
    )
    crest_src = _write_crest(tmp_path / "crests_src" / "team_a.png")
    crest_by_team = {"Team A": crest_src, "Team B": crest_src}
    color_by_team = {"Team A": "#111111", "Team B": "#222222"}
    matches_df = _matches_df(TWO_MATCH_ROWS)

    date, columns, teams = _export_snapshot(
        str(csv_path),
        crest_by_team,
        color_by_team,
        str(tmp_path / "crests"),
        (),
        matches_df,
        "Serie A",
        2025,
    )

    assert date == "2025-01-01"
    assert columns == [
        {"key": "title", "label": "Título"},
        {"key": "rebaixamento", "label": "Rebaixamento"},
    ]
    assert teams[0] == {
        "team": "Team A",
        "crest": "assets/crests/team_a.png",
        "color": "#111111",
        "standings": {"points": 3, "played": 1, "goals_for": 2, "goals_against": 1, "goal_diff": 1},
        "probs": {"title": 0.4321, "rebaixamento": 0.0},
    }
    assert teams[1]["standings"] == {
        "points": 0,
        "played": 1,
        "goals_for": 1,
        "goals_against": 2,
        "goal_diff": -1,
    }


def test_export_snapshot_standings_only_count_matches_up_to_reference_date(tmp_path):
    # Same fixture, but dated *after* both matches -- the draw should count too.
    csv_path = tmp_path / "2025_06_01.csv"
    _write_results_csv(
        csv_path,
        [
            {"team": "Team A", "expected_position": 1.0, "prob_title": 0.5},
            {"team": "Team B", "expected_position": 2.0, "prob_title": 0.1},
        ],
    )
    matches_df = _matches_df(TWO_MATCH_ROWS)
    crest_src = _write_crest(tmp_path / "crests_src" / "team.png")

    _, _, teams = _export_snapshot(
        str(csv_path),
        {"Team A": crest_src, "Team B": crest_src},
        {"Team A": "#111111", "Team B": "#222222"},
        str(tmp_path / "crests"),
        (),
        matches_df,
        "Serie A",
        2025,
    )

    by_team = {t["team"]: t["standings"] for t in teams}
    assert by_team["Team A"] == {
        "points": 4,
        "played": 2,
        "goals_for": 2,
        "goals_against": 1,
        "goal_diff": 1,
    }
    assert by_team["Team B"] == {
        "points": 1,
        "played": 2,
        "goals_for": 1,
        "goals_against": 2,
        "goal_diff": -1,
    }


def test_export_snapshot_raises_on_spot_with_no_portuguese_label(tmp_path):
    csv_path = tmp_path / "2025_01_01.csv"
    _write_results_csv(
        csv_path, [{"team": "Team A", "expected_position": 1.0, "prob_unmapped_spot": 0.5}]
    )

    with pytest.raises(ValueError, match="unmapped_spot"):
        _export_snapshot(
            str(csv_path),
            {"Team A": "x.png"},
            {},
            str(tmp_path / "crests"),
            (),
            _matches_df(TWO_MATCH_ROWS),
            "Serie A",
            2025,
        )


def test_export_snapshot_raises_on_team_with_no_crest_path(tmp_path):
    csv_path = tmp_path / "2025_01_01.csv"
    _write_results_csv(csv_path, [{"team": "Team A", "expected_position": 1.0, "prob_title": 0.5}])

    with pytest.raises(ValueError, match="Team A"):
        _export_snapshot(
            str(csv_path),
            {},
            {},
            str(tmp_path / "crests"),
            (),
            _matches_df(TWO_MATCH_ROWS),
            "Serie A",
            2025,
        )


def test_build_columns_nests_an_aggregates_children_and_total_under_one_group():
    aggregates = (
        AggregateConfig(name="libertadores", of=("libertadores_grupos", "libertadores_pre")),
    )
    raw_names = ["title", "libertadores_grupos", "libertadores_pre", "sulamericana", "libertadores"]

    columns = _build_columns(raw_names, aggregates)

    assert columns == [
        {"key": "title", "label": "Título"},
        {
            "key": "libertadores",
            "label": "Libertadores",
            "children": [
                {"key": "libertadores_grupos", "label": "Fase de grupos"},
                {"key": "libertadores_pre", "label": "Pré-fase"},
                {"key": "libertadores", "label": "Geral"},
            ],
        },
        {"key": "sulamericana", "label": "Sul-Americana"},
    ]


def test_export_snapshot_raises_on_aggregate_with_no_group_label(tmp_path):
    csv_path = tmp_path / "2025_01_01.csv"
    _write_results_csv(
        csv_path,
        [
            {
                "team": "Team A",
                "expected_position": 1.0,
                "prob_title": 0.5,
                "prob_new_agg": 0.5,
                "prob_child": 0.5,
            }
        ],
    )
    aggregates = (AggregateConfig(name="new_agg", of=("child",)),)
    # "child" needs a SPOT_LABELS entry too, or it'd (correctly) fail on that first --
    # patch it in via a real one so this test isolates the group-label gap.
    from src.site import export_site_data as module

    module.SPOT_LABELS["child"] = "Filho"
    try:
        with pytest.raises(ValueError, match="new_agg"):
            _export_snapshot(
                str(csv_path),
                {"Team A": "x.png"},
                {},
                str(tmp_path / "crests"),
                aggregates,
                _matches_df(TWO_MATCH_ROWS),
                "Serie A",
                2025,
            )
    finally:
        del module.SPOT_LABELS["child"]


def test_export_season_collects_every_date_keyed_by_snapshot(tmp_path):
    season_dir = tmp_path / "results"
    csv_2025_01 = season_dir / "2025_01_01.csv"
    csv_2025_06 = season_dir / "2025_06_01.csv"
    _write_results_csv(
        csv_2025_01, [{"team": "Team A", "expected_position": 1.0, "prob_title": 0.5}]
    )
    _write_results_csv(
        csv_2025_06, [{"team": "Team A", "expected_position": 1.0, "prob_title": 0.6}]
    )
    crest_by_team = {"Team A": _write_crest(tmp_path / "crests_src" / "a.png")}
    color_by_team = {"Team A": "#111111"}

    data = _export_season(
        [str(csv_2025_01), str(csv_2025_06)],
        crest_by_team,
        color_by_team,
        str(tmp_path / "crests"),
        (),
        _matches_df(TWO_MATCH_ROWS),
        "Serie A",
        2025,
    )

    assert data["dates"] == ["2025-01-01", "2025-06-01"]
    assert set(data["snapshots"]) == {"2025-01-01", "2025-06-01"}
    assert data["snapshots"]["2025-01-01"]["teams"][0]["probs"]["title"] == 0.5
    assert data["snapshots"]["2025-06-01"]["teams"][0]["probs"]["title"] == 0.6
    assert data["columns"] == [{"key": "title", "label": "Título"}]


def test_export_site_data_writes_manifest_and_skips_missing_seasons(tmp_path):
    results_dir = tmp_path / "results"
    _write_results_csv(
        results_dir / "serie_a" / "2025" / "2025_06_01.csv",
        [{"team": "Team A", "expected_position": 1.0, "prob_title": 0.6}],
    )
    _write_results_csv(
        results_dir / "serie_b" / "2026" / "2026_06_01.csv",
        [{"team": "Team B", "expected_position": 2.0, "prob_rebaixamento": 0.3}],
    )
    # serie_a/2026 and serie_b/2025 are deliberately left without any results CSV.

    crest_src = _write_crest(tmp_path / "crests_src" / "team.png")
    club_infos = tmp_path / "club_infos.csv"
    pd.DataFrame(
        [
            {"club": "Team A", "crest_path": crest_src, "primary_color": "#111111"},
            {"club": "Team B", "crest_path": crest_src, "primary_color": "#222222"},
        ]
    ).to_csv(club_infos, index=False)
    matches_path = tmp_path / "matches.csv"
    _write_matches_csv(matches_path, TWO_MATCH_ROWS)

    site_dir = tmp_path / "site"
    export_site_data(
        seasons=[2025, 2026],
        results_dir=str(results_dir),
        club_infos_path=str(club_infos),
        matches_path=str(matches_path),
        site_dir=str(site_dir),
    )

    manifest = json.loads((site_dir / "data" / "manifest.json").read_text())
    assert manifest == {
        "competitions": [
            {"competition": "Serie A", "slug": "serie_a", "seasons": [2025]},
            {"competition": "Serie B", "slug": "serie_b", "seasons": [2026]},
        ]
    }

    serie_a_2025 = json.loads((site_dir / "data" / "serie_a" / "2025.json").read_text())
    assert serie_a_2025["dates"] == ["2025-06-01"]
    assert serie_a_2025["snapshots"]["2025-06-01"]["teams"][0]["team"] == "Team A"
    assert not (site_dir / "data" / "serie_a" / "2026.json").exists()
    assert (site_dir / "assets" / "crests" / "team.png").read_bytes() == b"crest-bytes"


def test_export_site_data_keeps_going_after_a_missing_crest(tmp_path, capsys):
    results_dir = tmp_path / "results"
    _write_results_csv(
        results_dir / "serie_a" / "2025" / "2025_06_01.csv",
        [{"team": "Team A", "expected_position": 1.0, "prob_title": 0.6}],
    )
    _write_results_csv(
        results_dir / "serie_b" / "2025" / "2025_06_01.csv",
        [{"team": "No Crest FC", "expected_position": 2.0, "prob_rebaixamento": 0.3}],
    )

    crest_src = _write_crest(tmp_path / "crests_src" / "team.png")
    club_infos = tmp_path / "club_infos.csv"
    # no "No Crest FC" row
    pd.DataFrame([{"club": "Team A", "crest_path": crest_src, "primary_color": "#111111"}]).to_csv(
        club_infos, index=False
    )
    matches_path = tmp_path / "matches.csv"
    _write_matches_csv(matches_path, TWO_MATCH_ROWS)

    site_dir = tmp_path / "site"
    export_site_data(
        seasons=[2025],
        results_dir=str(results_dir),
        club_infos_path=str(club_infos),
        matches_path=str(matches_path),
        site_dir=str(site_dir),
    )

    assert "No Crest FC" in capsys.readouterr().out
    manifest = json.loads((site_dir / "data" / "manifest.json").read_text())
    assert manifest == {
        "competitions": [{"competition": "Serie A", "slug": "serie_a", "seasons": [2025]}]
    }
    assert not (site_dir / "data" / "serie_b").exists()
