"""Unit tests for src/models/hyperparameter_sweep.py's control flow --
coordinate_sweep's baseline-once/best-per-param-selection/single-confirm-call
logic, and evaluate()'s own sweep_results.csv dedup -- without ever fitting
real Stan (mirrors tests/models/test_backtest.py's own no-Stan convention).

coordinate_sweep is tested against a FAKE evaluate() with a small synthetic
Brier surface (monkeypatching the module-level `evaluate` name that
coordinate_sweep calls), so no walk_forward_backtest/Stan machinery is
exercised there at all. evaluate() itself is tested separately, with only
walk_forward_backtest monkeypatched (aggregate_metrics and the CSV writer run
for real -- they're pure Python, no Stan).
"""

import pandas as pd

import src.models.hyperparameter_sweep as sweep


# --- coordinate_sweep (fake evaluate, synthetic Brier surface) ---


def _make_fake_evaluate(call_log: list[dict]):
    """Synthetic surface: Brier improves as "a" moves toward 1 and "b" moves
    toward 20, independently -- lets the test assert the coordinate-wise
    winner without any real backtest."""

    def fake_evaluate(model, params, **backtest_kwargs):
        call_log.append(dict(params))
        brier = 0.6 + abs(params.get("a", 2) - 1) * 0.01 + abs(params.get("b", 10) - 20) * 0.001
        return {
            "hash": "fake",
            "model": model,
            "n": 100,
            "brier": brier,
            "brier_uniform_baseline": 0.667,
            "brier_climatology_baseline": 0.65,
            "direction_accuracy": 0.5,
            **params,
        }

    return fake_evaluate


def test_coordinate_sweep_computes_baseline_exactly_once(monkeypatch):
    call_log: list[dict] = []
    monkeypatch.setattr(sweep, "evaluate", _make_fake_evaluate(call_log))

    param_grid = {"a": [1, 2, 3], "b": [10, 20]}
    defaults = {"a": 2, "b": 10}

    sweep.coordinate_sweep("fake_model", param_grid, defaults)

    # defaults={"a": 2, "b": 10} is exactly the baseline point; it must be
    # evaluated exactly once, even though "a"=2 and "b"=10 both also appear
    # as (skipped) candidates in their own parameter's grid.
    assert call_log.count({"a": 2, "b": 10}) == 1


def test_coordinate_sweep_skips_candidates_equal_to_the_default(monkeypatch):
    call_log: list[dict] = []
    monkeypatch.setattr(sweep, "evaluate", _make_fake_evaluate(call_log))

    param_grid = {"a": [1, 2, 3], "b": [10, 20]}
    defaults = {"a": 2, "b": 10}

    sweep.coordinate_sweep("fake_model", param_grid, defaults)

    # Only non-default candidates are ever evaluated: a=1, a=3, b=20 (b=10 is
    # the default, skipped) -- plus the baseline and the final confirm call.
    assert call_log.count({"a": 1, "b": 10}) == 1
    assert call_log.count({"a": 3, "b": 10}) == 1
    assert call_log.count({"a": 2, "b": 20}) == 1


def test_coordinate_sweep_picks_the_best_value_per_parameter_independently(monkeypatch):
    call_log: list[dict] = []
    monkeypatch.setattr(sweep, "evaluate", _make_fake_evaluate(call_log))

    param_grid = {"a": [1, 2, 3], "b": [10, 20]}
    defaults = {"a": 2, "b": 10}

    result = sweep.coordinate_sweep("fake_model", param_grid, defaults)

    # a=1 and b=20 minimize the synthetic surface independently.
    assert result["best"] == {"a": 1, "b": 20}


def test_coordinate_sweep_confirms_the_combined_best_exactly_once(monkeypatch):
    call_log: list[dict] = []
    monkeypatch.setattr(sweep, "evaluate", _make_fake_evaluate(call_log))

    param_grid = {"a": [1, 2, 3], "b": [10, 20]}
    defaults = {"a": 2, "b": 10}

    result = sweep.coordinate_sweep("fake_model", param_grid, defaults)

    # {"a": 1, "b": 20} (every parameter's individually-best value combined)
    # is a genuinely new combination never evaluated during the per-parameter
    # sweeps (which only vary one parameter at a time) -- it must appear
    # exactly once, from the final confirm call.
    assert call_log.count({"a": 1, "b": 20}) == 1
    assert result["confirm"]["brier"] == 0.6
    assert call_log[-1] == {"a": 1, "b": 20}


def test_coordinate_sweep_per_param_rows_cover_every_candidate(monkeypatch):
    call_log: list[dict] = []
    monkeypatch.setattr(sweep, "evaluate", _make_fake_evaluate(call_log))

    param_grid = {"a": [1, 2, 3], "b": [10, 20]}
    defaults = {"a": 2, "b": 10}

    result = sweep.coordinate_sweep("fake_model", param_grid, defaults)

    assert len(result["per_param"]["a"]) == 3
    assert len(result["per_param"]["b"]) == 2
    # The skipped (default) candidate's row is the baseline itself.
    assert result["per_param"]["a"][1] is result["baseline"]
    assert result["per_param"]["b"][0] is result["baseline"]


# --- _neighbor_values (pure function) ---


def test_neighbor_values_returns_both_sides_for_a_middle_value():
    assert sweep._neighbor_values([12, 18, 25, 35, 52], 25) == [18, 35]


def test_neighbor_values_returns_only_the_right_side_for_the_first_value():
    assert sweep._neighbor_values([12, 18, 25, 35, 52], 12) == [18]


def test_neighbor_values_returns_only_the_left_side_for_the_last_value():
    assert sweep._neighbor_values([12, 18, 25, 35, 52], 52) == [35]


def test_neighbor_values_returns_the_other_element_for_a_2_value_grid():
    assert sweep._neighbor_values([1.0, 0.5], 0.5) == [1.0]


# --- neighborhood_check (fake evaluate, synthetic Brier surface with a real interaction) ---


def _make_interacting_fake_evaluate(call_log: list[dict]):
    """Brier table over a 4x4 grid, crafted so that:
    - a's own sweep (b held at default=1) picks a=3 (an edge value, whose
      only grid neighbor is 2, NOT the default 1).
    - b's own sweep (a held at default=1) picks b=3 (same shape).
    - coordinate_sweep's combined "best" is therefore {a: 3, b: 3}, a
      combination never evaluated during either independent sweep.
    - {a: 2, b: 3} -- a's neighbor combined with b's BEST (not default) --
      is a genuinely new combination only neighborhood_check would ever
      try, and it beats {a: 3, b: 3}: a real interaction effect the
      independent, defaults-anchored coordinate_sweep can't see.
    - {a: 3, b: 2} -- b's neighbor combined with a's best -- is also new,
      but does NOT improve on {a: 3, b: 3}, so it must not be picked.
    """
    table = {
        (1, 1): 0.600,  # baseline / defaults
        (0, 1): 0.620,
        (2, 1): 0.598,
        (3, 1): 0.590,  # a's own sweep winner (b at default)
        (1, 0): 0.610,
        (1, 2): 0.595,
        (1, 3): 0.592,  # b's own sweep winner (a at default)
        (3, 3): 0.593,  # coordinate_sweep's combined "best" -- confirm result
        (2, 3): 0.585,  # only reachable via neighborhood_check -- the real winner
        (3, 2): 0.596,  # only reachable via neighborhood_check -- not an improvement
    }

    def fake_evaluate(model, params, **backtest_kwargs):
        call_log.append(dict(params))
        a, b = params.get("a", 1), params.get("b", 1)
        return {
            "hash": f"fake-{a}-{b}",
            "model": model,
            "n": 100,
            "brier": table[(a, b)],
            "brier_uniform_baseline": 0.667,
            "brier_climatology_baseline": 0.65,
            "direction_accuracy": 0.5,
            **params,
        }

    return fake_evaluate


def _run_coordinate_sweep_then_neighborhood_check(monkeypatch):
    call_log: list[dict] = []
    monkeypatch.setattr(sweep, "evaluate", _make_interacting_fake_evaluate(call_log))

    param_grid = {"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]}
    defaults = {"a": 1, "b": 1}

    result = sweep.coordinate_sweep("fake_model", param_grid, defaults)
    assert result["best"] == {"a": 3, "b": 3}  # sanity check on the fixture itself

    call_log_before_neighborhood = list(call_log)
    neighborhood = sweep.neighborhood_check(
        "fake_model", param_grid, result["best"], result["confirm"]
    )
    return neighborhood, call_log, call_log_before_neighborhood


def test_neighborhood_check_holds_other_params_at_best_not_defaults(monkeypatch):
    _, call_log, call_log_before = _run_coordinate_sweep_then_neighborhood_check(monkeypatch)

    new_calls = call_log[len(call_log_before) :]
    # {a: 2, b: 3} and {a: 3, b: 2} both combine a non-default value of one
    # parameter with the BEST (not default) value of the other -- neither
    # was ever evaluated during coordinate_sweep's own defaults-anchored
    # per-parameter sweeps.
    assert {"a": 2, "b": 3} in new_calls
    assert {"a": 3, "b": 2} in new_calls
    assert {"a": 2, "b": 3} not in call_log_before
    assert {"a": 3, "b": 2} not in call_log_before


def test_neighborhood_check_finds_an_interaction_coordinate_sweep_missed(monkeypatch):
    neighborhood, _, _ = _run_coordinate_sweep_then_neighborhood_check(monkeypatch)

    assert neighborhood["best"] == {"a": 2, "b": 3}
    assert neighborhood["confirm"]["brier"] == 0.585


def test_neighborhood_check_does_not_pick_a_non_improving_neighbor(monkeypatch):
    neighborhood, _, _ = _run_coordinate_sweep_then_neighborhood_check(monkeypatch)

    # {a: 3, b: 2} (0.596) is worse than the original best (0.593) -- must
    # not be picked even though it's a legitimate neighbor trial.
    assert neighborhood["best"] != {"a": 3, "b": 2}


def test_neighborhood_check_keeps_the_original_best_when_no_neighbor_improves(monkeypatch):
    call_log: list[dict] = []
    monkeypatch.setattr(sweep, "evaluate", _make_fake_evaluate(call_log))

    # This surface is fully separable (no interaction) -- coordinate_sweep
    # already finds the true joint optimum, so no neighbor should beat it.
    param_grid = {"a": [1, 2, 3], "b": [10, 20]}
    defaults = {"a": 2, "b": 10}
    result = sweep.coordinate_sweep("fake_model", param_grid, defaults)

    neighborhood = sweep.neighborhood_check(
        "fake_model", param_grid, result["best"], result["confirm"]
    )

    assert neighborhood["best"] == result["best"]
    assert neighborhood["confirm"] is result["confirm"]


# --- evaluate (fake walk_forward_backtest, real aggregate_metrics + CSV) ---


def _fake_records(n=10, brier_seed=0):
    """Cheap synthetic per-match score records aggregate_metrics can process
    for real -- shaped exactly like src.models.backtest._score_match's
    output, just hand-built instead of Stan-derived."""
    records = []
    for i in range(n):
        home = 0.6 if i % 2 == brier_seed % 2 else 0.3
        draw = 0.25
        away = 1 - home - draw
        records.append(
            {
                "home_team": "A",
                "away_team": "B",
                "match_datetime": pd.Timestamp("2026-01-01"),
                "competition": "Serie A",
                "season": 2026,
                "actual_outcome": "home",
                "home": home,
                "draw": draw,
                "away": away,
            }
        )
    return records


def test_evaluate_appends_one_row_per_new_params(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "walk_forward_backtest", lambda **kwargs: _fake_records())
    results_path = str(tmp_path / "sweep_results.csv")

    sweep.evaluate("poisson_home", {"half_life_weeks": 25}, results_path=results_path)

    df = pd.read_csv(results_path)
    assert len(df) == 1
    assert df.iloc[0]["model"] == "poisson_home"


def test_evaluate_dedups_the_same_params_on_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "walk_forward_backtest", lambda **kwargs: _fake_records())
    results_path = str(tmp_path / "sweep_results.csv")

    sweep.evaluate("poisson_home", {"half_life_weeks": 25}, results_path=results_path)
    sweep.evaluate("poisson_home", {"half_life_weeks": 25}, results_path=results_path)

    df = pd.read_csv(results_path)
    assert len(df) == 1


def test_evaluate_appends_a_new_row_for_different_params(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "walk_forward_backtest", lambda **kwargs: _fake_records())
    results_path = str(tmp_path / "sweep_results.csv")

    sweep.evaluate("poisson_home", {"half_life_weeks": 25}, results_path=results_path)
    sweep.evaluate("poisson_home", {"half_life_weeks": 12}, results_path=results_path)

    df = pd.read_csv(results_path)
    assert len(df) == 2
    assert set(df["half_life_weeks"]) == {25, 12}


def test_evaluate_fills_in_defaults_for_params_not_swept(tmp_path, monkeypatch):
    """A poisson_home params dict need not mention
    group_prior_mean/group_prior_sd -- evaluate must still record today's
    production default for them."""
    captured = {}

    def fake_walk_forward_backtest(**kwargs):
        captured.update(kwargs)
        return _fake_records()

    monkeypatch.setattr(sweep, "walk_forward_backtest", fake_walk_forward_backtest)
    results_path = str(tmp_path / "sweep_results.csv")

    result = sweep.evaluate("poisson_home", {"rho_prior_sd": 0.2}, results_path=results_path)

    assert captured["group_prior_mean"] == (0.3, 0.1, -0.1, -0.3)
    assert captured["group_prior_sd"] == 1.0
    assert result["group_prior_mean"] == (0.3, 0.1, -0.1, -0.3)


def test_evaluate_returns_the_aggregated_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "walk_forward_backtest", lambda **kwargs: _fake_records())
    results_path = str(tmp_path / "sweep_results.csv")

    result = sweep.evaluate("poisson_home", {"half_life_weeks": 25}, results_path=results_path)

    assert result["n"] == 10
    assert 0.0 <= result["brier"] <= 3.0
    assert "brier_uniform_baseline" in result
    assert "brier_climatology_baseline" in result
    assert "direction_accuracy" in result


def test_evaluate_passes_report_progress_as_the_on_season_done_callback(tmp_path, monkeypatch):
    captured = {}

    def fake_walk_forward_backtest(**kwargs):
        captured.update(kwargs)
        return _fake_records()

    monkeypatch.setattr(sweep, "walk_forward_backtest", fake_walk_forward_backtest)
    results_path = str(tmp_path / "sweep_results.csv")

    sweep.evaluate("poisson_home", {"half_life_weeks": 25}, results_path=results_path)

    # Live per-season progress reporting (feedback_backtest_monitor_style) is
    # wired in by default, not opt-in -- every evaluate() call gets it.
    assert captured["on_season_done"] is sweep._report_progress


# --- _report_progress (apples-to-apples same-season comparison, per the user's 2026-07-17 request) ---


def _fake_scored_record(season, actual="home", home=0.6, draw=0.25):
    return {
        "home_team": "A",
        "away_team": "B",
        "match_datetime": pd.Timestamp(f"{season}-06-01"),
        "competition": "Serie A",
        "season": season,
        "actual_outcome": actual,
        "home": home,
        "draw": draw,
        "away": 1 - home - draw,
    }


def _make_fake_original_tournament_records():
    return {
        "hierarchical_home": [
            _fake_scored_record(2022),
            _fake_scored_record(2023),
        ],
        "negbin_home": [
            _fake_scored_record(2022),
        ],
    }


def test_report_progress_prints_cumulative_brier_and_apples_to_apples_comparison(
    capsys, monkeypatch
):
    monkeypatch.setattr(
        sweep, "_load_original_tournament_records", _make_fake_original_tournament_records
    )
    monkeypatch.setattr(sweep, "_load_best_model_records", lambda: [])

    records = [_fake_scored_record(2022), _fake_scored_record(2022)]
    sweep._report_progress("poisson_home", 2022, records)

    out = capsys.readouterr().out
    assert "[poisson_home] cumulative through season 2022 (seasons 2022)" in out
    assert "apples-to-apples" in out
    assert "poisson_home (this run)" in out
    assert "<== this run" in out
    assert "hierarchical_home" in out
    assert "negbin_home" in out


def test_report_progress_restricts_other_models_to_the_same_seasons_only(capsys, monkeypatch):
    monkeypatch.setattr(
        sweep, "_load_original_tournament_records", _make_fake_original_tournament_records
    )
    monkeypatch.setattr(sweep, "_load_best_model_records", lambda: [])

    # This run has only reached season 2022 -- hierarchical_home's fake data
    # includes a 2023 record too, but it must NOT be counted here (n=1, not
    # n=2), since that would no longer be the same seasons this run covers.
    records = [_fake_scored_record(2022)]
    sweep._report_progress("poisson_home", 2022, records)

    out = capsys.readouterr().out
    hierarchical_line = next(line for line in out.splitlines() if "hierarchical_home" in line)
    assert "n=1" in hierarchical_line


def test_report_progress_sorts_rows_by_brier_ascending(capsys, monkeypatch):
    monkeypatch.setattr(
        sweep,
        "_load_original_tournament_records",
        lambda: {
            "hierarchical_home": [
                _fake_scored_record(2022, home=0.9)
            ],  # low Brier (confident+right)
            "negbin_home": [_fake_scored_record(2022, home=0.1)],  # high Brier (confident+wrong)
        },
    )
    monkeypatch.setattr(sweep, "_load_best_model_records", lambda: [])

    records = [_fake_scored_record(2022, home=0.5)]
    sweep._report_progress("poisson_home", 2022, records)

    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "hierarchical_home" in line or "negbin_home" in line
    ]
    assert lines[0].strip().startswith("hierarchical_home")
    assert lines[-1].strip().startswith("negbin_home")


def test_report_progress_is_silent_when_nothing_was_scored(capsys):
    # aggregate_metrics(records)["all"] is None for an empty records list --
    # must not raise (e.g. on a KeyError/format error from a None summary),
    # and must print nothing.
    sweep._report_progress("poisson_home", 2024, [])

    assert capsys.readouterr().out == ""


def test_load_original_tournament_records_reads_flat_cache_files_per_model(tmp_path, monkeypatch):
    model_dir = tmp_path / "poisson_home"
    model_dir.mkdir()
    pd.DataFrame([_fake_scored_record(2022)]).to_csv(model_dir / "2022_04_01.csv", index=False)

    monkeypatch.setattr(sweep, "BACKTEST_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(sweep, "MODEL_REGISTRY", {"poisson_home": object()})
    sweep._ORIGINAL_TOURNAMENT_RECORDS = None

    records_by_model = sweep._load_original_tournament_records()

    assert len(records_by_model["poisson_home"]) == 1
    assert records_by_model["poisson_home"][0]["season"] == 2022

    sweep._ORIGINAL_TOURNAMENT_RECORDS = None  # don't leak into other tests


def test_load_original_tournament_records_caches_across_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "BACKTEST_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(sweep, "MODEL_REGISTRY", {"poisson_home": object()})
    sweep._ORIGINAL_TOURNAMENT_RECORDS = None

    first = sweep._load_original_tournament_records()
    # A second call must return the SAME object (no re-read from disk) --
    # even though glob would now find nothing new to invalidate against.
    second = sweep._load_original_tournament_records()

    assert first is second

    sweep._ORIGINAL_TOURNAMENT_RECORDS = None  # don't leak into other tests
