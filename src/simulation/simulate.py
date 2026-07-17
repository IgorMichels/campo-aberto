"""Monte Carlo simulation of a competition, defined as a sequence of phases.

A competition (see src/simulation/config.py and configs/README.md) is a list of
phases, each either:
  - `round_robin`: every team (or, with `groups`, every team within its own
    group) plays every other once at home and once away.
  - `playoff`: a bracket of pairs, seeded from an earlier phase, decided over
    one or two legs.

This module only orchestrates phases (round-robin fixtures, playoff pairs,
cascade-guaranteed slots, standings) from already-sampled scores -- it never
knows how a score was sampled. That's a src.models.adapter.ModelAdapter's
job (see src/models/registry.py for which one is active); this module only
ever calls `draw_params.adapter.sample_scores`/`.sample_scores_single`. See
src/models/adapters/poisson_home.py for the actual score-sampling
implementation (and its performance rationale) of today's production model.

Turning a completed round_robin phase into standings still needs a per-draw
pass (standings.rank_table) since the CBF head-to-head/random tiebreak isn't a
simple vectorizable reduction -- but that pass is cheap once it's just merging
already-simulated scores, not sampling them. The same is true of a
`pool_position` spot (e.g. "best 8 third-placed teams"): it's one more
rank_table call per draw, over the pooled candidates.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanMCMC

from src.models.adapter import ModelAdapter
from src.models.registry import DEFAULT_MODEL, MODEL_REGISTRY
from src.simulation import fixtures
from src.simulation.config import (
    CompetitionConfig,
    PlayoffPhaseConfig,
    RoundRobinPhaseConfig,
    SlotRef,
)
from src.simulation.standings import rank_table, resolve_cascade


@dataclass
class DrawParams:
    """Everything the round-robin/playoff/cascade orchestration below needs
    from a model's posterior, without knowing that model's own parameter
    names -- only `adapter` (src/models/adapter.py) ever interprets
    `team_params`/`shared_params`.
    """

    adapter: ModelAdapter
    team_params: dict[str, np.ndarray]  # name -> (n_draws, T)
    shared_params: dict[str, np.ndarray]  # name -> (n_draws,)
    team_index: dict[str, int]
    n_draws: int


def _simulate_remaining_all_draws(
    remaining_fixtures, draw_params: DrawParams, rng
) -> tuple[np.ndarray, np.ndarray]:
    """Simulates every remaining fixture for every draw at once.

    Returns (home_goals, away_goals), each shape (n_draws, n_fixtures).
    """
    # dtype=int64 matters even though team_index values are already ints:
    # np.array([]) with no explicit dtype defaults to float64, and an empty
    # remaining_fixtures (a round-robin phase already fully played as of
    # reference_date -- e.g. the season's final backtest checkpoint) would
    # otherwise produce a float index array that numpy's fancy indexing
    # rejects outright.
    team_index = draw_params.team_index
    home_idx = np.array([team_index[home] for home, _ in remaining_fixtures], dtype=np.int64)
    away_idx = np.array([team_index[away] for _, away in remaining_fixtures], dtype=np.int64)
    return draw_params.adapter.sample_scores(
        draw_params.team_params, draw_params.shared_params, home_idx, away_idx, rng
    )


@dataclass
class RoundRobinResult:
    """group_orders[d][group_id] is that group's finishing order (1st place first)
    for draw d. An ungrouped phase has a single group, "_all". group_all_results is
    only populated when a `pool_position` spot needs it (see _tabulate).
    """

    group_orders: list[dict[str, list[str]]]
    group_all_results: list[dict[str, list[tuple]]] | None


@dataclass
class PlayoffResult:
    """winners[i] is the winning team name per draw (shape (n_draws,)) of pair i,
    in the same order as the phase's resolved pairs -- `bracket_adjacent` phases
    consume this ordering directly (pair 2i vs pair 2i+1 of the previous round).
    """

    winners: dict[int, np.ndarray]


PhaseResult = RoundRobinResult | PlayoffResult


def _run_round_robin_phase(
    phase_cfg: RoundRobinPhaseConfig,
    competition: str,
    season: int,
    reference_date: pd.Timestamp,
    matches_df: pd.DataFrame,
    draw_params: DrawParams,
    rng: np.random.Generator,
) -> RoundRobinResult:
    n_draws = draw_params.n_draws

    if phase_cfg.legs != 2:
        raise NotImplementedError(
            f"phase {phase_cfg.id!r}: legs={phase_cfg.legs} (a single round-robin, e.g. a "
            f"World Cup-style group stage) isn't implemented -- fixtures.split_fixtures derives "
            f"the remaining-fixture list purely combinatorially from the team roster, which only "
            f"works for a double round-robin (every ordered (home, away) pair occurs exactly "
            f"once). A single round-robin has no such derivation: who hosts each pair is a real "
            f"scheduling/draw decision, so this needs an actual remaining-fixture source (e.g. a "
            f"schedule/draw file) before it can be simulated, not just a team list."
        )

    if phase_cfg.groups is None:
        groups = {"_all": fixtures.season_teams(matches_df, competition, season)}
    else:
        groups = {}
        for i, group in enumerate(phase_cfg.groups):
            team_names = []
            for entry in group:
                if isinstance(entry, SlotRef):
                    raise NotImplementedError(
                        f"phase {phase_cfg.id!r}: a round_robin group referencing another "
                        f"phase's winner ({entry}) isn't implemented -- resolving it would "
                        f"vary group membership per draw, which breaks the shared-fixture-list "
                        f"vectorization this module relies on (see the module docstring). No "
                        f"current competition config needs this; if one does, simulate that "
                        f"phase's fixtures with a per-draw loop instead of "
                        f"_simulate_remaining_all_draws."
                    )
                team_names.append(entry)
            groups[f"group_{i}"] = team_names

    played_by_group: dict[str, list[tuple]] = {}
    remaining_by_group: dict[str, list[tuple[str, str]]] = {}
    for group_id, group_teams in groups.items():
        played, remaining, _ = fixtures.split_fixtures(
            matches_df, competition, season, reference_date, teams=group_teams
        )
        played_by_group[group_id] = played
        remaining_by_group[group_id] = remaining

    all_remaining = [f for group_id in groups for f in remaining_by_group[group_id]]
    home_goals, away_goals = _simulate_remaining_all_draws(all_remaining, draw_params, rng)

    offsets = {}
    offset = 0
    for group_id in groups:
        n = len(remaining_by_group[group_id])
        offsets[group_id] = offset
        offset += n

    need_pool_stats = any(spot.pool_position is not None for spot in phase_cfg.spots)

    group_orders: list[dict[str, list[str]]] = []
    group_all_results: list[dict[str, list[tuple]]] | None = [] if need_pool_stats else None
    for d in range(n_draws):
        orders_d = {}
        results_d = {} if need_pool_stats else None
        for group_id, group_teams in groups.items():
            start = offsets[group_id]
            remaining = remaining_by_group[group_id]
            simulated = [
                (home, away, int(home_goals[d, start + k]), int(away_goals[d, start + k]))
                for k, (home, away) in enumerate(remaining)
            ]
            results = played_by_group[group_id] + simulated
            orders_d[group_id] = rank_table(group_teams, results, rng, phase_cfg.head_to_head_mode)
            if need_pool_stats:
                results_d[group_id] = results
        group_orders.append(orders_d)
        if need_pool_stats:
            group_all_results.append(results_d)

    return RoundRobinResult(group_orders=group_orders, group_all_results=group_all_results)


def _resolve_manual_side(
    side, phase_results: dict[str, PhaseResult], team_index, n_draws
) -> np.ndarray:
    if isinstance(side, SlotRef):
        source = phase_results[side.from_phase]
        if not isinstance(source, PlayoffResult):
            raise ValueError(
                f"manual pair references {side.from_phase!r}, which is not a playoff phase"
            )
        return np.array([team_index[name] for name in source.winners[side.pair]])
    return np.full(n_draws, team_index[side])


def _simulate_playoff_pair(
    idx_a, idx_b, draw_params: DrawParams, teams, phase_cfg: PlayoffPhaseConfig, rng
):
    """idx_a, idx_b: (n_draws,) team indices for the two seeds of one pair -- 'a' is
    the better-seeded team (e.g. higher table position, or the pair listed first in
    a bracket_adjacent/manual matchup), 'b' the other one. Returns winner team names,
    shape (n_draws,).

    `leg_order` decides who hosts leg 1; the other team hosts leg 2 (the only leg,
    when `legs == 1`).
    """
    if phase_cfg.leg_order == "worse_seed_home_first":
        home1, away1 = idx_b, idx_a
    else:
        home1, away1 = idx_a, idx_b

    g_home1, g_away1 = draw_params.adapter.sample_scores_single(
        draw_params.team_params, draw_params.shared_params, home1, away1, rng
    )

    if phase_cfg.legs == 1:
        if phase_cfg.leg_order == "worse_seed_home_first":
            a_goals, b_goals = g_away1, g_home1
        else:
            a_goals, b_goals = g_home1, g_away1
        # No extra-time/penalties model (no disciplinary/shootout data) -- a drawn
        # single match falls back to a coin flip, the same statistical stand-in used
        # elsewhere in this module for a last-resort sorteio.
        coin = rng.random(len(a_goals)) < 0.5
        better_wins = (a_goals > b_goals) | ((a_goals == b_goals) & coin)
        winner_idx = np.where(better_wins, idx_a, idx_b)
        return np.array(teams)[winner_idx]

    home2, away2 = away1, home1
    g_home2, g_away2 = draw_params.adapter.sample_scores_single(
        draw_params.team_params, draw_params.shared_params, home2, away2, rng
    )

    if phase_cfg.leg_order == "worse_seed_home_first":
        b_g1, a_g1 = g_home1, g_away1
        a_g2, b_g2 = g_home2, g_away2
    else:
        a_g1, b_g1 = g_home1, g_away1
        b_g2, a_g2 = g_home2, g_away2

    points_a = 3 * (a_g1 > b_g1) + (a_g1 == b_g1) + 3 * (a_g2 > b_g2) + (a_g2 == b_g2)
    points_b = 3 * (b_g1 > a_g1) + (b_g1 == a_g1) + 3 * (b_g2 > a_g2) + (b_g2 == a_g2)
    goal_diff_a = (a_g1 - b_g1) + (a_g2 - b_g2)

    if phase_cfg.tiebreak == "points_then_goal_diff":
        # A full tie (points and aggregate goal diff both equal) favors the better
        # seed, per e.g. REC Art. 13 par. 4 -- not a simplification, the real rule.
        better_wins = (points_a > points_b) | ((points_a == points_b) & (goal_diff_a >= 0))
    else:
        better_wins = goal_diff_a >= 0

    winner_idx = np.where(better_wins, idx_a, idx_b)
    return np.array(teams)[winner_idx]


def _run_playoff_phase(
    phase_cfg: PlayoffPhaseConfig,
    phase_results: dict[str, PhaseResult],
    draw_params: DrawParams,
    teams: list[str],
    rng: np.random.Generator,
) -> PlayoffResult:
    team_index = draw_params.team_index
    n_draws = draw_params.n_draws

    if phase_cfg.pairing == "table_position":
        source = phase_results[phase_cfg.source_phase]
        if not isinstance(source, RoundRobinResult):
            raise ValueError(
                f"phase {phase_cfg.id!r}: pairing 'table_position' requires a round_robin source_phase"
            )
        pair_sides = [
            (
                np.array(
                    [team_index[source.group_orders[d]["_all"][pos_a - 1]] for d in range(n_draws)]
                ),
                np.array(
                    [team_index[source.group_orders[d]["_all"][pos_b - 1]] for d in range(n_draws)]
                ),
            )
            for pos_a, pos_b in phase_cfg.pairs
        ]
    elif phase_cfg.pairing == "bracket_adjacent":
        source = phase_results[phase_cfg.source_phase]
        if not isinstance(source, PlayoffResult):
            raise ValueError(
                f"phase {phase_cfg.id!r}: pairing 'bracket_adjacent' requires a playoff source_phase"
            )
        n_source_pairs = len(source.winners)
        pair_sides = [
            (
                np.array([team_index[name] for name in source.winners[2 * i]]),
                np.array([team_index[name] for name in source.winners[2 * i + 1]]),
            )
            for i in range(n_source_pairs // 2)
        ]
    else:  # manual -- pairs are literal team names and/or slot references, no
        # source_phase needed: each side already names which phase it comes from.
        pair_sides = [
            (
                _resolve_manual_side(team_a, phase_results, team_index, n_draws),
                _resolve_manual_side(team_b, phase_results, team_index, n_draws),
            )
            for team_a, team_b in phase_cfg.pairs
        ]

    winners = {
        pair_index: _simulate_playoff_pair(idx_a, idx_b, draw_params, teams, phase_cfg, rng)
        for pair_index, (idx_a, idx_b) in enumerate(pair_sides)
    }
    return PlayoffResult(winners=winners)


def _tabulate(
    config: CompetitionConfig,
    phase_results: dict[str, PhaseResult],
    n_draws: int,
    rng: np.random.Generator,
    guaranteed_slots: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    guaranteed_slots = guaranteed_slots or {}
    spot_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    position_sum: dict[str, int] = {}
    all_teams: set[str] = set()

    for phase_cfg in config.phases:
        result = phase_results[phase_cfg.id]
        if isinstance(phase_cfg, RoundRobinPhaseConfig):
            assert isinstance(result, RoundRobinResult)
            cascade_names = set(phase_cfg.cascade)
            cascade_spots = [
                next(s for s in phase_cfg.spots if s.name == name) for name in phase_cfg.cascade
            ]
            phase_position_sum: dict[str, int] = defaultdict(int)
            for d in range(n_draws):
                for order in result.group_orders[d].values():
                    for position, team in enumerate(order, start=1):
                        all_teams.add(team)
                        phase_position_sum[team] += position
                        for spot in phase_cfg.spots:
                            if spot.name in cascade_names:
                                continue
                            if (
                                spot.positions
                                and spot.positions[0] <= position <= spot.positions[1]
                            ):
                                spot_counts[team][spot.name] += 1
                    if cascade_spots:
                        credited = resolve_cascade(order, cascade_spots, guaranteed_slots)
                        for spot_name, recipients in credited.items():
                            for team in recipients:
                                spot_counts[team][spot_name] += 1

                for spot in phase_cfg.spots:
                    if spot.pool_position is None:
                        continue
                    pool_candidates = [
                        order[spot.pool_position - 1]
                        for order in result.group_orders[d].values()
                        if len(order) >= spot.pool_position
                    ]
                    combined_results = [
                        r
                        for group_results in result.group_all_results[d].values()
                        for r in group_results
                    ]
                    pool_order = rank_table(
                        pool_candidates, combined_results, rng, phase_cfg.head_to_head_mode
                    )
                    for team in pool_order[: spot.top]:
                        spot_counts[team][spot.name] += 1
            position_sum = dict(phase_position_sum)  # the latest round_robin phase wins
        else:
            assert isinstance(result, PlayoffResult)
            for winner_names in result.winners.values():
                all_teams.update(winner_names)
                for spot in phase_cfg.spots:
                    if spot.result == "winner":
                        for name in winner_names:
                            spot_counts[name][spot.name] += 1

    all_spot_names = [spot.name for phase_cfg in config.phases for spot in phase_cfg.spots]
    rows = []
    for team in sorted(all_teams):
        row: dict[str, float] = {"team": team}
        if team in position_sum:
            row["expected_position"] = position_sum[team] / n_draws
        for spot_name in all_spot_names:
            row[f"prob_{spot_name}"] = spot_counts[team][spot_name] / n_draws
        for agg in config.aggregates:
            row[f"prob_{agg.name}"] = sum(row[f"prob_{s}"] for s in agg.of)
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_col = "expected_position" if "expected_position" in df.columns else df.columns[1]
    return df.sort_values(sort_col).reset_index(drop=True)


def _attach_team_strengths(
    df: pd.DataFrame,
    mcmc_fit: CmdStanMCMC,
    teams: list[str],
    adapter: ModelAdapter,
    team_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Posterior MEAN team/shared params -- one column per name in
    `adapter.team_param_names` (mapped by df["team"]) and
    `adapter.shared_param_names` (broadcast as the same scalar on every row)
    -- plus a `model` column naming `adapter.name`, so the result is
    self-describing. A deliberate simplification vs. the full-posterior
    resampling this module uses for season odds (draw_params inside
    simulate_competition is already a resampled *subset*; this reads the
    FULL posterior directly via mcmc_fit.stan_variables(), independent of
    n_draws/seed, so team strength doesn't jitter between runs). Consumed
    downstream by src.site.export_matches_data for the Confrontos page's
    params.json.

    `team_aliases` (optional, {alias_team: real_team}) is simulate_competition's
    debut/stale-data substitution (see that function's docstring): `teams` is
    always the REAL, Stan-fitted roster only (unlike simulate_competition's own
    internal team list, which is extended with alias entries) -- an alias name
    reuses its substitute's exact index here rather than getting its own new
    one, since this function only ever reads a team param *by* index, never
    the other way around (no `teams[i]` reverse lookup anywhere in this
    function), so aliasing the index is sufficient and correct."""
    stan_vars = mcmc_fit.stan_variables()
    index = {team: i for i, team in enumerate(teams)}
    for alias, real in (team_aliases or {}).items():
        index[alias] = index[real]
    df = df.copy()
    df["model"] = adapter.name
    for param in adapter.team_param_names:
        mean = stan_vars[param].mean(axis=0)
        df[param] = [float(mean[index[t]]) for t in df["team"]]
    for param in adapter.shared_param_names:
        df[param] = float(stan_vars[param].mean())
    return df


def simulate_competition(
    config: CompetitionConfig,
    mcmc_fit: CmdStanMCMC,
    teams: list[str],
    matches_df: pd.DataFrame,
    season: int,
    reference_date: pd.Timestamp | None = None,
    n_draws: int = 200,
    seed: int = 0,
    guaranteed_slots: dict[str, list[str]] | None = None,
    team_aliases: dict[str, str] | None = None,
    model: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Monte Carlo simulates every phase of `config` in order and reports the
    probability of each declared spot (see configs/README.md for the schema).

    Args:
        reference_date: the "as of" date phases are simulated from -- matches up
            to this date count as played (see fixtures.split_fixtures), and it
            gates any config-level `guaranteed_slots` entry (see
            GuaranteedSlotConfig). Defaults to matches_df's latest match_datetime.
        guaranteed_slots: {team: [spot_name, ...]} for teams with an externally
            guaranteed slot (e.g. a Copa do Brasil berth) that bypasses table
            position -- repeat a spot_name to give one team multiple independent
            guarantees of the same tier (e.g. Libertadores champion + Copa do
            Brasil champion). Merged with any date-gated entries declared in
            `config.guaranteed_slots`. Only has an effect on phases whose
            `cascade` lists spot_name (see RoundRobinPhaseConfig.cascade /
            standings.resolve_cascade).
        team_aliases: {debut_or_stale_team: previous_season_relegated_team} for
            a team `teams`/`mcmc_fit` has no posterior for at all (e.g. the
            pre-season backtest checkpoint src.simulation.run_rounds.reference_dates
            adds before round 1, when a newly-promoted or long-absent team has
            zero matches inside the training window) -- see
            src.simulation.run_rounds._relegated_teams_previous_season for how
            substitutes are picked. Each alias team gets its own, separate
            index carrying an exact COPY of its substitute's team params (not
            a shared index) -- this matters specifically for
            _simulate_playoff_pair's `np.array(teams)[winner_idx]`, which
            converts a winner's numeric index back into a name positionally:
            sharing an index would make an alias team's own playoff advancement
            get reported under its substitute's name instead of its own.
        model: which src.models.registry.MODEL_REGISTRY entry `mcmc_fit` was
            sampled from -- selects which Stan variable names are read and
            which score-sampling math runs (see src/models/adapter.py).
    """
    if reference_date is None:
        reference_date = matches_df["match_datetime"].max()

    guaranteed_slots = {team: list(spots) for team, spots in (guaranteed_slots or {}).items()}
    for entry in config.guaranteed_slots:
        if reference_date >= entry.known_from:
            guaranteed_slots.setdefault(entry.team, []).append(entry.spot)

    adapter = MODEL_REGISTRY[model]
    stan_vars = mcmc_fit.stan_variables()
    total_draws = next(iter(stan_vars.values())).shape[0]
    rng = np.random.default_rng(seed)
    # More simulation replicates than posterior draws just means resampling draws with
    # replacement -- each reused parameter vector still gets a fresh, independent match
    # outcome every time, so this is a normal posterior-predictive resample, not a shortcut.
    draw_indices = rng.choice(total_draws, size=n_draws, replace=n_draws > total_draws)

    team_params = {name: stan_vars[name][draw_indices] for name in adapter.team_param_names}
    shared_params = {name: stan_vars[name][draw_indices] for name in adapter.shared_param_names}
    sim_teams = teams
    if team_aliases:
        team_index = {team: i for i, team in enumerate(teams)}
        alias_names = list(team_aliases)
        substitute_cols = [team_index[team_aliases[name]] for name in alias_names]
        team_params = {
            name: np.concatenate([arr, arr[:, substitute_cols]], axis=1)
            for name, arr in team_params.items()
        }
        sim_teams = [*teams, *alias_names]

    team_index = {team: i for i, team in enumerate(sim_teams)}
    draw_params = DrawParams(
        adapter=adapter,
        team_params=team_params,
        shared_params=shared_params,
        team_index=team_index,
        n_draws=n_draws,
    )

    phase_results: dict[str, PhaseResult] = {}
    for phase_cfg in config.phases:
        if isinstance(phase_cfg, RoundRobinPhaseConfig):
            phase_results[phase_cfg.id] = _run_round_robin_phase(
                phase_cfg, config.name, season, reference_date, matches_df, draw_params, rng
            )
        else:
            phase_results[phase_cfg.id] = _run_playoff_phase(
                phase_cfg, phase_results, draw_params, sim_teams, rng
            )

    result = _tabulate(config, phase_results, n_draws, rng, guaranteed_slots)
    return _attach_team_strengths(result, mcmc_fit, teams, adapter, team_aliases)
