"""Monte Carlo simulation of a competition, defined as a sequence of phases.

A competition (see src/simulation/config.py and configs/README.md) is a list of
phases, each either:
  - `round_robin`: every team (or, with `groups`, every team within its own
    group) plays every other once at home and once away.
  - `playoff`: a bracket of pairs, seeded from an earlier phase, decided over
    one or two legs.

Score sampling is vectorized over (posterior draws x remaining fixtures) with
numpy, via rejection sampling: draw (x, y) from the two *independent*
Poissons (numpy's native, vectorized, C-level rng.poisson), and accept/reject
against the Dixon-Coles tau(x, y) correction (REC-equivalent to dc_log_prob
in poisson_home.stan), which only reweights the 4 cells x,y in {0,1}. This
was tried two ways:
  - Gathering only the still-"pending" (draw, fixture) cells each round via
    boolean/fancy indexing (`arr[rows, cols]`). Fewer values processed per
    round, but fancy-indexed gather/scatter on a scattered subset of a large
    2-D array is cache-hostile -- this was *slower* than the dense grid
    approach it replaced (measured: ~98s vs ~37s for 100k draws x ~200
    fixtures).
  - Recomputing the *whole* (n_draws, n_fixtures) array every round with
    np.where, leaving already-accepted cells alone instead of compacting them
    out. Wastes some resampling on already-accepted cells, but every
    operation stays a plain elementwise, contiguous-memory numpy op -- this
    is the one below, and it's ~5x faster than even the original dense
    (draws x fixtures x 11 x 11) grid + cumsum approach, a common way to
    vectorize this kind of score sampling, since it never materializes a
    per-score-pair grid at all. It also has no truncation (rng.poisson has no
    upper bound), unlike a fixed max-goals grid.
Acceptance per round is 1/bound, and bound is close to 1 whenever rho is (our
prior is ~N(0, 0.1)), so this converges in a handful of rounds in practice.

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

from src.simulation import fixtures
from src.simulation.config import (
    CompetitionConfig,
    PlayoffPhaseConfig,
    RoundRobinPhaseConfig,
    SlotRef,
    SpotConfig,
)
from src.simulation.standings import rank_table

DrawParams = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]


def simulate_scores(
    mu_home: np.ndarray,
    mu_away: np.ndarray,
    rho: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch Dixon-Coles-adjusted Poisson score sampling via rejection sampling.

    Args:
        mu_home, mu_away: shape (n_draws, n_matches).
        rho: shape (n_draws,).

    Returns:
        (home_goals, away_goals), each shape (n_draws, n_matches), int.
    """
    n_draws, n_matches = mu_home.shape
    rho = np.broadcast_to(rho[:, None], (n_draws, n_matches))

    tau00 = 1 - mu_home * mu_away * rho
    tau01 = 1 + mu_home * rho
    tau10 = 1 + mu_away * rho
    tau11 = 1 - rho
    bound = np.maximum.reduce([np.ones_like(mu_home), tau00, tau01, tau10, tau11])

    home_goals = np.zeros((n_draws, n_matches), dtype=np.int64)
    away_goals = np.zeros((n_draws, n_matches), dtype=np.int64)
    pending = np.ones((n_draws, n_matches), dtype=bool)

    while pending.any():
        x = rng.poisson(mu_home)
        y = rng.poisson(mu_away)

        tau = np.ones_like(mu_home)
        tau = np.where((x == 0) & (y == 0), np.maximum(tau00, 0), tau)
        tau = np.where((x == 0) & (y == 1), np.maximum(tau01, 0), tau)
        tau = np.where((x == 1) & (y == 0), np.maximum(tau10, 0), tau)
        tau = np.where((x == 1) & (y == 1), np.maximum(tau11, 0), tau)

        accept = pending & (rng.random((n_draws, n_matches)) < (tau / bound))
        home_goals = np.where(accept, x, home_goals)
        away_goals = np.where(accept, y, away_goals)
        pending &= ~accept

    return home_goals, away_goals


def _match_rates(attack, defense, eta, beta_home, home_idx, away_idx) -> tuple[np.ndarray, np.ndarray]:
    """attack, defense: (n_draws, T). eta, beta_home: (n_draws,). home_idx, away_idx: (n_matches,)."""
    mu_home = np.exp(attack[:, home_idx] - defense[:, away_idx] + eta[:, None] + beta_home[:, None])
    mu_away = np.exp(attack[:, away_idx] - defense[:, home_idx] + eta[:, None])
    return mu_home, mu_away


def _match_rates_per_draw(attack, defense, eta, beta_home, home_idx, away_idx) -> tuple[np.ndarray, np.ndarray]:
    """Same as _match_rates, but home_idx/away_idx name a *different* single match per draw
    (shape (n_draws,)) instead of a shared batch of fixtures -- used for playoffs, where
    who plays whom depends on that draw's own outcome so far.
    """
    row = np.arange(attack.shape[0])
    mu_home = np.exp(attack[row, home_idx] - defense[row, away_idx] + eta + beta_home)
    mu_away = np.exp(attack[row, away_idx] - defense[row, home_idx] + eta)
    return mu_home, mu_away


def _simulate_remaining_all_draws(
    remaining_fixtures, attack, defense, eta, beta_home, rho, team_index, rng
) -> tuple[np.ndarray, np.ndarray]:
    """Simulates every remaining fixture for every draw at once.

    Returns (home_goals, away_goals), each shape (n_draws, n_fixtures).
    """
    home_idx = np.array([team_index[home] for home, _ in remaining_fixtures])
    away_idx = np.array([team_index[away] for _, away in remaining_fixtures])
    mu_home, mu_away = _match_rates(attack, defense, eta, beta_home, home_idx, away_idx)
    return simulate_scores(mu_home, mu_away, rho, rng)


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
    attack, defense, eta, beta_home, rho, team_index = draw_params
    n_draws = attack.shape[0]

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
    home_goals, away_goals = _simulate_remaining_all_draws(
        all_remaining, attack, defense, eta, beta_home, rho, team_index, rng
    )

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


def _resolve_manual_side(side, phase_results: dict[str, PhaseResult], team_index, n_draws) -> np.ndarray:
    if isinstance(side, SlotRef):
        source = phase_results[side.from_phase]
        if not isinstance(source, PlayoffResult):
            raise ValueError(f"manual pair references {side.from_phase!r}, which is not a playoff phase")
        return np.array([team_index[name] for name in source.winners[side.pair]])
    return np.full(n_draws, team_index[side])


def _simulate_playoff_pair(idx_a, idx_b, attack, defense, eta, beta_home, rho, teams, phase_cfg: PlayoffPhaseConfig, rng):
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

    mu_home1, mu_away1 = _match_rates_per_draw(attack, defense, eta, beta_home, home1, away1)
    g_home1, g_away1 = simulate_scores(mu_home1[:, None], mu_away1[:, None], rho, rng)
    g_home1, g_away1 = g_home1[:, 0], g_away1[:, 0]

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
    mu_home2, mu_away2 = _match_rates_per_draw(attack, defense, eta, beta_home, home2, away2)
    g_home2, g_away2 = simulate_scores(mu_home2[:, None], mu_away2[:, None], rho, rng)
    g_home2, g_away2 = g_home2[:, 0], g_away2[:, 0]

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
    attack, defense, eta, beta_home, rho, team_index = draw_params
    n_draws = attack.shape[0]

    if phase_cfg.pairing == "table_position":
        source = phase_results[phase_cfg.source_phase]
        if not isinstance(source, RoundRobinResult):
            raise ValueError(f"phase {phase_cfg.id!r}: pairing 'table_position' requires a round_robin source_phase")
        pair_sides = [
            (
                np.array([team_index[source.group_orders[d]["_all"][pos_a - 1]] for d in range(n_draws)]),
                np.array([team_index[source.group_orders[d]["_all"][pos_b - 1]] for d in range(n_draws)]),
            )
            for pos_a, pos_b in phase_cfg.pairs
        ]
    elif phase_cfg.pairing == "bracket_adjacent":
        source = phase_results[phase_cfg.source_phase]
        if not isinstance(source, PlayoffResult):
            raise ValueError(f"phase {phase_cfg.id!r}: pairing 'bracket_adjacent' requires a playoff source_phase")
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
        pair_index: _simulate_playoff_pair(idx_a, idx_b, attack, defense, eta, beta_home, rho, teams, phase_cfg, rng)
        for pair_index, (idx_a, idx_b) in enumerate(pair_sides)
    }
    return PlayoffResult(winners=winners)


def _resolve_cascade(
    order: list[str],
    cascade_spots: list[SpotConfig],
    guaranteed_slots: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Allocates cascade_spots' table-position slots, honoring externally guaranteed
    slots (e.g. a Copa do Brasil berth) that let a team skip straight to a better
    spot than its table position alone would earn -- see configs/README.md for the
    worked example this implements.

    A team occupies exactly one seat, the best (lowest rank in cascade_spots)
    among its table-position ("natural") spot and *all* of its guarantees --
    e.g. a team that's both this year's Libertadores champion and Copa do Brasil
    champion holds two separate libertadores_grupos guarantees, but still only
    fills one groups seat itself. Every one of its OTHER guarantees (unused ones,
    including duplicates of the tier it does occupy) becomes a *bonus* seat in
    its own tier, handed to the next team in `order` not yet credited anywhere
    (the "first team outside the spot"). Likewise, if a guarantee bumps a team
    out of its natural tier, that tier's now-vacant seat is backfilled the same
    way, from the next team past its normal window.

    All of this reduces to one mechanic: each tier fills `capacity + bonus -
    locked` seats by scanning `order` in position order and skipping teams
    already credited elsewhere, so vacancies and bonus seats both cascade
    downward through `order` until claimed.

    Args:
        order: this group's final table order, 1st place first.
        cascade_spots: the phase's cascade spots (see RoundRobinPhaseConfig.cascade),
            in priority order (best first).
        guaranteed_slots: {team: [spot_name, ...]}, one entry per guarantee the
            team holds (repeat a spot_name for multiple independent guarantees
            of the same tier). Entries for teams not in `order` are ignored.

    Returns:
        {spot_name: [credited team, ...]}.
    """
    rank = {spot.name: i for i, spot in enumerate(cascade_spots)}
    capacity = {spot.name: spot.positions[1] - spot.positions[0] + 1 for spot in cascade_spots}
    position_of = {team: i for i, team in enumerate(order)}
    worst_rank = len(cascade_spots)

    def natural_spot(team: str) -> str | None:
        position = position_of[team] + 1
        for spot in cascade_spots:
            if spot.positions[0] <= position <= spot.positions[1]:
                return spot.name
        return None

    credited: dict[str, str] = {}
    bonus: dict[str, int] = defaultdict(int)
    locked: dict[str, int] = defaultdict(int)
    for team, guarantees in guaranteed_slots.items():
        if team not in position_of:
            continue
        guarantees = [g for g in guarantees if g in rank]
        if not guarantees:
            continue

        # Every guarantee is an independent extra berth that must go *somewhere*
        # (bonus, added to its own tier's capacity below), regardless of whether
        # this team ends up being the one to claim it. The team itself occupies
        # exactly one physical seat (locked), the best of its natural spot and
        # all its guarantees -- any other guarantee, including a second one for
        # that same tier, is simply unclaimed by this team and cascades on.
        natural = natural_spot(team)
        best_spot = natural
        best_rank = rank[natural] if natural is not None else worst_rank
        for g in guarantees:
            if rank[g] < best_rank:
                best_rank, best_spot = rank[g], g

        credited[team] = best_spot
        locked[best_spot] += 1
        for g in guarantees:
            bonus[g] += 1

    result: dict[str, list[str]] = {}
    scan_index = 0
    for spot in cascade_spots:
        recipients = [team for team, s in credited.items() if s == spot.name]
        needed = capacity[spot.name] + bonus[spot.name] - locked[spot.name]
        filled = 0
        while filled < needed and scan_index < len(order):
            team = order[scan_index]
            scan_index += 1
            if team in credited:
                continue
            recipients.append(team)
            credited[team] = spot.name
            filled += 1
        result[spot.name] = recipients
    return result


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
            cascade_spots = [next(s for s in phase_cfg.spots if s.name == name) for name in phase_cfg.cascade]
            phase_position_sum: dict[str, int] = defaultdict(int)
            for d in range(n_draws):
                for order in result.group_orders[d].values():
                    for position, team in enumerate(order, start=1):
                        all_teams.add(team)
                        phase_position_sum[team] += position
                        for spot in phase_cfg.spots:
                            if spot.name in cascade_names:
                                continue
                            if spot.positions and spot.positions[0] <= position <= spot.positions[1]:
                                spot_counts[team][spot.name] += 1
                    if cascade_spots:
                        credited = _resolve_cascade(order, cascade_spots, guaranteed_slots)
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
                        r for group_results in result.group_all_results[d].values() for r in group_results
                    ]
                    pool_order = rank_table(pool_candidates, combined_results, rng, phase_cfg.head_to_head_mode)
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


def simulate_competition(
    config: CompetitionConfig,
    mcmc_fit: CmdStanMCMC,
    teams: list[str],
    matches_df: pd.DataFrame,
    season: int,
    reference_date: pd.Timestamp,
    n_draws: int = 200,
    seed: int = 0,
    guaranteed_slots: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Monte Carlo simulates every phase of `config` in order and reports the
    probability of each declared spot (see configs/README.md for the schema).

    Args:
        guaranteed_slots: {team: [spot_name, ...]} for teams with an externally
            guaranteed slot (e.g. a Copa do Brasil berth) that bypasses table
            position -- repeat a spot_name to give one team multiple independent
            guarantees of the same tier (e.g. Libertadores champion + Copa do
            Brasil champion). Only has an effect on phases whose `cascade` lists
            spot_name (see RoundRobinPhaseConfig.cascade / _resolve_cascade).
    """
    team_index = {team: i for i, team in enumerate(teams)}
    stan_vars = mcmc_fit.stan_variables()
    total_draws = stan_vars["eta"].shape[0]
    rng = np.random.default_rng(seed)
    # More simulation replicates than posterior draws just means resampling draws with
    # replacement -- each reused parameter vector still gets a fresh, independent match
    # outcome every time, so this is a normal posterior-predictive resample, not a shortcut.
    draw_indices = rng.choice(total_draws, size=n_draws, replace=n_draws > total_draws)

    draw_params: DrawParams = (
        stan_vars["attack"][draw_indices],
        stan_vars["defense"][draw_indices],
        stan_vars["eta"][draw_indices],
        stan_vars["beta_home"][draw_indices],
        stan_vars["rho"][draw_indices],
        team_index,
    )

    phase_results: dict[str, PhaseResult] = {}
    for phase_cfg in config.phases:
        if isinstance(phase_cfg, RoundRobinPhaseConfig):
            phase_results[phase_cfg.id] = _run_round_robin_phase(
                phase_cfg, config.name, season, reference_date, matches_df, draw_params, rng
            )
        else:
            phase_results[phase_cfg.id] = _run_playoff_phase(phase_cfg, phase_results, draw_params, teams, rng)

    return _tabulate(config, phase_results, n_draws, rng, guaranteed_slots)
