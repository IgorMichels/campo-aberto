"""Builds a league table from match results and applies the CBF tiebreak order.

Tiebreak order (REC Art. 15 for Serie A / Art. 12 for Serie B): wins, goal
difference, goals scored, head-to-head (only when exactly two clubs are tied),
then cards and a final draw -- see configs/*.yaml for what we approximate and
why, per competition.

Also home to resolve_cascade, which turns a rank_table order into per-spot
classification, honoring externally guaranteed slots (e.g. a Copa do Brasil
berth) -- shared by src.simulation.simulate (per-draw, inside the Monte
Carlo loop) and src.site.export_site_data (once, on the real table)."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.simulation.config import SpotConfig


@dataclass
class _TeamRecord:
    points: int = 0
    wins: int = 0
    played: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against


def _build_records(teams: list[str], results: list[tuple]) -> dict[str, _TeamRecord]:
    """results: iterable of (home, away, home_goals, away_goals).

    `teams` may be a subset of the teams appearing in `results` -- matches
    involving a team outside `teams` still update whichever side is tracked.
    This is what lets a cross-group spot (e.g. "best 8 third-placed teams")
    rerank a pool of teams using each one's own group-stage record, without
    the untracked group-mates they actually beat raising a KeyError.
    """
    records = {team: _TeamRecord() for team in teams}
    for home, away, home_goals, away_goals in results:
        home_rec = records.get(home)
        away_rec = records.get(away)
        if home_rec is not None:
            home_rec.played += 1
            home_rec.goals_for += home_goals
            home_rec.goals_against += away_goals
        if away_rec is not None:
            away_rec.played += 1
            away_rec.goals_for += away_goals
            away_rec.goals_against += home_goals

        if home_goals > away_goals:
            if home_rec is not None:
                home_rec.points += 3
                home_rec.wins += 1
        elif home_goals < away_goals:
            if away_rec is not None:
                away_rec.points += 3
                away_rec.wins += 1
        else:
            if home_rec is not None:
                home_rec.points += 1
            if away_rec is not None:
                away_rec.points += 1
    return records


def _head_to_head(team_a: str, team_b: str, results: list[tuple]) -> tuple[int, int]:
    """(points, goal_diff) earned by team_a across its two matches vs team_b."""
    points = 0
    goal_diff = 0
    for home, away, home_goals, away_goals in results:
        if home == team_a and away == team_b:
            goal_diff += home_goals - away_goals
            points += 3 if home_goals > away_goals else (1 if home_goals == away_goals else 0)
        elif home == team_b and away == team_a:
            goal_diff += away_goals - home_goals
            points += 3 if away_goals > home_goals else (1 if away_goals == home_goals else 0)
    return points, goal_diff


def team_records(teams: list[str], results: list[tuple]) -> dict[str, dict]:
    """Plain per-team aggregates (points, wins, played, goals_for, goals_against,
    goal_diff) from already-played results -- e.g. a real standings snapshot as of
    some reference date. No tiebreak/ordering involved (see rank_table for that);
    this is just the raw numbers a standings table displays alongside it."""
    records = _build_records(teams, results)
    return {
        team: {
            "points": rec.points,
            "wins": rec.wins,
            "played": rec.played,
            "goals_for": rec.goals_for,
            "goals_against": rec.goals_against,
            "goal_diff": rec.goal_diff,
        }
        for team, rec in records.items()
    }


def rank_table(
    teams: list[str],
    results: list[tuple],
    rng: np.random.Generator,
    head_to_head_mode: str = "points_then_goal_diff",
) -> list[str]:
    """Returns team names ordered by final classification (1st place first)."""
    records = _build_records(teams, results)

    groups: dict[tuple, list[str]] = {}
    for team in teams:
        rec = records[team]
        key = (rec.points, rec.wins, rec.goal_diff, rec.goals_for)
        groups.setdefault(key, []).append(team)

    final_order = []
    for key in sorted(groups, reverse=True):
        tied = groups[key]
        if len(tied) == 1:
            final_order.extend(tied)
        elif len(tied) == 2:
            team_a, team_b = tied
            points_a, gd_a = _head_to_head(team_a, team_b, results)
            points_b, gd_b = _head_to_head(team_b, team_a, results)
            if head_to_head_mode == "points_then_goal_diff" and points_a != points_b:
                final_order.extend([team_a, team_b] if points_a > points_b else [team_b, team_a])
            elif gd_a != gd_b:
                final_order.extend([team_a, team_b] if gd_a > gd_b else [team_b, team_a])
            else:
                shuffled = [team_a, team_b]
                rng.shuffle(shuffled)
                final_order.extend(shuffled)
        else:
            shuffled = list(tied)
            rng.shuffle(shuffled)
            final_order.extend(shuffled)

    return final_order


def resolve_cascade(
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
