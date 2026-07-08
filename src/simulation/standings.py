"""Builds a league table from match results and applies the CBF tiebreak order.

Tiebreak order (REC Art. 15 for Serie A / Art. 12 for Serie B): wins, goal
difference, goals scored, head-to-head (only when exactly two clubs are tied),
then cards and a final draw -- see configs/*.yaml for what we approximate and
why, per competition.
"""

from dataclasses import dataclass

import numpy as np


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
