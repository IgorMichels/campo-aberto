"""Loads and validates a competition YAML config into a CompetitionConfig.

See configs/README.md for the full schema reference and worked examples
(Serie A, Serie B, and sketches for knockout/group formats like Copa do
Brasil, Libertadores and a World-Cup-style "best third place" pool).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

HEAD_TO_HEAD_MODES = {"points_then_goal_diff", "goal_diff_only"}
PAIRINGS = {"table_position", "bracket_adjacent", "manual"}
LEG_ORDERS = {"worse_seed_home_first", "better_seed_home_first"}


@dataclass(frozen=True)
class SlotRef:
    """A team slot only known per Monte Carlo draw: the winner of an earlier phase.

    Used inside `manual` playoff pairs (e.g. a bracket round whose participant is
    "whoever wins the other bracket") or, in principle, inside `groups` (e.g. a
    group slot filled by a preliminary-round winner) -- see the "not yet wired up"
    note on `PhaseConfig.groups` in simulate.py before relying on the latter.
    """

    from_phase: str
    pair: int


TeamOrSlot = Union[str, SlotRef]


@dataclass(frozen=True)
class SpotConfig:
    """A named outcome a team is credited with a share of Monte Carlo draws for.

    Exactly one resolution mode is set:
    - positions: a 1-indexed (from, to) range. In a round_robin phase this is
      evaluated per group -- with no `groups` that's just the global table
      position; with `groups`, each group's own local position.
    - result: "winner", only valid on a playoff phase -- credits every pair's
      winner.
    - pool_position + top: pulls the team finishing in `pool_position` from
      every group of a round_robin phase, re-ranks that pooled set with the
      phase's own tiebreak rules, and keeps the top `top` (e.g. "best 8
      third-placed teams" across World Cup groups).
    """

    name: str
    positions: tuple[int, int] | None = None
    result: str | None = None
    pool_position: int | None = None
    top: int | None = None


@dataclass(frozen=True)
class RoundRobinPhaseConfig:
    id: str
    head_to_head_mode: str
    spots: tuple[SpotConfig, ...]
    groups: tuple[tuple[TeamOrSlot, ...], ...] | None = None
    type: str = "round_robin"


@dataclass(frozen=True)
class PlayoffPhaseConfig:
    id: str
    pairing: str
    spots: tuple[SpotConfig, ...]
    source_phase: str | None = None
    pairs: tuple | None = None
    legs: int = 2
    leg_order: str = "worse_seed_home_first"
    tiebreak: str = "points_then_goal_diff"
    type: str = "playoff"


PhaseConfig = Union[RoundRobinPhaseConfig, PlayoffPhaseConfig]


@dataclass(frozen=True)
class AggregateConfig:
    name: str
    of: tuple[str, ...]


@dataclass(frozen=True)
class CompetitionConfig:
    name: str
    n_teams: int
    phases: tuple[PhaseConfig, ...]
    aggregates: tuple[AggregateConfig, ...] = ()

    def phase(self, phase_id: str) -> PhaseConfig:
        for p in self.phases:
            if p.id == phase_id:
                return p
        raise KeyError(f"unknown phase id {phase_id!r} in competition {self.name!r}")


def load_competition_config(path: str | Path) -> CompetitionConfig:
    """Parses and validates a competition YAML file into a CompetitionConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    return _parse_competition(raw, source=str(path))


def _parse_team_or_slot(value, context: str) -> TeamOrSlot:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "from_phase" in value and "pair" in value:
        return SlotRef(from_phase=value["from_phase"], pair=int(value["pair"]))
    raise ValueError(f"{context}: expected a team name or {{from_phase, pair}}, got {value!r}")


def _parse_spot(raw_spot: dict, phase_id: str) -> SpotConfig:
    name = raw_spot.get("name")
    if not name:
        raise ValueError(f"phase {phase_id!r}: spot missing 'name'")

    modes_present = [k for k in ("positions", "result", "pool_position") if k in raw_spot]
    if len(modes_present) != 1:
        raise ValueError(
            f"phase {phase_id!r} spot {name!r}: exactly one of 'positions', 'result', "
            f"'pool_position' must be set, got {modes_present}"
        )

    positions = None
    if "positions" in raw_spot:
        pos = raw_spot["positions"]
        positions = (int(pos["from"]), int(pos["to"]))
        if positions[0] > positions[1]:
            raise ValueError(f"phase {phase_id!r} spot {name!r}: positions.from > positions.to")

    result = None
    if "result" in raw_spot:
        result = raw_spot["result"]
        if result != "winner":
            raise ValueError(f"phase {phase_id!r} spot {name!r}: 'result' only supports 'winner'")

    pool_position = top = None
    if "pool_position" in raw_spot:
        pool_position = int(raw_spot["pool_position"])
        if "top" not in raw_spot:
            raise ValueError(f"phase {phase_id!r} spot {name!r}: 'pool_position' requires 'top'")
        top = int(raw_spot["top"])
        if top < 1:
            raise ValueError(f"phase {phase_id!r} spot {name!r}: 'top' must be >= 1")

    return SpotConfig(name=name, positions=positions, result=result, pool_position=pool_position, top=top)


def _parse_round_robin_phase(raw_phase: dict, phase_id: str, spots: tuple[SpotConfig, ...]) -> RoundRobinPhaseConfig:
    head_to_head_mode = raw_phase.get("head_to_head_mode")
    if head_to_head_mode not in HEAD_TO_HEAD_MODES:
        raise ValueError(f"phase {phase_id!r}: 'head_to_head_mode' must be one of {HEAD_TO_HEAD_MODES}")

    for spot in spots:
        if spot.result is not None:
            raise ValueError(f"phase {phase_id!r}: 'result' spots are only valid on playoff phases")

    groups = None
    if "groups" in raw_phase:
        groups = tuple(
            tuple(_parse_team_or_slot(t, f"phase {phase_id!r} group entry") for t in group)
            for group in raw_phase["groups"]
        )
        if len(groups) < 2:
            raise ValueError(f"phase {phase_id!r}: 'groups' must list at least 2 groups")

    return RoundRobinPhaseConfig(id=phase_id, head_to_head_mode=head_to_head_mode, spots=spots, groups=groups)


def _parse_playoff_phase(raw_phase: dict, phase_id: str, spots: tuple[SpotConfig, ...]) -> PlayoffPhaseConfig:
    pairing = raw_phase.get("pairing")
    if pairing not in PAIRINGS:
        raise ValueError(f"phase {phase_id!r}: 'pairing' must be one of {PAIRINGS}")

    source_phase = raw_phase.get("source_phase")
    if pairing in ("table_position", "bracket_adjacent") and not source_phase:
        raise ValueError(f"phase {phase_id!r}: pairing {pairing!r} requires 'source_phase'")

    pairs = None
    if pairing == "table_position":
        pairs = tuple((int(a), int(b)) for a, b in raw_phase.get("pairs", []))
        if not pairs:
            raise ValueError(f"phase {phase_id!r}: pairing 'table_position' requires non-empty 'pairs'")
    elif pairing == "manual":
        pairs = tuple(
            (
                _parse_team_or_slot(a, f"phase {phase_id!r} pair entry"),
                _parse_team_or_slot(b, f"phase {phase_id!r} pair entry"),
            )
            for a, b in raw_phase.get("pairs", [])
        )
        if not pairs:
            raise ValueError(f"phase {phase_id!r}: pairing 'manual' requires non-empty 'pairs'")
    elif "pairs" in raw_phase:
        raise ValueError(
            f"phase {phase_id!r}: pairing 'bracket_adjacent' derives pairs from source_phase "
            f"automatically, 'pairs' must not be set"
        )

    legs = int(raw_phase.get("legs", 2))
    if legs not in (1, 2):
        raise ValueError(f"phase {phase_id!r}: 'legs' must be 1 or 2")

    leg_order = raw_phase.get("leg_order", "worse_seed_home_first")
    if leg_order not in LEG_ORDERS:
        raise ValueError(f"phase {phase_id!r}: 'leg_order' must be one of {LEG_ORDERS}")

    tiebreak = raw_phase.get("tiebreak", "points_then_goal_diff")
    if tiebreak not in HEAD_TO_HEAD_MODES:
        raise ValueError(f"phase {phase_id!r}: 'tiebreak' must be one of {HEAD_TO_HEAD_MODES}")

    for spot in spots:
        if spot.result is None:
            raise ValueError(f"phase {phase_id!r}: playoff phase spots must use 'result: winner'")

    return PlayoffPhaseConfig(
        id=phase_id,
        source_phase=source_phase,
        pairing=pairing,
        spots=spots,
        pairs=pairs,
        legs=legs,
        leg_order=leg_order,
        tiebreak=tiebreak,
    )


def _parse_phase(raw_phase: dict) -> PhaseConfig:
    phase_id = raw_phase.get("id")
    if not phase_id:
        raise ValueError("phase missing 'id'")

    phase_type = raw_phase.get("type")
    spots = tuple(_parse_spot(s, phase_id) for s in raw_phase.get("spots", []))

    if phase_type == "round_robin":
        return _parse_round_robin_phase(raw_phase, phase_id, spots)
    if phase_type == "playoff":
        return _parse_playoff_phase(raw_phase, phase_id, spots)
    raise ValueError(f"phase {phase_id!r}: 'type' must be 'round_robin' or 'playoff', got {phase_type!r}")


def _parse_competition(raw: dict, source: str) -> CompetitionConfig:
    name = raw.get("name")
    if not name:
        raise ValueError(f"{source}: competition missing 'name'")

    n_teams = raw.get("n_teams")
    if not isinstance(n_teams, int) or n_teams <= 0:
        raise ValueError(f"{source}: 'n_teams' must be a positive integer")

    raw_phases = raw.get("phases")
    if not raw_phases:
        raise ValueError(f"{source}: competition must define at least one phase")
    phases = tuple(_parse_phase(p) for p in raw_phases)

    phase_ids = [p.id for p in phases]
    if len(set(phase_ids)) != len(phase_ids):
        raise ValueError(f"{source}: duplicate phase id in {phase_ids}")

    for i, p in enumerate(phases):
        if isinstance(p, PlayoffPhaseConfig) and p.source_phase is not None:
            if p.source_phase not in phase_ids:
                raise ValueError(f"{source}: phase {p.id!r} source_phase {p.source_phase!r} is not a defined phase")
            source_index = phase_ids.index(p.source_phase)
            if source_index >= i:
                raise ValueError(f"{source}: phase {p.id!r} source_phase {p.source_phase!r} must come before it")

    for p in phases:
        if isinstance(p, RoundRobinPhaseConfig) and p.groups is None:
            for spot in p.spots:
                if spot.positions and spot.positions[1] > n_teams:
                    raise ValueError(
                        f"{source}: phase {p.id!r} spot {spot.name!r}: "
                        f"positions.to={spot.positions[1]} exceeds n_teams={n_teams}"
                    )

    all_spot_names = {spot.name for p in phases for spot in p.spots}
    aggregates = tuple(AggregateConfig(name=a["name"], of=tuple(a["of"])) for a in raw.get("aggregates", []))
    for agg in aggregates:
        for spot_name in agg.of:
            if spot_name not in all_spot_names:
                raise ValueError(f"{source}: aggregate {agg.name!r} references unknown spot {spot_name!r}")

    return CompetitionConfig(name=name, n_teams=n_teams, phases=phases, aggregates=aggregates)
