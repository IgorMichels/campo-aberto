# Competition configs

Each YAML file here defines one competition as a sequence of **phases**, each
either a `round_robin` (everyone plays everyone) or a `playoff` (a bracket of
pairs). Parsing and validation lives in
[`src/simulation/config.py`](../src/simulation/config.py); simulation of a
parsed config lives in
[`src/simulation/simulate.py`](../src/simulation/simulate.py)
(`simulate_competition`).

Only `serie_a.yaml` and `serie_b.yaml` are wired to real data today. The rest
of this document also describes fields those two configs don't use --
`groups`, `pool_position`, `bracket_adjacent`/`manual` pairing, `legs: 1` --
which exist so a Copa do Brasil / Libertadores / World-Cup-shaped config can
be dropped in later without changing the engine. See "Formats not yet
configured" below for worked (illustrative, not validated end-to-end)
sketches of each.

## Top level

```yaml
name: Serie A          # must match the `competition` column in matches.csv
n_teams: 20             # total clubs in the competition; used to sanity-check spot positions
phases: [ ... ]         # required, at least one
aggregates: [ ... ]     # optional, see "Aggregates" below
```

## Phases

Every phase has `id` (unique within the competition) and `type`
(`round_robin` or `playoff`).

### `round_robin`

```yaml
- id: league
  type: round_robin
  head_to_head_mode: points_then_goal_diff  # or goal_diff_only
  groups: [ ... ]        # optional, see "Groups" below
  spots: [ ... ]
```

`head_to_head_mode` controls the tiebreak used when exactly two clubs are
tied on points/wins/goal-difference/goals-scored (see
[`standings.py`](../src/simulation/standings.py)):
- `points_then_goal_diff`: points earned across both head-to-head legs, then
  their combined goal difference (CBF Serie A rule).
- `goal_diff_only`: combined goal difference of the two legs only, points
  aren't consulted (CBF Serie B rule).

Everything past a two-way tie (three-plus-way ties, and the last-resort
sorteio) is a random draw -- there's no disciplinary/card data to break it
more finely, and CBF's own last resort is a real drawing of lots, so this is
a correct stand-in, not an approximation.

#### Groups

Without `groups`, all of the competition's teams (for the given
`season`/`reference_date`, derived from the matches data) play in a single
round-robin table. With `groups`, each group runs its own independent
round-robin (its own turno/returno), and any `positions`-based spot on that
phase is evaluated **per group** rather than on a single combined table --
e.g. `positions: {from: 1, to: 2}` means "1st and 2nd of *each* group", which
is exactly what a World Cup or Libertadores group stage needs.

```yaml
groups:
  - [Team A, Team B, Team C, Team D]
  - [Team E, Team F, Team G, Team H]
```

A group entry can also be a reference to another phase's winner instead of a
literal team name (see "Slot references" below) -- **this specific
combination (a `round_robin` group containing a slot reference) is not yet
implemented**. `simulate.py` raises `NotImplementedError` with the reason if
you try: resolving it would make group membership vary per Monte Carlo draw,
which conflicts with the shared-fixture-list vectorization the round-robin
simulator relies on for performance (see the module docstring in
`simulate.py`). It's real Libertadores behavior (a preliminary-round winner
fills a fixed group slot), just not implemented since no current config
exercises it -- if you add one, that's the place to start.

### `playoff`

```yaml
- id: acesso
  type: playoff
  source_phase: league     # id of an earlier phase this one seeds from
  pairing: table_position  # table_position | bracket_adjacent | manual
  pairs: [[3, 6], [4, 5]]  # meaning depends on `pairing`, see below
  legs: 2                  # 1 or 2
  leg_order: worse_seed_home_first  # or better_seed_home_first
  tiebreak: points_then_goal_diff   # or goal_diff_only
  spots: [ ... ]
```

- `pairing: table_position` -- `source_phase` must be a `round_robin` phase.
  `pairs` is a list of `[position_a, position_b]` (1-indexed, from that
  phase's table). This is the Serie B "cruzamento olimpico".
- `pairing: bracket_adjacent` -- `source_phase` must be a `playoff` phase.
  Pairs are derived automatically: winner of that phase's pair 0 vs winner of
  its pair 1, pair 2 vs pair 3, and so on. `pairs` must not be set. This is
  how you chain knockout rounds (round of 16 -> quarterfinals -> ...).
- `pairing: manual` -- `pairs` is a list of `[team_a, team_b]`, where each
  side is either a literal team name or a slot reference (see below). Use
  this for anything decided by an external draw you can't derive
  automatically (e.g. Libertadores' post-group knockout draw, which avoids
  same-country pairings and isn't just "adjacent bracket slots"). Unlike the
  other two pairing modes, `manual` doesn't need `source_phase` at all --
  each side already says which phase (if any) it comes from -- so it's the
  only pairing that can start a competition with no earlier `round_robin`
  phase (e.g. Copa do Brasil's first round, seeded purely by literal team
  names).

`legs: 2` (default) simulates two legs: the team named first in the pair
("the better seed" -- table position 1 beats table position 2 in
`table_position` pairing, or bracket order in the others) hosts leg 2, the
decisive one; `leg_order` picks who hosts leg 1. Aggregate points (3/1/0 per
leg) decide the winner; if points also tie, aggregate goal difference; if
*that* also ties, the better seed advances (`tiebreak: points_then_goal_diff`)
or goal difference alone decides (`tiebreak: goal_diff_only`) -- both are real
rules, not a random fallback (see Serie B's `acesso` phase for the CBF Art. 13
version).

`legs: 1` simulates a single match (e.g. a World-Cup-style single-match
final). A drawn single match falls back to a coin flip, since there's no
extra-time/penalty-shootout model here -- the same kind of statistically
faithful stand-in as the round-robin's random tiebreak.

#### Slot references

A team slot can reference another phase's winner instead of a literal name:

```yaml
pairs:
  - [{from_phase: prelim_1, pair: 0}, Team C]
```

This resolves per Monte Carlo draw (a different draw may have a different
team win `prelim_1`'s pair 0), which is exactly how `_match_rates_per_draw`
already works in `simulate.py` -- so slot references inside `manual` playoff
pairs are fully implemented. The same construct inside a `round_robin`
phase's `groups` is not (see "Groups" above).

### Spots

A spot is a named outcome; the simulation reports what share of Monte Carlo
draws each team ends up in it, as a `prob_<name>` column. Exactly one of
three resolution modes is set per spot:

```yaml
- name: title
  positions: {from: 1, to: 1}      # round_robin only; per-group if grouped

- name: playoff_promotion
  result: winner                    # playoff only; credits every pair's winner

- name: qualified_best_third
  pool_position: 3                  # round_robin only, requires `groups`:
  top: 8                            #   take the Nth-place team of every group,
                                     #   re-rank that pool with the phase's own
                                     #   tiebreak rules, keep the best `top`
                                     #   (e.g. "best 8 third-placed teams")
```

### Aggregates

A derived spot that's just the sum of other spots' probabilities, e.g. total
promotion probability regardless of route:

```yaml
aggregates:
  - name: promotion
    of: [direct_promotion, playoff_promotion]
```

## Formats not yet configured

These aren't wired to real data or fixtures, but sketch how the fields above
compose for shapes beyond Serie A/B. Treat them as a starting point, not a
tested config.

**Copa do Brasil** -- no `round_robin` phase at all, just a chain of
two-legged (single-legged for the final) `playoff` phases, seeded by an
external draw (`pairing: manual`) since there's no round-robin table or prior
bracket to derive pairs from for the very first round:

```yaml
phases:
  - id: round_1
    type: playoff
    pairing: manual
    pairs: [[Team A, Team B], [Team C, Team D], ...]
    legs: 2
    spots: []
  - id: round_2
    type: playoff
    source_phase: round_1
    pairing: bracket_adjacent
    legs: 2
    spots: []
  # ... more bracket_adjacent rounds ...
  - id: final
    type: playoff
    source_phase: semifinal
    pairing: bracket_adjacent
    legs: 1
    spots:
      - name: champion
        result: winner
```

This composes with what's already implemented: `simulate_competition` draws
posterior parameters once per call and hands them to every phase directly
(not just to a leading `round_robin`), and `manual` pairing doesn't require
`source_phase`, so a competition made entirely of chained `playoff` phases
works with no `round_robin` phase at all.

**Libertadores** -- group stage as a `round_robin` with `groups`, each
group's top 2 feeding a `manual`-paired (external draw) knockout stage:

```yaml
phases:
  - id: groups
    type: round_robin
    head_to_head_mode: points_then_goal_diff
    groups: [[Team A, Team B, Team C, Team D], ...]  # 8 groups of 4
    spots:
      - name: qualified
        positions: {from: 1, to: 2}
  - id: round_of_16
    type: playoff
    source_phase: groups
    pairing: manual   # CONMEBOL's cross-group draw, not simple adjacency
    pairs: [[Team A, Team F], ...]
    legs: 2
    spots: []
  - id: quarterfinal
    type: playoff
    source_phase: round_of_16
    pairing: bracket_adjacent
    legs: 2
    spots: []
  # ... semifinal, single-leg final ...
```

**World Cup (48-team format)** -- group stage as a `round_robin` with
`groups`, using `pool_position` to pick the best 8 third-placed teams
alongside the direct group qualifiers:

```yaml
phases:
  - id: groups
    type: round_robin
    head_to_head_mode: points_then_goal_diff
    groups: [[Team A, Team B, Team C], ...]  # 12 groups of 4... err, of 3-4
    spots:
      - name: qualified_group
        positions: {from: 1, to: 2}
      - name: qualified_best_third
        pool_position: 3
        top: 8
  - id: round_of_32
    type: playoff
    source_phase: groups
    pairing: manual   # FIFA's seeding draw, not simple adjacency
    pairs: [ ... ]
    legs: 1
    spots: []
  # ... more bracket_adjacent rounds ...
```
