# Competition configs

Each YAML file here defines one competition as a sequence of **phases**, each
either a `round_robin` (everyone plays everyone) or a `playoff` (a bracket of
pairs). Parsing and validation lives in
[`src/simulation/config.py`](../src/simulation/config.py); simulation of a
parsed config lives in
[`src/simulation/simulate.py`](../src/simulation/simulate.py)
(`simulate_competition`).

Only `serie_a_*.yaml` and `serie_b_*.yaml` are wired to real data today. The
rest of this document also describes fields those configs don't use --
`groups`, `pool_position`, `bracket_adjacent`/`manual` pairing, `legs: 1` --
which exist so a Copa do Brasil / Libertadores / World-Cup-shaped config can
be dropped in later without changing the engine. See "Formats not yet
configured" below for worked (illustrative, not validated end-to-end)
sketches of each.

### Per-season configs

The REC (Regulamento Especifico da Competicao) changes year to year, so each
competition has one YAML **per season** it's wired up for, suffixed with the
year: `serie_a_2025.yaml` vs `serie_a_2026.yaml`, `serie_b_2025.yaml` vs
`serie_b_2026.yaml`. `name`/`n_teams`/`phases` are still matched against
matches.csv's `competition` column only (not `season` -- that's a separate
`--season` argument to `simulate_competition`/the CLIs), so picking the
right season's file for the season you're simulating/backtesting is on the
caller: `DEFAULT_CONFIGS` in `src/constants.py` only points at the current
season (2026); pass `--configs` explicitly for any other season, e.g.:

```bash
python -m src.simulation.run --reference-date 2025-11-01 --season 2025 \
    --configs configs/serie_a_2025.yaml configs/serie_b_2025.yaml
```

Known rule changes so far: Serie A 2025 had an extra Libertadores
preliminary-phase slot (2 instead of 2026's 1, pushing sulamericana's range
one position later); Serie B 2025 had no access playoff yet (all 4 promotion
spots awarded directly by table position, introduced as a playoff only in
2026's REC).

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
  legs: 2                # 1 or 2, default 2
  spots: [ ... ]
```

`legs: 2` (default) is a double round-robin -- every team plays every other
twice, once at each venue, e.g. Brasileirao Serie A/B. This is the only shape
[`fixtures.py`](../src/simulation/fixtures.py) supports today: it derives the
full remaining-fixture list purely combinatorially from the team roster
(every ordered `(home, away)` pair occurs exactly once), which only works
because a double round-robin leaves no scheduling choice to make. `legs: 1`
(a single round-robin, e.g. a World Cup-style group stage) has no such
derivation -- who hosts each pair is a real schedule/draw decision this
engine doesn't have data for yet -- so `simulate.py` raises
`NotImplementedError` rather than guessing a home/away split. Adding it needs
an actual remaining-fixture source (a schedule or draw file), not just a
team list.

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

### Cascade (externally guaranteed slots)

```yaml
- id: league
  type: round_robin
  spots: [ ... ]
  cascade: [libertadores_grupos, libertadores_pre, sulamericana]  # best first
```

`cascade` names a subset of this phase's `positions`-based spots, ranked best
first, that compete for table-position slots subject to slots guaranteed by
something outside this competition -- e.g. the Copa do Brasil champion and
runner-up each get a Libertadores berth (Art. 6 par. 1) independent of their
Serie A table position. Since we don't simulate the Copa do Brasil, its
finalists are supplied at call time instead, via `simulate_competition`'s
`guaranteed_slots: dict[team, list[spot_name]]` (or `--guaranteed-slot
TEAM:SPOT`, repeatable, on `src.simulation.run` / `src.pipeline`) -- repeat
the same team for multiple *independent* guarantees, e.g. a team that's both
this year's Libertadores champion (also worth a `libertadores_grupos` berth)
and Copa do Brasil champion.

A team occupies exactly one seat: the best (closest to the front of
`cascade`) among its own table position and *all* of its guarantees. Every
other guarantee it holds -- including a second one for the same tier it
already occupies -- goes unused and becomes a bonus seat in its own tier:
- An unused guarantee (table spot already as good or better, or a
  second/third guarantee for the tier the team already occupies) is handed
  to the next team in the table not yet credited anywhere (the "first team
  outside the spot").
- A guarantee better than the team's table spot is credited instead,
  vacating the table-position tier's seat -- which is backfilled the same
  way, from the next team past that tier's normal window.

Worked example (`cascade: [libertadores_grupos, libertadores_pre,
sulamericana]`, capacities 4/1/6 from `configs/serie_a_2026.yaml`):

- Team Z guaranteed `libertadores_grupos`, finishes 8th (a `sulamericana`
  position): Z is credited `libertadores_grupos` (5th recipient that draw,
  alongside the table's top 4), and `sulamericana` backfills its vacated 8th
  seat from 12th place, keeping 6 recipients.
- Team Y guaranteed `libertadores_pre`, finishes 3rd (a `libertadores_grupos`
  position): Y is credited `libertadores_grupos` (its table spot is already
  better), so its unused pré guarantee goes to 6th place instead (the first
  team outside pré's 5th-place window) -- crediting *two* `libertadores_pre`
  recipients that draw. 6th place's own vacated `sulamericana` seat then
  backfills from 12th, same as above.
- Team X is both this year's Libertadores champion and Copa do Brasil
  champion -- two independent `libertadores_grupos` guarantees -- and
  finishes 9th (a `sulamericana` position): both guarantees are extra berths,
  so `libertadores_grupos` needs 2 more seats than usual, filled by scanning
  onward from 1st place regardless of tier boundaries -- giving X *and*
  1st-5th place all `libertadores_grupos` (6 recipients, reaching one spot
  past the table's normal top 4). Since 5th place -- the table's natural
  `libertadores_pre` recipient -- got pulled into groups instead, `pre` falls
  to 6th place, and `sulamericana` keeps 7th/8th/10th/11th from its normal
  window plus 12th and 13th backfilling the two seats vacated by 6th (now
  pré) and 9th/X (now groups) -- still 6 recipients.

A guaranteed team not currently in the phase (e.g. simulating Serie B while
its guaranteed slot names a Serie A team) is simply ignored. `title` is
deliberately left out of Serie A's `cascade`: it's a bonus nested inside
`libertadores_grupos` (the champion is also a groups qualifier), not a tier
competing for seats.

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
