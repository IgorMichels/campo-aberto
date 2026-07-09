# Treated dataset (Brazil)

This folder holds the output of the Brazil ingestion pipeline (`src/ingestion/brazil/`). It's built from two sources merged into one file:

```
CBF dockets     →  data/raw/brazil/cbf/{competition}_{year}.csv  ─┐
   (scrape)               raw, no name treatment                  ├─→  data/processed/brazil/matches.csv
ESPN scoreboard →  data/raw/brazil/espn/{competition}_{year}.csv ─┘         treated, mapped names, one row per match
   (schedule)             raw, no name treatment
```

Each country gets its own subfolder under `data/raw/` and `data/processed/` (mirroring `src/ingestion/`), so this layout is expected to repeat once other countries are added.

- **`matches.csv`** — one row per match, played or not: `competition, season, match_datetime, venue, home_team, away_team, home_goals, away_goals, status`. `status` is `played`, `scheduled`, or `postponed`; `home_goals`/`away_goals` are empty for any non-`played` row.
- **`team_name_mapping.csv`** — the raw → normalized club-name "de-para" table (2 columns: `raw_name, normalized_name`). Manually curated over time; the pipeline never overwrites it once it exists. Shared by both the CBF and ESPN raw layers.
- **`unmapped_team_names_log.csv`** — raw club names found in `data/raw/brazil/cbf/` (CBF dockets) or `data/raw/brazil/espn/` (ESPN scoreboard) with no entry in the mapping yet, with rapidfuzz suggestions. Regenerated fresh on every run (not appended).
- **`score_discrepancies.csv`** — matches where CBF and ESPN both report a final score and they disagree (`competition, season, home_team, away_team, cbf_home_goals, cbf_away_goals, espn_home_goals, espn_away_goals`). CBF's score is always the one kept in `matches.csv`, regardless of what's in this file -- it exists purely as a manual-review flag (a mismatch usually means a docket or ESPN data-entry error worth a second look). Regenerated fresh on every run; empty (header only) is the expected steady state.

## Running an update

The whole pipeline, from the repo root:

```bash
python -m src.ingestion.brazil.run_pipeline
```

This just runs the three stages below in order. You can also run any one on its own:

```bash
python -m src.ingestion.brazil.scrape_raw_matches    # updates data/raw/brazil/cbf/*.csv (CBF)
python -m src.ingestion.brazil.espn_fixtures         # updates data/raw/brazil/espn/*.csv (ESPN)
python -m src.ingestion.brazil.build_treated_dataset # rebuilds matches.csv + the unmapped-name log + score_discrepancies.csv
```

All of these are incremental and safe to re-run at any time (e.g. on a schedule) — they only fetch what's actually new, and `build_treated_dataset` alone makes zero network calls once `team_name_mapping.csv` exists.

### How `scrape_raw_matches` decides what to fetch

For each competition/season CSV, it looks at the `game_id` column already on disk:

- If it already has 380 rows (`GAMES_PER_SEASON` in `src/ingestion/brazil/constants.py`), the season is done and is **skipped entirely** — zero requests to CBF.
- Otherwise, it takes `resume_from = min(MAX(game_id), 380)` and:
  1. **Retries gaps**: any `game_id` between 1 and `resume_from` that isn't in the CSV yet.
  2. **Advances**: probes forward from `resume_from` until either 40 dockets in a row come back missing (`CONSECUTIVE_MISS_LIMIT` in `src/ingestion/brazil/season_scraper.py`) or it reaches game 380, whichever happens first.

There's no separate "last probed" bookkeeping — the `game_id` column already on disk is the only state the resume logic needs.

### New rounds (e.g. a season in progress)

Nothing special to do: just run `scrape_raw_matches` (or `run_pipeline`) again. Since the season doesn't have 380 rows yet, it isn't skipped, and step 2 above naturally picks up any newly published rounds by probing past the previous `MAX(game_id)`.

### A docket that exists but won't parse

Some dockets return HTTP 200 but have a corrupted PDF structure `cbf_docket.py`'s parser can't recover from (seen once: a bad `startxref` pointer making PyPDF2 raise `negative seek value -1` — different from the "garbage bytes after `%%EOF`" case the scraper already repairs for). These are indistinguishable from a genuinely unplayed/cancelled game using the 40-consecutive-miss signal alone, so `scrape_raw_matches` will keep retrying them forever (cheap — one request per run) without ever recovering them.

If a gap persists across many runs, check manually whether the docket actually exists (`curl -I` the CBF URL). If it does, either extend `cbf_docket._parse_docket` to fall back to a more lenient PDF library (e.g. `pypdf`/`pikepdf`/`pdfplumber`) when PyPDF2 fails, or fill in that one row by hand in the raw CSV.

## Fixtures & schedule (ESPN)

CBF's docket scraping only ever produces a row for a match that's already been played (that's what a súmula _is_). To know the schedule of what's coming up -- and to have _some_ score for a match CBF hasn't published a docket for yet -- `espn_fixtures.py` fetches ESPN's public scoreboard API (`site.api.espn.com`, no auth) for the whole current season, played and unplayed matches alike, and caches it under `data/raw/brazil/espn/{competition}_{year}.csv`. This is a deliberate, narrowly-scoped exception to this repo's normal "CBF is the only external source" rule -- it exists purely to fill in scheduling, never to override a CBF score.

`build_treated_dataset` merges the two sources with one precedence rule: **CBF's score always wins**. For every ESPN row:

- If CBF already has that exact `(competition, season, home_team, away_team)` match, the CBF row is kept as-is (`status="played"`); ESPN's row is dropped. If ESPN also reported a final score there and it disagrees with CBF's, that disagreement is logged to `score_discrepancies.csv` instead of changing anything in `matches.csv`.
- If CBF doesn't have that match yet, ESPN's row is added with `home_goals`/`away_goals` left empty, `status` set to `"scheduled"` or `"postponed"`, and `match_datetime` converted from ESPN's UTC timestamp to Brazil local time (`America/Sao_Paulo`, a fixed UTC-3 offset year-round since Brazil dropped DST in 2019).

Run it as part of the full pipeline (`python -m src.ingestion.brazil.run_pipeline`, which runs it before `build_treated_dataset` so the merge always sees fresh data), or on its own with `python -m src.ingestion.brazil.espn_fixtures`.

Unmapped ESPN team names (ESPN spells every club with no `/UF` suffix at all, e.g. `"Atlético-MG"` instead of `"Atlético Mineiro / MG"`) land in the _same_ `unmapped_team_names_log.csv` described above, alongside any unmapped CBF names from the same run -- resolve them the same way (see "Resolving unmapped team names" below).

## Resolving unmapped team names

`build_treated_dataset` keeps the raw name in `matches.csv` for any club it can't resolve, and logs every occurrence to `unmapped_team_names_log.csv` with the competition, season, game number, and the top 3 rapidfuzz suggestions (e.g. `Coritiba / PR (100); Londrina / PR (54); Operário / PR (46)`). ESPN-sourced rows have no CBF game number, so `game_code` is left blank for them.

To resolve them:

1. Open `unmapped_team_names_log.csv` and dedupe by `raw_name` (the same raw spelling usually shows up in many games).
2. For each one, decide the correct normalized spelling — the top suggestion is usually right when its score is close to 100 (e.g. a `SAF`-suffix variant of a club already in the mapping). A low top score (well under 100) usually means a genuinely new club with no prior spelling to match against — pick/confirm the spelling yourself.
3. Add a `raw_name,normalized_name` row to `team_name_mapping.csv` (keep it sorted by `raw_name` for easy diffing/reviewing).
4. Re-run `python -m src.ingestion.brazil.build_treated_dataset` — it reloads the mapping, so the rows you just added disappear from the next `unmapped_team_names_log.csv`, and `matches.csv` gets the corrected name.

The mapping is looked up case-insensitively, so you don't need a separate entry for every capitalization CBF happens to use for the same raw spelling.
