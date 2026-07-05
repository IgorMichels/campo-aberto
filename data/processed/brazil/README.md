# Treated dataset (Brazil)

This folder holds the output of the Brazil ingestion pipeline (`src/ingestion/brazil/`). It's built in two stages:

```
CBF dockets  →  data/raw/brazil/{competition}_{year}.csv  →  data/processed/brazil/matches.csv
   (scrape)              raw, no name treatment                treated, mapped names
```

Each country gets its own subfolder under `data/raw/` and `data/processed/` (mirroring `src/ingestion/`), so this layout is expected to repeat once other countries are added.

- **`matches.csv`** — one row per match: `competition, season, match_datetime, venue, home_team, away_team, home_goals, away_goals`.
- **`team_name_mapping.csv`** — the raw → normalized club-name "de-para" table (2 columns: `raw_name, normalized_name`). Manually curated over time; the pipeline never overwrites it once it exists.
- **`unmapped_team_names_log.csv`** — raw club names found in `data/raw/brazil/` with no entry in the mapping yet, with rapidfuzz suggestions. Regenerated fresh on every run (not appended).

## Running an update

The whole pipeline, from the repo root:

```bash
python -m src.ingestion.brazil.run_pipeline
```

This just runs the two stages below in order. You can also run either one on its own:

```bash
python -m src.ingestion.brazil.scrape_raw_matches    # updates data/raw/brazil/*.csv
python -m src.ingestion.brazil.build_treated_dataset # rebuilds data/processed/brazil/matches.csv + the unmapped-name log
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

## Resolving unmapped team names

`build_treated_dataset` keeps the raw name in `matches.csv` for any club it can't resolve, and logs every occurrence to `unmapped_team_names_log.csv` with the competition, season, game number, and the top 3 rapidfuzz suggestions (e.g. `Coritiba / PR (100); Londrina / PR (54); Operário / PR (46)`).

To resolve them:

1. Open `unmapped_team_names_log.csv` and dedupe by `raw_name` (the same raw spelling usually shows up in many games).
2. For each one, decide the correct normalized spelling — the top suggestion is usually right when its score is close to 100 (e.g. a `SAF`-suffix variant of a club already in the mapping). A low top score (well under 100) usually means a genuinely new club with no prior spelling to match against — pick/confirm the spelling yourself.
3. Add a `raw_name,normalized_name` row to `team_name_mapping.csv` (keep it sorted by `raw_name` for easy diffing/reviewing).
4. Re-run `python -m src.ingestion.brazil.build_treated_dataset` — it reloads the mapping, so the rows you just added disappear from the next `unmapped_team_names_log.csv`, and `matches.csv` gets the corrected name.

The mapping is looked up case-insensitively, so you don't need a separate entry for every capitalization CBF happens to use for the same raw spelling.
