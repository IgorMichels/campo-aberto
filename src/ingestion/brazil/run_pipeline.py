"""Runs the full Brazil ingestion pipeline end to end:

1. scrape_raw_matches  -- update data/raw/brazil/cbf/*.csv from CBF
2. espn_fixtures       -- update data/raw/brazil/espn/*.csv from ESPN's
   schedule (current season's fixtures/scores; must run before step 3)
3. build_treated_dataset -- rebuild data/processed/brazil/matches.csv (and the
   team-name mapping / unmapped-name log) from whatever is on disk

All three stages are incremental on their own, so running this repeatedly
only does the work that's actually new.
"""

from src.ingestion.brazil import build_treated_dataset, espn_fixtures, scrape_raw_matches


def main() -> None:
    scrape_raw_matches.main()
    espn_fixtures.main()
    build_treated_dataset.main()


if __name__ == "__main__":
    main()
