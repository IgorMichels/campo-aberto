"""Shared constants for the Brazilian football ingestion pipeline."""

COMPETITIONS = {
    "Serie_A": "Serie A",
    "Serie_B": "Serie B",
}
START_YEAR = 2020
END_YEAR = 2026
GAMES_PER_SEASON = 380

# cbf_docket.py
DOCKET_URL = "https://conteudo.cbf.com.br/sumulas/{year}/{code}{game}se.pdf"
COMPETITION_CODES = {
    "Serie_A": "142",
    "Serie_B": "242",
}
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2

# season_scraper.py
CONSECUTIVE_MISS_LIMIT = 40

# team_name_mapping.py
MAPPING_PATH = "data/processed/brazil/team_name_mapping.csv"
SUGGESTION_COUNT = 3

# build_treated_dataset.py
OUTPUT_PATH = "data/processed/brazil/matches.csv"
UNMAPPED_LOG_PATH = "data/processed/brazil/unmapped_team_names_log.csv"

# scrape_raw_matches.py
CBF_CACHE_DIR = "data/raw/brazil/cbf"

# espn_fixtures.py
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/scoreboard"
)
ESPN_LEAGUE_CODES = {"Serie_A": "bra.1", "Serie_B": "bra.2"}  # keyed like COMPETITIONS
ESPN_CACHE_DIR = "data/raw/brazil/espn"  # sibling of CBF_CACHE_DIR -- each
# external source gets its own subdirectory under data/raw/brazil/.
