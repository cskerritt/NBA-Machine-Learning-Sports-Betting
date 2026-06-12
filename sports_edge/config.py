"""Per-sport configuration and filesystem paths."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models_store"
REPORTS_DIR = ROOT_DIR / "reports"
DB_PATH = DATA_DIR / "games.sqlite"


@dataclass(frozen=True)
class SportConfig:
    key: str                    # internal id: "mlb" | "nba" | "nfl" | "nhl"
    name: str                   # display name
    odds_api_key: str           # sport key used by the-odds-api.com
    espn_path: str              # ESPN API path, e.g. "football/nfl"
    elo_k: float                # Elo K-factor (lower = noisier sport)
    elo_home_advantage: float   # Elo points added to home team
    elo_season_regression: float  # fraction regressed to mean between seasons
    rolling_window: int         # games used for rolling form features
    season_start_month: int     # month a season starts (decides "current" season)
    # Regular-season fetch window: ((start month, day), (end month, day));
    # the end rolls into the next calendar year when it precedes the start.
    season_window: tuple
    rest_cap_days: int = 7      # rest days are capped here
    min_history_games: int = 5  # team games needed before features are "warm"
    seasons_default: list = field(default_factory=list)


MLB = SportConfig(
    key="mlb",
    name="MLB",
    odds_api_key="baseball_mlb",
    espn_path="baseball/mlb",
    elo_k=4.0,
    elo_home_advantage=24.0,
    elo_season_regression=0.33,
    rolling_window=20,
    season_start_month=3,
    season_window=((3, 15), (10, 10)),
    seasons_default=[2021, 2022, 2023, 2024, 2025],
)

NBA = SportConfig(
    key="nba",
    name="NBA",
    odds_api_key="basketball_nba",
    espn_path="basketball/nba",
    elo_k=20.0,
    elo_home_advantage=80.0,
    elo_season_regression=0.25,
    rolling_window=10,
    season_start_month=10,
    season_window=((10, 1), (4, 30)),
    seasons_default=[2020, 2021, 2022, 2023, 2024, 2025],
)

NFL = SportConfig(
    key="nfl",
    name="NFL",
    odds_api_key="americanfootball_nfl",
    espn_path="football/nfl",
    elo_k=20.0,
    elo_home_advantage=48.0,
    elo_season_regression=0.33,
    rolling_window=5,
    season_start_month=8,
    season_window=((9, 1), (1, 15)),
    rest_cap_days=14,
    min_history_games=3,
    seasons_default=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
)

NHL = SportConfig(
    key="nhl",
    name="NHL",
    odds_api_key="icehockey_nhl",
    espn_path="hockey/nhl",
    elo_k=6.0,
    elo_home_advantage=33.0,
    elo_season_regression=0.30,
    rolling_window=10,
    season_start_month=10,
    season_window=((10, 1), (4, 30)),
    seasons_default=[2020, 2021, 2022, 2023, 2024, 2025],
)

SPORTS = {"mlb": MLB, "nba": NBA, "nfl": NFL, "nhl": NHL}


def get_sport(key: str) -> SportConfig:
    try:
        return SPORTS[key.lower()]
    except KeyError:
        raise ValueError(f"Unknown sport '{key}'. Choose from: {', '.join(SPORTS)}")


def current_season(cfg: SportConfig, today: date | None = None) -> int:
    """The season (identified by its starting year) that `today` falls in.

    Cross-year sports (NBA/NHL/NFL) belong to the previous calendar year
    until the new season starts; MLB seasons match the calendar year.
    """
    today = today or date.today()
    return today.year if today.month >= cfg.season_start_month else today.year - 1
