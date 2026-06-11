"""Per-sport configuration and filesystem paths."""

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models_store"
DB_PATH = DATA_DIR / "games.sqlite"


@dataclass(frozen=True)
class SportConfig:
    key: str                    # internal id: "mlb" | "nba"
    name: str                   # display name
    odds_api_key: str           # sport key used by the-odds-api.com
    elo_k: float                # Elo K-factor (lower = noisier sport)
    elo_home_advantage: float   # Elo points added to home team
    elo_season_regression: float  # fraction regressed to mean between seasons
    rolling_window: int         # games used for rolling form features
    rest_cap_days: int = 7      # rest days are capped here
    min_history_games: int = 5  # team games needed before features are "warm"
    seasons_default: list = field(default_factory=list)


MLB = SportConfig(
    key="mlb",
    name="MLB",
    odds_api_key="baseball_mlb",
    elo_k=4.0,
    elo_home_advantage=24.0,
    elo_season_regression=0.33,
    rolling_window=20,
    seasons_default=[2021, 2022, 2023, 2024, 2025, 2026],
)

NBA = SportConfig(
    key="nba",
    name="NBA",
    odds_api_key="basketball_nba",
    elo_k=20.0,
    elo_home_advantage=80.0,
    elo_season_regression=0.25,
    rolling_window=10,
    seasons_default=[2020, 2021, 2022, 2023, 2024, 2025],
)

SPORTS = {"mlb": MLB, "nba": NBA}


def get_sport(key: str) -> SportConfig:
    try:
        return SPORTS[key.lower()]
    except KeyError:
        raise ValueError(f"Unknown sport '{key}'. Choose from: {', '.join(SPORTS)}")
