"""Data-source interface for all sports. Everything comes from ESPN.

The store records which source each sport's games came from; if the
source ever changes, the sport's games are purged and refetched so that
source-specific game ids never mix (which would create duplicates).
"""

from datetime import date

from sports_edge.config import SportConfig, current_season
from sports_edge.data import espn
from sports_edge.data.store import Game, GameStore

SOURCE = "espn"


def fetch_season(cfg: SportConfig, season: int) -> list[Game]:
    return espn.fetch_season(cfg, season)


def todays_games(cfg: SportConfig, day: date | None = None) -> list[dict]:
    return espn.todays_games(cfg, day)


def update_history(cfg: SportConfig, store: GameStore,
                   today: date | None = None) -> dict[int, int]:
    """Fetch any default seasons missing from the store, plus always refresh
    the in-progress season. Returns {season: games_saved}. Incremental, so a
    daily run only re-downloads the current season."""
    store.ensure_source(cfg.key, SOURCE)
    have = set(store.seasons(cfg.key))
    now_season = current_season(cfg, today)
    wanted = [s for s in cfg.seasons_default if s not in have and s <= now_season]
    if now_season not in wanted:
        wanted.append(now_season)

    saved = {}
    for season in sorted(wanted):
        games = fetch_season(cfg, season)
        saved[season] = store.upsert_games(games) if games else 0
    return saved
