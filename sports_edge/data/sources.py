"""Dispatch to the right data source for each sport."""

from datetime import date

from sports_edge.config import SportConfig, current_season
from sports_edge.data.store import Game, GameStore


def fetch_season(cfg: SportConfig, season: int) -> list[Game]:
    if cfg.key == "mlb":
        from sports_edge.data import mlb
        return mlb.fetch_season(season)
    if cfg.key == "nba":
        from sports_edge.data import nba
        return nba.fetch_season(season)
    from sports_edge.data import espn
    return espn.fetch_season(cfg, season)


def todays_games(cfg: SportConfig, day: date | None = None) -> list[dict]:
    if cfg.key == "mlb":
        from sports_edge.data import mlb
        return mlb.todays_games(day)
    if cfg.key == "nba":
        from sports_edge.data import nba
        return nba.todays_games(day)
    from sports_edge.data import espn
    return espn.todays_games(cfg, day)


def update_history(cfg: SportConfig, store: GameStore,
                   today: date | None = None) -> dict[int, int]:
    """Fetch any default seasons missing from the store, plus always refresh
    the in-progress season. Returns {season: games_saved}. Incremental, so a
    daily run only re-downloads the current season."""
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
