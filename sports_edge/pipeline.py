"""One-call orchestration shared by the CLI, web dashboard, and daily runs:
today's slate -> live odds (optional) -> full predictions."""

from datetime import date

from sports_edge.config import SportConfig
from sports_edge.data.sources import todays_games
from sports_edge.data.store import GameStore
from sports_edge.models.predict import Prediction, predict_games


def predict_slate(cfg: SportConfig, upcoming: list[dict], *,
                  with_odds: bool = True, api_key: str | None = None,
                  bookmaker: str | None = None, kelly_multiplier: float = 0.25,
                  store: GameStore | None = None) -> tuple[list[Prediction], str | None]:
    """Predictions for a known slate. Returns (predictions, odds_error).

    Odds failures never block predictions: when the odds fetch fails the
    error string is returned alongside probability-only predictions, and
    each caller decides how loudly to surface it.
    """
    if not upcoming:
        return [], None

    odds, odds_error = None, None
    if with_odds:
        from sports_edge.betting.odds_api import fetch_moneylines
        try:
            odds = fetch_moneylines(cfg, api_key=api_key, bookmaker=bookmaker)
        except Exception as e:
            odds_error = str(e)

    preds = predict_games(cfg, upcoming, odds=odds, store=store,
                          kelly_multiplier=kelly_multiplier)
    if odds is not None and odds_error is None:
        unmatched = sum(1 for p in preds if p.home_ml is None)
        if unmatched:
            odds_error = (f"odds fetched but {unmatched} of {len(preds)} games "
                          f"had no matching lines (team-name mismatch or market "
                          f"not yet posted)")
    return preds, odds_error


def predict_today(cfg: SportConfig, *, day: date | None = None,
                  **kwargs) -> tuple[list[Prediction], str | None]:
    """Fetch today's slate and predict it. See predict_slate for kwargs."""
    return predict_slate(cfg, todays_games(cfg, day), **kwargs)
