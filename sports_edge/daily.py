"""Daily pipeline: update data, retrain, predict, and write reports for
every sport that has games today. Designed for cron / GitHub Actions."""

from dataclasses import dataclass, field
from datetime import date

from sports_edge.config import SPORTS, SportConfig
from sports_edge.data.sources import todays_games, update_history
from sports_edge.data.store import GameStore


@dataclass
class DailyResult:
    sport: str
    status: str                 # "ok" | "no-games" | "error"
    n_games: int = 0
    n_bets: int = 0
    report_md: str | None = None
    error: str | None = None
    fetched: dict = field(default_factory=dict)


def run_sport(cfg: SportConfig, store: GameStore, day: date,
              retrain: bool = True, bankroll: float = 1000.0,
              api_key: str | None = None) -> DailyResult:
    from sports_edge.features.builder import build_dataset
    from sports_edge.models.predict import predict_games
    from sports_edge.models.train import model_path, train
    from sports_edge.reports import write_report

    result = DailyResult(sport=cfg.key, status="ok")
    try:
        result.fetched = update_history(cfg, store, today=day)

        upcoming = todays_games(cfg, day)
        if not upcoming:
            result.status = "no-games"
            return result

        if retrain or not model_path(cfg).exists():
            games = store.load_games(cfg.key)
            train(build_dataset(games, cfg), cfg)

        odds = None
        try:
            from sports_edge.betting.odds_api import fetch_moneylines
            odds = fetch_moneylines(cfg, api_key=api_key)
        except Exception:
            pass  # odds are optional; report still gets probabilities

        preds = predict_games(cfg, upcoming, odds=odds, store=store)
        _, md_path = write_report(cfg, preds, day, bankroll=bankroll)
        result.n_games = len(preds)
        result.n_bets = sum(1 for p in preds if p.best_bet)
        result.report_md = str(md_path)
    except Exception as e:
        result.status = "error"
        result.error = str(e)
    return result


def run_daily(sports: list[str] | None = None, day: date | None = None,
              retrain: bool = True, bankroll: float = 1000.0,
              api_key: str | None = None,
              store: GameStore | None = None) -> list[DailyResult]:
    day = day or date.today()
    store = store or GameStore()
    keys = sports or list(SPORTS)
    return [run_sport(SPORTS[k], store, day, retrain=retrain,
                      bankroll=bankroll, api_key=api_key) for k in keys]
