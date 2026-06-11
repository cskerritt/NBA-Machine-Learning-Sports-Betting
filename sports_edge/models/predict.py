"""Predict upcoming games and attach betting analysis."""

from dataclasses import dataclass

import numpy as np

from sports_edge.betting.odds_math import (
    american_to_implied_prob,
    expected_value,
    kelly_fraction,
    no_vig_probs,
)
from sports_edge.config import SportConfig
from sports_edge.data.store import GameStore
from sports_edge.features.builder import replay_state
from sports_edge.features.state import FEATURE_NAMES
from sports_edge.models.train import load_model


@dataclass
class Prediction:
    home_team: str
    away_team: str
    date: str
    home_win_prob: float
    elo_prob: float
    home_ml: float | None = None
    away_ml: float | None = None
    home_book: str | None = None
    away_book: str | None = None
    home_ev: float | None = None       # expected profit per $100
    away_ev: float | None = None
    home_kelly: float | None = None    # fraction of bankroll
    away_kelly: float | None = None
    home_fair_prob: float | None = None  # market's no-vig probability

    @property
    def away_win_prob(self) -> float:
        return 1.0 - self.home_win_prob

    @property
    def pick(self) -> str:
        return self.home_team if self.home_win_prob >= 0.5 else self.away_team

    @property
    def best_bet(self) -> str | None:
        """Side with positive EV, if any."""
        if self.home_ev is None or self.away_ev is None:
            return None
        best = max(self.home_ev, self.away_ev)
        if best <= 0:
            return None
        return self.home_team if self.home_ev >= self.away_ev else self.away_team


def predict_games(cfg: SportConfig, upcoming: list[dict],
                  odds: dict[tuple[str, str], dict] | None = None,
                  store: GameStore | None = None,
                  kelly_multiplier: float = 0.25) -> list[Prediction]:
    """Win probabilities (and EV/Kelly when odds are given) for upcoming games."""
    store = store or GameStore()
    games = store.load_games(cfg.key)
    if not games:
        raise RuntimeError(
            f"No historical {cfg.name} games in the database. "
            f"Run: python main.py fetch --sport {cfg.key}"
        )
    state = replay_state(games, cfg)
    bundle = load_model(cfg)
    model = bundle["model"]

    preds = []
    for g in upcoming:
        home, away = g["home_team"], g["away_team"]
        feats = state.features_for(home, away, g["date"])
        x = np.array([[feats[name] for name in FEATURE_NAMES]])
        p_home = float(model.predict_proba(x)[0, 1])
        pred = Prediction(
            home_team=home, away_team=away, date=g["date"],
            home_win_prob=p_home, elo_prob=feats["elo_prob_home"],
        )
        if odds and (home, away) in odds:
            o = odds[(home, away)]
            pred.home_ml, pred.away_ml = o["home_ml"], o["away_ml"]
            pred.home_book, pred.away_book = o.get("home_book"), o.get("away_book")
            pred.home_fair_prob, _ = no_vig_probs(
                american_to_implied_prob(pred.home_ml),
                american_to_implied_prob(pred.away_ml),
            )
            pred.home_ev = expected_value(p_home, pred.home_ml)
            pred.away_ev = expected_value(1 - p_home, pred.away_ml)
            pred.home_kelly = kelly_fraction(p_home, pred.home_ml, kelly_multiplier)
            pred.away_kelly = kelly_fraction(1 - p_home, pred.away_ml, kelly_multiplier)
        preds.append(pred)
    return preds
