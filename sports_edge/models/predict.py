"""Predict upcoming games with full outcome detail and betting analysis."""

from dataclasses import dataclass, field

import numpy as np

from sports_edge.betting.odds_api import team_key
from sports_edge.betting.odds_math import (
    american_to_implied_prob,
    expected_value,
    kelly_fraction,
    no_vig_probs,
)
from sports_edge.config import SportConfig, current_season
from sports_edge.data.store import GameStore
from sports_edge.features.builder import replay_state
from sports_edge.features.state import FEATURE_NAMES, parse_date
from sports_edge.models.train import blend, load_model


@dataclass
class Prediction:
    home_team: str
    away_team: str
    date: str
    home_win_prob: float         # blended (model + Elo) probability
    model_prob: float            # raw calibrated classifier probability
    elo_prob: float
    pred_margin: float           # predicted home margin (negative = away by that much)
    pred_total: float
    pred_home_score: float
    pred_away_score: float
    home_form: dict = field(default_factory=dict)   # elo, rank, last-N, streak, ppg...
    away_form: dict = field(default_factory=dict)
    home_ml: float | None = None
    away_ml: float | None = None
    home_book: str | None = None
    away_book: str | None = None
    home_ev: float | None = None       # expected profit per $100
    away_ev: float | None = None
    home_kelly: float | None = None    # fraction of bankroll
    away_kelly: float | None = None
    home_fair_prob: float | None = None  # market's no-vig home probability

    @property
    def away_win_prob(self) -> float:
        return 1.0 - self.home_win_prob

    @property
    def pick(self) -> str:
        return self.home_team if self.home_win_prob >= 0.5 else self.away_team

    @property
    def pick_prob(self) -> float:
        return max(self.home_win_prob, self.away_win_prob)

    @property
    def edge(self) -> float | None:
        """Model probability minus market no-vig probability, for the pick side."""
        if self.home_fair_prob is None:
            return None
        market = self.home_fair_prob if self.pick == self.home_team \
            else 1.0 - self.home_fair_prob
        return self.pick_prob - market

    @property
    def confidence(self) -> str:
        """Tier based on market edge when odds exist, else on probability."""
        if self.edge is not None:
            if self.edge >= 0.05:
                return "STRONG"
            if self.edge >= 0.02:
                return "LEAN"
            return "PASS"
        if self.pick_prob >= 0.65:
            return "STRONG"
        if self.pick_prob >= 0.575:
            return "LEAN"
        return "TOSS-UP"

    @property
    def best_bet(self) -> str | None:
        """Side with positive EV, if any."""
        if self.home_ev is None or self.away_ev is None:
            return None
        best = max(self.home_ev, self.away_ev)
        if best <= 0:
            return None
        return self.home_team if self.home_ev >= self.away_ev else self.away_team

    def _best_bet_field(self, home_value, away_value):
        if self.best_bet is None:
            return None
        return home_value if self.best_bet == self.home_team else away_value

    @property
    def best_bet_ev(self) -> float | None:
        return self._best_bet_field(self.home_ev, self.away_ev)

    @property
    def best_bet_kelly(self) -> float | None:
        return self._best_bet_field(self.home_kelly, self.away_kelly)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d.update({
            "pick": self.pick,
            "pick_prob": self.pick_prob,
            "edge": self.edge,
            "confidence": self.confidence,
            "best_bet": self.best_bet,
        })
        return d


def predict_games(cfg: SportConfig, upcoming: list[dict],
                  odds: dict[tuple[str, str], dict] | None = None,
                  store: GameStore | None = None,
                  kelly_multiplier: float = 0.25) -> list[Prediction]:
    """Full predictions (probability, score, form, EV/Kelly) for upcoming games."""
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
    w = bundle.get("blend_weight", 1.0)
    odds_by_key = {(team_key(h), team_key(a)): v for (h, a), v in (odds or {}).items()}

    preds = []
    for g in upcoming:
        home, away = g["home_team"], g["away_team"]
        # If these games open a new season, apply between-season regression
        # and form resets first — mirrors what build_dataset does in training.
        state.rollover_if_new_season(current_season(cfg, parse_date(g["date"])))
        feats = state.features_for(home, away, g["date"])
        x = np.array([[feats[name] for name in FEATURE_NAMES]])
        p_model = float(model.predict_proba(x)[0, 1])
        p_elo = feats["elo_prob_home"]
        p_home = float(blend(p_model, p_elo, w))

        margin = float(bundle["margin_model"].predict(x)[0])
        total = float(bundle["total_model"].predict(x)[0])
        pred = Prediction(
            home_team=home, away_team=away, date=g["date"],
            home_win_prob=p_home, model_prob=p_model, elo_prob=p_elo,
            pred_margin=margin, pred_total=total,
            pred_home_score=(total + margin) / 2.0,
            pred_away_score=(total - margin) / 2.0,
            home_form=state.team_summary(home),
            away_form=state.team_summary(away),
        )
        o = odds_by_key.get((team_key(home), team_key(away)))
        if o:
            try:
                pred.home_fair_prob, _ = no_vig_probs(
                    american_to_implied_prob(o["home_ml"]),
                    american_to_implied_prob(o["away_ml"]),
                )
                pred.home_ev = expected_value(p_home, o["home_ml"])
                pred.away_ev = expected_value(1 - p_home, o["away_ml"])
                pred.home_kelly = kelly_fraction(p_home, o["home_ml"], kelly_multiplier)
                pred.away_kelly = kelly_fraction(1 - p_home, o["away_ml"],
                                                 kelly_multiplier)
                pred.home_ml, pred.away_ml = o["home_ml"], o["away_ml"]
                pred.home_book, pred.away_book = o.get("home_book"), o.get("away_book")
            except (ValueError, KeyError, TypeError):
                pass  # malformed odds for this game: keep probability-only output
        preds.append(pred)
    return preds
