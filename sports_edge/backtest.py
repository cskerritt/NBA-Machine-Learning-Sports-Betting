"""Walk-forward backtest: for each season, train only on prior seasons.

Optionally simulates a betting strategy against real odds supplied as a CSV
with columns: date, home_team, away_team, home_ml, away_ml (American odds).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from sports_edge.betting.odds_math import american_to_decimal, expected_value
from sports_edge.config import SportConfig
from sports_edge.features.state import FEATURE_NAMES
from sports_edge.models.train import _make_estimator


@dataclass
class SeasonResult:
    season: int
    n_games: int
    accuracy: float
    log_loss: float
    brier: float
    elo_accuracy: float
    n_bets: int = 0
    profit: float = 0.0  # flat $100 stakes
    roi: float | None = None


@dataclass
class BacktestResult:
    seasons: list[SeasonResult] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        n = sum(s.n_games for s in self.seasons)
        return sum(s.accuracy * s.n_games for s in self.seasons) / n

    @property
    def total_profit(self) -> float:
        return sum(s.profit for s in self.seasons)

    @property
    def total_bets(self) -> int:
        return sum(s.n_bets for s in self.seasons)


def _merge_odds(df: pd.DataFrame, odds_csv: str | None) -> pd.DataFrame:
    if not odds_csv:
        df["home_ml"] = np.nan
        df["away_ml"] = np.nan
        return df
    odds = pd.read_csv(odds_csv)
    required = {"date", "home_team", "away_team", "home_ml", "away_ml"}
    missing = required - set(odds.columns)
    if missing:
        raise ValueError(f"Odds CSV missing columns: {sorted(missing)}")
    return df.merge(odds[sorted(required)], on=["date", "home_team", "away_team"],
                    how="left")


def run_backtest(df: pd.DataFrame, cfg: SportConfig, odds_csv: str | None = None,
                 min_ev: float = 2.0, fast: bool = False) -> BacktestResult:
    """min_ev: only bet sides with expected profit above this (per $100 flat stake)."""
    df = df[df["warm"] == 1].sort_values("date").reset_index(drop=True)
    df = _merge_odds(df, odds_csv)
    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        raise ValueError("Backtest needs at least 2 seasons of data.")

    result = BacktestResult()
    for season in seasons[1:]:
        train_df = df[df["season"] < season]
        test_df = df[df["season"] == season]
        if len(train_df) < 200 or len(test_df) < 30:
            continue

        model = CalibratedClassifierCV(_make_estimator(fast), method="isotonic", cv=3)
        model.fit(train_df[FEATURE_NAMES].to_numpy(dtype=float),
                  train_df["home_win"].to_numpy())
        p = model.predict_proba(test_df[FEATURE_NAMES].to_numpy(dtype=float))[:, 1]
        y = test_df["home_win"].to_numpy()
        p_elo = test_df["elo_prob_home"].to_numpy(dtype=float)

        sr = SeasonResult(
            season=int(season),
            n_games=len(test_df),
            accuracy=accuracy_score(y, p > 0.5),
            log_loss=log_loss(y, p, labels=[0, 1]),
            brier=brier_score_loss(y, p),
            elo_accuracy=accuracy_score(y, p_elo > 0.5),
        )

        # Betting simulation when odds are available: flat $100 per qualifying bet.
        for prob, won, home_ml, away_ml in zip(
            p, y, test_df["home_ml"].to_numpy(), test_df["away_ml"].to_numpy()
        ):
            if np.isnan(home_ml) or np.isnan(away_ml):
                continue
            ev_home = expected_value(prob, home_ml)
            ev_away = expected_value(1 - prob, away_ml)
            side_ev, side_ml, side_won = (
                (ev_home, home_ml, won == 1) if ev_home >= ev_away
                else (ev_away, away_ml, won == 0)
            )
            if side_ev < min_ev:
                continue
            sr.n_bets += 1
            sr.profit += (american_to_decimal(side_ml) - 1.0) * 100.0 if side_won else -100.0
        if sr.n_bets:
            sr.roi = sr.profit / (sr.n_bets * 100.0) * 100.0
        result.seasons.append(sr)
    return result
