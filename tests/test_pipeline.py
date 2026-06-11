"""End-to-end: store -> features -> train -> predict -> backtest on synthetic data."""

import dataclasses

import pytest

import sports_edge.models.train as train_mod
from sports_edge.backtest import run_backtest
from sports_edge.config import NBA
from sports_edge.data.store import GameStore
from sports_edge.features.builder import build_dataset
from sports_edge.models.predict import predict_games
from sports_edge.models.train import train

from tests.synthetic import synthetic_seasons

# NBA model params, but the sport key used by the synthetic data.
TEST_CFG = dataclasses.replace(NBA, key="test", seasons_default=[])


@pytest.fixture
def store(tmp_path):
    s = GameStore(tmp_path / "test.sqlite")
    yield s
    s.close()


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    path = tmp_path / "models"
    monkeypatch.setattr(train_mod, "MODELS_DIR", path)
    return path


def test_store_roundtrip(store):
    games = synthetic_seasons([2022])
    n = store.upsert_games(games)
    assert n == len(games)
    loaded = store.load_games("test")
    assert len(loaded) == len(games)
    assert loaded[0].date <= loaded[-1].date
    # Upsert again: no duplicates.
    store.upsert_games(games)
    assert store.count("test") == len(games)


def test_train_and_predict(store, models_dir):
    games = synthetic_seasons([2021, 2022, 2023])
    store.upsert_games(games)

    df = build_dataset(store.load_games("test"), TEST_CFG)
    result = train(df, TEST_CFG, fast=True)
    assert result.accuracy > 0.60  # skill-driven league is predictable
    assert result.log_loss < 0.69  # better than coin flip
    assert result.model_path.exists()

    upcoming = [{"home_team": "Team A", "away_team": "Team B", "date": "2024-01-01"}]
    odds = {("Team A", "Team B"): {"home_ml": -120, "away_ml": 105,
                                   "home_book": "x", "away_book": "y"}}
    preds = predict_games(TEST_CFG, upcoming, odds=odds, store=store)
    assert len(preds) == 1
    p = preds[0]
    assert 0.0 < p.home_win_prob < 1.0
    assert p.home_ml == -120
    assert p.home_ev is not None and p.away_ev is not None
    assert p.home_kelly is not None and 0.0 <= p.home_kelly <= 0.10
    # EV math is consistent with the probability.
    assert (p.home_ev > 0) == (p.home_win_prob > 120 / 220)
    assert p.pick in ("Team A", "Team B")
    # Score predictions are plausible for the synthetic league (~100-120 pts).
    assert 80 < p.pred_home_score < 140
    assert 80 < p.pred_away_score < 140
    assert p.pred_total == pytest.approx(p.pred_home_score + p.pred_away_score)
    assert p.confidence in ("STRONG", "LEAN", "PASS")
    assert p.home_form["elo_rank"] is not None
    assert "-" in p.home_form["last_n"]
    assert result.blend_weight is not None and 0.0 <= result.blend_weight <= 1.0
    assert result.margin_mae > 0 and result.total_mae > 0


def test_predict_without_odds(store, models_dir):
    games = synthetic_seasons([2021, 2022])
    store.upsert_games(games)
    train(build_dataset(store.load_games("test"), TEST_CFG), TEST_CFG, fast=True)

    upcoming = [{"home_team": "Team C", "away_team": "Team D", "date": "2023-01-01"}]
    preds = predict_games(TEST_CFG, upcoming, store=store)
    assert preds[0].home_ev is None
    assert preds[0].best_bet is None


def test_predict_requires_history(store, models_dir):
    with pytest.raises(RuntimeError, match="No historical"):
        predict_games(TEST_CFG, [{"home_team": "A", "away_team": "B",
                                  "date": "2024-01-01"}], store=store)


def test_backtest():
    games = synthetic_seasons([2021, 2022, 2023])
    df = build_dataset(games, TEST_CFG)
    result = run_backtest(df, TEST_CFG, fast=True)
    assert len(result.seasons) == 2  # first season is train-only
    assert result.overall_accuracy > 0.60
    for s in result.seasons:
        assert s.n_bets == 0  # no odds csv supplied


def test_backtest_with_odds_csv(tmp_path):
    games = synthetic_seasons([2021, 2022, 2023])
    df = build_dataset(games, TEST_CFG)
    # Fake market: every game priced at -110 both sides.
    odds = df[["date", "home_team", "away_team"]].copy()
    odds["home_ml"] = -110
    odds["away_ml"] = -110
    csv = tmp_path / "odds.csv"
    odds.to_csv(csv, index=False)

    result = run_backtest(df, TEST_CFG, odds_csv=str(csv), min_ev=2.0, fast=True)
    assert result.total_bets > 0
    # A skill-driven league priced as a coin flip should be very profitable.
    assert result.total_profit > 0
