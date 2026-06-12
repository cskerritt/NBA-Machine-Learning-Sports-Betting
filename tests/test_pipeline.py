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
    if p.best_bet is not None:
        assert p.best_bet_ev == (p.home_ev if p.best_bet == "Team A" else p.away_ev)
        assert p.best_bet_kelly is not None
    # Score predictions are plausible for the synthetic league (~100-120 pts).
    assert 80 < p.pred_home_score < 140
    assert 80 < p.pred_away_score < 140
    assert p.pred_total == pytest.approx(p.pred_home_score + p.pred_away_score)
    assert p.confidence in ("STRONG", "LEAN", "PASS")
    assert p.home_form["elo_rank"] is not None
    assert "-" in p.home_form["last_n"]
    assert result.blend_weight is not None and 0.0 <= result.blend_weight <= 1.0
    assert result.margin_mae > 0 and result.total_mae > 0


def test_predict_new_season_applies_regression(store, models_dir):
    """Predicting a new season's opener must use regressed Elo ratings,
    matching how build_dataset treats season boundaries in training."""
    store.upsert_games(synthetic_seasons([2021, 2022, 2023]))
    train(build_dataset(store.load_games("test"), TEST_CFG), TEST_CFG, fast=True)

    from sports_edge.features.elo import Elo
    baseline = Elo(TEST_CFG.elo_k, TEST_CFG.elo_home_advantage,
                   TEST_CFG.elo_season_regression).expected_home_win_prob("X", "Y")

    matchup = {"home_team": "Team A", "away_team": "Team B"}
    # TEST_CFG seasons start in October: 2024-01-01 is still season 2023
    # (no rollover); 2024-12-01 is season 2024 (rollover applies).
    p_same = predict_games(TEST_CFG, [dict(matchup, date="2024-01-01")], store=store)[0]
    p_new = predict_games(TEST_CFG, [dict(matchup, date="2024-12-01")], store=store)[0]
    assert abs(p_new.elo_prob - baseline) < abs(p_same.elo_prob - baseline)


def test_stale_model_rejected(store, models_dir):
    """A pickle trained on a different feature set must be refused, not
    silently fed misaligned features."""
    import pickle

    store.upsert_games(synthetic_seasons([2021, 2022]))
    result = train(build_dataset(store.load_games("test"), TEST_CFG), TEST_CFG, fast=True)

    with open(result.model_path, "rb") as f:
        bundle = pickle.load(f)
    bundle["feature_names"] = ["elo_home", "elo_away"]  # stale feature set
    with open(result.model_path, "wb") as f:
        pickle.dump(bundle, f)

    with pytest.raises(ValueError, match="different feature set"):
        predict_games(TEST_CFG, [{"home_team": "Team A", "away_team": "Team B",
                                  "date": "2023-01-01"}], store=store)


def test_malformed_odds_skipped(store, models_dir):
    """Invalid American odds (between -100 and +100) must not crash the run."""
    store.upsert_games(synthetic_seasons([2021, 2022]))
    train(build_dataset(store.load_games("test"), TEST_CFG), TEST_CFG, fast=True)

    upcoming = [{"home_team": "Team A", "away_team": "Team B", "date": "2023-01-01"}]
    odds = {("Team A", "Team B"): {"home_ml": 50, "away_ml": -110,
                                   "home_book": "x", "away_book": "y"}}
    p = predict_games(TEST_CFG, upcoming, odds=odds, store=store)[0]
    assert p.home_ml is None and p.home_ev is None  # falls back to prob-only


def test_predict_slate_flags_unmatched_odds(store, models_dir, monkeypatch):
    store.upsert_games(synthetic_seasons([2021, 2022]))
    train(build_dataset(store.load_games("test"), TEST_CFG), TEST_CFG, fast=True)

    import sports_edge.betting.odds_api as odds_api
    monkeypatch.setattr(odds_api, "fetch_moneylines", lambda *a, **k: {})
    from sports_edge.pipeline import predict_slate

    upcoming = [{"home_team": "Team A", "away_team": "Team B", "date": "2023-01-01"}]
    preds, odds_error = predict_slate(TEST_CFG, upcoming, store=store, api_key="k")
    assert len(preds) == 1
    assert "no matching lines" in odds_error


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
