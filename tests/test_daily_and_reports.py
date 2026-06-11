"""Daily pipeline and report generation, fully offline via monkeypatching."""

import dataclasses
import json
from datetime import date

import pytest

import sports_edge.daily as daily_mod
import sports_edge.models.train as train_mod
import sports_edge.reports as reports_mod
from sports_edge.daily import run_sport
from sports_edge.data.store import GameStore
from sports_edge.features.builder import build_dataset
from sports_edge.models.train import train
from sports_edge.reports import write_report
from tests.synthetic import synthetic_seasons
from tests.test_pipeline import TEST_CFG


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Seeded store + redirected models/reports dirs + pre-trained model."""
    store = GameStore(tmp_path / "test.sqlite")
    store.upsert_games(synthetic_seasons([2021, 2022, 2023]))
    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(reports_mod, "REPORTS_DIR", tmp_path / "reports")
    train(build_dataset(store.load_games("test"), TEST_CFG), TEST_CFG, fast=True)
    yield store, tmp_path
    store.close()


def test_write_report(env):
    store, tmp_path = env
    from sports_edge.models.predict import predict_games

    upcoming = [{"home_team": "Team A", "away_team": "Team B", "date": "2024-01-01"}]
    odds = {("Team A", "Team B"): {"home_ml": 250, "away_ml": -300,
                                   "home_book": "x", "away_book": "y"}}
    preds = predict_games(TEST_CFG, upcoming, odds=odds, store=store)
    json_path, md_path = write_report(TEST_CFG, preds, date(2024, 1, 1))

    data = json.loads(json_path.read_text())
    assert data["sport"] == "test"
    assert len(data["games"]) == 1
    g = data["games"][0]
    assert {"pick", "pick_prob", "edge", "confidence", "pred_home_score",
            "home_form"} <= set(g)

    md = md_path.read_text()
    assert "Team A" in md and "Predicted score" in md and "Odds:" in md
    assert (tmp_path / "reports" / "latest" / "test.md").exists()


def test_run_sport_daily(env, monkeypatch):
    store, tmp_path = env
    monkeypatch.setattr(daily_mod, "update_history", lambda cfg, s, today=None: {2023: 0})
    monkeypatch.setattr(
        daily_mod, "todays_games",
        lambda cfg, day=None: [{"home_team": "Team C", "away_team": "Team D",
                                "date": "2024-01-02"}])

    r = run_sport(TEST_CFG, store, date(2024, 1, 2), retrain=False)
    assert r.status == "ok", r.error
    assert r.n_games == 1
    assert r.report_md and (tmp_path / "reports" / "2024-01-02" / "test.md").exists()


def test_run_sport_no_games(env, monkeypatch):
    store, _ = env
    monkeypatch.setattr(daily_mod, "update_history", lambda cfg, s, today=None: {})
    monkeypatch.setattr(daily_mod, "todays_games", lambda cfg, day=None: [])
    r = run_sport(TEST_CFG, store, date(2024, 7, 1), retrain=False)
    assert r.status == "no-games"


def test_run_sport_error_is_captured(env, monkeypatch):
    store, _ = env

    def boom(cfg, s, today=None):
        raise RuntimeError("api down")

    monkeypatch.setattr(daily_mod, "update_history", boom)
    r = run_sport(TEST_CFG, store, date(2024, 1, 2))
    assert r.status == "error"
    assert "api down" in r.error


def test_update_history_incremental(tmp_path, monkeypatch):
    from sports_edge.data import sources

    store = GameStore(tmp_path / "x.sqlite")
    cfg = dataclasses.replace(TEST_CFG, seasons_default=[2022, 2023],
                              season_start_month=1)
    fetched = []

    def fake_fetch(cfg_, season):
        fetched.append(season)
        # Prefix ids with the season: synthetic ids restart at 1 every call,
        # unlike real league game ids which are globally unique.
        return [dataclasses.replace(g, game_id=f"{season}-{g.game_id}")
                for g in synthetic_seasons([season])]

    monkeypatch.setattr(sources, "fetch_season", fake_fetch)
    sources.update_history(cfg, store, today=date(2023, 6, 1))
    assert fetched == [2022, 2023]

    # Second run: only the current season is refreshed.
    fetched.clear()
    sources.update_history(cfg, store, today=date(2023, 6, 1))
    assert fetched == [2023]
    store.close()
