from datetime import date

from sports_edge.betting.odds_api import team_key
from sports_edge.config import MLB, NBA, NFL, NHL, current_season
from sports_edge.data import espn


def _event(eid, day, home, away, hs, as_, completed=True):
    return {
        "id": eid,
        "date": f"{day}T00:00Z",
        "competitions": [{
            "status": {"type": {"completed": completed}},
            "competitors": [
                {"homeAway": "home", "team": {"displayName": home}, "score": str(hs)},
                {"homeAway": "away", "team": {"displayName": away}, "score": str(as_)},
            ],
        }],
    }


def test_parse_event():
    g = espn._parse_event(_event("401", "2024-10-06", "Buffalo Bills",
                                 "Houston Texans", 20, 23), "nfl", 2024)
    assert g.home_team == "Buffalo Bills"
    assert g.away_team == "Houston Texans"
    assert g.home_score == 20 and g.away_score == 23
    assert not g.home_win
    assert g.season == 2024


def test_parse_event_skips_unfinished():
    g = espn._parse_event(_event("401", "2024-10-06", "A", "B", 0, 0,
                                 completed=False), "nfl", 2024)
    assert g is None


def test_fetch_season_chunks_and_dedupes(monkeypatch):
    calls = []

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"events": [_event("1", "2024-10-06", "H", "A", 21, 14)]}

    def fake_get(url, params=None, timeout=None):
        calls.append(params["dates"])
        return FakeResp()

    monkeypatch.setattr(espn.requests, "get", fake_get)
    games = espn.fetch_season(NFL, 2024)
    assert len(calls) > 3            # several date chunks queried
    assert len(games) == 1           # same game in every chunk -> deduped


def test_todays_games(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"events": [_event("9", "2026-06-11", "Florida Panthers",
                                      "Edmonton Oilers", 0, 0, completed=False)]}

    monkeypatch.setattr(espn.requests, "get", lambda *a, **k: FakeResp())
    games = espn.todays_games(NHL, date(2026, 6, 11))
    assert games == [{"home_team": "Florida Panthers",
                      "away_team": "Edmonton Oilers", "date": "2026-06-11"}]


def test_current_season():
    assert current_season(MLB, date(2026, 6, 11)) == 2026
    assert current_season(MLB, date(2026, 2, 1)) == 2025
    assert current_season(NBA, date(2026, 1, 15)) == 2025   # 2025-26 season
    assert current_season(NBA, date(2025, 11, 1)) == 2025
    assert current_season(NFL, date(2026, 1, 4)) == 2025
    assert current_season(NHL, date(2026, 10, 20)) == 2026


def test_all_sports_dispatch_to_espn(monkeypatch):
    """Every sport (MLB included) routes through the single ESPN fetcher."""
    from sports_edge.config import SPORTS
    from sports_edge.data import espn as espn_mod, sources

    seen = []
    monkeypatch.setattr(espn_mod, "fetch_season",
                        lambda cfg, season: seen.append((cfg.key, "fetch")) or [])
    monkeypatch.setattr(espn_mod, "todays_games",
                        lambda cfg, day=None: seen.append((cfg.key, "today")) or [])
    for cfg in SPORTS.values():
        sources.fetch_season(cfg, 2024)
        sources.todays_games(cfg)
        assert cfg.espn_path  # every sport has an ESPN path configured
    assert {s for s, _ in seen} == {"mlb", "nba", "nfl", "nhl"}


def test_ensure_source_purges_on_change(tmp_path):
    """Switching data sources must clear a sport's games: game ids are
    source-specific, so mixing sources would duplicate every game."""
    from sports_edge.data.store import Game, GameStore

    store = GameStore(tmp_path / "g.sqlite")
    store.upsert_games([Game("nba", "0022300001", "2024-01-01", 2023,
                             "Boston Celtics", "Miami Heat", 110, 100)])
    # Legacy database: games exist but no recorded source -> purge.
    assert store.ensure_source("nba", "espn") is True
    assert store.count("nba") == 0
    # Same source from then on -> no purge, data is kept.
    store.upsert_games([Game("nba", "401585601", "2024-01-02", 2023,
                             "Boston Celtics", "Miami Heat", 99, 101)])
    assert store.ensure_source("nba", "espn") is False
    assert store.count("nba") == 1
    # Other sports are untouched by a purge.
    store.upsert_games([Game("mlb", "1", "2024-05-01", 2024,
                             "Athletics", "Seattle Mariners", 4, 2)])
    store.ensure_source("mlb", "espn")
    assert store.count("nba") == 1
    store.close()


def test_update_history_records_source(tmp_path, monkeypatch):
    from sports_edge.config import NHL as NHL_CFG
    from sports_edge.data import sources
    from sports_edge.data.store import GameStore

    store = GameStore(tmp_path / "g.sqlite")
    monkeypatch.setattr(sources, "fetch_season", lambda cfg, season: [])
    sources.update_history(NHL_CFG, store, today=date(2026, 1, 10))
    assert store.ensure_source("nhl", "espn") is False  # already recorded
    store.close()


def test_state_merges_team_name_spellings():
    """History from stats.nba.com and a slate from ESPN must hit the same
    team state despite different spellings."""
    from sports_edge.config import NBA as NBA_CFG
    from sports_edge.data.store import Game
    from sports_edge.features.state import LeagueState

    state = LeagueState(NBA_CFG)
    for i in range(3):
        state.update(Game("nba", str(i), f"2024-01-0{i + 1}", 2023,
                          "Los Angeles Clippers", "Boston Celtics", 110, 100))
    feats = state.features_for("LA Clippers", "Boston Celtics", "2024-01-05")
    assert feats["elo_home"] > 1500  # the Clippers' wins are visible
    assert feats["games_played_home"] == 3
    assert state.team_summary("LA Clippers")["last_n"] == "3-0"


def test_team_key_normalization():
    assert team_key("Montréal Canadiens") == team_key("Montreal Canadiens")
    assert team_key("St. Louis Blues") == team_key("St Louis Blues")
    assert team_key("LA Clippers") == team_key("Los Angeles Clippers")
    assert team_key("Oakland Athletics") == team_key("Athletics")
    assert team_key("Boston Celtics") == "boston celtics"
