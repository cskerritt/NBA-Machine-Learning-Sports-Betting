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


def test_nba_todays_games_falls_back_to_espn(monkeypatch):
    """A 403 from cdn.nba.com (datacenter IP block) must fall through to ESPN."""
    import requests as requests_lib

    from sports_edge.data import espn as espn_mod, nba

    def blocked(*a, **k):
        raise requests_lib.HTTPError("403 Client Error: Forbidden")

    monkeypatch.setattr(nba.requests, "get", blocked)
    monkeypatch.setattr(
        espn_mod, "todays_games",
        lambda cfg, day=None: [{"home_team": "Boston Celtics",
                                "away_team": "Miami Heat", "date": "2026-06-12"}])
    games = nba.todays_games(date(2026, 6, 12))
    assert games[0]["home_team"] == "Boston Celtics"


def test_nba_fetch_season_falls_back_to_espn(monkeypatch):
    import requests as requests_lib

    from sports_edge.data import espn as espn_mod, nba

    def blocked(*a, **k):
        raise requests_lib.HTTPError("403 Client Error: Forbidden")

    monkeypatch.setattr(nba.requests, "get", blocked)
    sentinel = []
    monkeypatch.setattr(espn_mod, "fetch_season", lambda cfg, season: sentinel)
    assert nba.fetch_season(2024) is sentinel


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
