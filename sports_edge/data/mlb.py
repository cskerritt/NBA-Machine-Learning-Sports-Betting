"""MLB data via the free MLB Stats API (statsapi.mlb.com). No API key required."""

from datetime import date

import requests

from sports_edge.data.store import Game

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
TIMEOUT = 30


def _season_games_raw(season: int) -> list[dict]:
    params = {
        "sportId": 1,
        "gameTypes": "R",  # regular season
        "startDate": f"{season}-03-01",
        "endDate": f"{season}-11-15",
    }
    resp = requests.get(SCHEDULE_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    games = []
    for day in resp.json().get("dates", []):
        games.extend(day.get("games", []))
    return games


def fetch_season(season: int) -> list[Game]:
    """All finished regular-season games for a season."""
    out = []
    for g in _season_games_raw(season):
        status = g.get("status", {}).get("codedGameState")
        teams = g.get("teams", {})
        home, away = teams.get("home", {}), teams.get("away", {})
        if status != "F" or "score" not in home or "score" not in away:
            continue
        out.append(Game(
            sport="mlb",
            game_id=str(g["gamePk"]),
            date=g["officialDate"],
            season=season,
            home_team=home["team"]["name"],
            away_team=away["team"]["name"],
            home_score=home["score"],
            away_score=away["score"],
        ))
    return out


def todays_games(day: date | None = None) -> list[dict]:
    """Scheduled games for a date: [{'home_team', 'away_team', 'date'}, ...]."""
    day = day or date.today()
    params = {"sportId": 1, "date": day.isoformat()}
    resp = requests.get(SCHEDULE_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    games = []
    for d in resp.json().get("dates", []):
        for g in d.get("games", []):
            teams = g.get("teams", {})
            games.append({
                "home_team": teams["home"]["team"]["name"],
                "away_team": teams["away"]["team"]["name"],
                "date": g.get("officialDate", day.isoformat()),
            })
    return games
