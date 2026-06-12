"""Game data for every sport via ESPN's public scoreboard API.

One source for MLB, NBA, NFL, and NHL: no API key, identical response
shapes across leagues, consistent team names, and — unlike nba.com —
no blocking of datacenter/cloud IPs.
"""

from datetime import date, timedelta

import requests

from sports_edge.config import SportConfig
from sports_edge.data.store import Game

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
TIMEOUT = 30


def _date_chunks(start: date, end: date, days: int = 28):
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def _fetch_events(path: str, start: date, end: date) -> list[dict]:
    params = {
        "dates": f"{start:%Y%m%d}-{end:%Y%m%d}",
        "limit": 1000,  # a 4-week MLB chunk has ~450 games; leave headroom
        "seasontype": 2,  # regular season
    }
    resp = requests.get(SCOREBOARD_URL.format(path=path), params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("events", [])


def _parse_event(event: dict, sport_key: str, season: int) -> Game | None:
    comp = event.get("competitions", [{}])[0]
    if not comp.get("status", event.get("status", {})).get("type", {}).get("completed"):
        return None
    sides = {}
    for c in comp.get("competitors", []):
        score = c.get("score")
        if score in (None, ""):
            return None
        sides[c["homeAway"]] = (c["team"]["displayName"], int(float(score)))
    if set(sides) != {"home", "away"}:
        return None
    return Game(
        sport=sport_key,
        game_id=str(event["id"]),
        date=event["date"][:10],
        season=season,
        home_team=sides["home"][0],
        away_team=sides["away"][0],
        home_score=sides["home"][1],
        away_score=sides["away"][1],
    )


def fetch_season(cfg: SportConfig, season: int) -> list[Game]:
    """All finished regular-season games for a season (starting-year convention)."""
    (sm, sd), (em, ed) = cfg.season_window
    start = date(season, sm, sd)
    end = date(season if em >= sm else season + 1, em, ed)
    end = min(end, date.today())

    games: dict[str, Game] = {}
    for chunk_start, chunk_end in _date_chunks(start, end):
        for event in _fetch_events(cfg.espn_path, chunk_start, chunk_end):
            g = _parse_event(event, cfg.key, season)
            if g:
                games[g.game_id] = g
    out = sorted(games.values(), key=lambda g: (g.date, g.game_id))
    return out


def todays_games(cfg: SportConfig, day: date | None = None) -> list[dict]:
    """Scheduled games for a date: [{'home_team', 'away_team', 'date'}, ...]."""
    day = day or date.today()
    params = {"dates": f"{day:%Y%m%d}", "limit": 100}
    resp = requests.get(SCOREBOARD_URL.format(path=cfg.espn_path), params=params,
                        timeout=TIMEOUT)
    resp.raise_for_status()
    games = []
    for event in resp.json().get("events", []):
        comp = event.get("competitions", [{}])[0]
        sides = {c["homeAway"]: c["team"]["displayName"]
                 for c in comp.get("competitors", [])}
        if set(sides) != {"home", "away"}:
            continue
        games.append({
            "home_team": sides["home"],
            "away_team": sides["away"],
            "date": event["date"][:10],
        })
    return games
