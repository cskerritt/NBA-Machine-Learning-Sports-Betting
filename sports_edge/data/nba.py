"""NBA data via stats.nba.com (history) and cdn.nba.com (today's slate).

Both nba.com hosts block many datacenter/cloud IPs (403) even with
browser-like headers, so every call transparently falls back to ESPN's
scoreboard API, which serves the same games. Team-name spelling differences
between the sources are reconciled by sports_edge.teams.team_key.
"""

from datetime import date

import requests

from sports_edge.data.store import Game

GAMELOG_URL = "https://stats.nba.com/stats/leaguegamelog"
SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
TIMEOUT = 60

# nba.com rejects requests without browser-like headers
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}
STATS_HEADERS = {
    **BROWSER_HEADERS,
    "Referer": "https://stats.nba.com/",
    "Origin": "https://stats.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def _espn():
    from sports_edge.config import NBA
    from sports_edge.data import espn
    return espn, NBA


def _season_str(season: int) -> str:
    """2023 -> '2023-24' (season identified by its starting year)."""
    return f"{season}-{(season + 1) % 100:02d}"


def fetch_season(season: int) -> list[Game]:
    """All finished regular-season games for a season (starting-year convention)."""
    try:
        return _fetch_season_stats_nba(season)
    except requests.RequestException:
        espn, cfg = _espn()
        return espn.fetch_season(cfg, season)


def _fetch_season_stats_nba(season: int) -> list[Game]:
    params = {
        "Counter": 0, "DateFrom": "", "DateTo": "", "Direction": "ASC",
        "LeagueID": "00", "PlayerOrTeam": "T", "Season": _season_str(season),
        "SeasonType": "Regular Season", "Sorter": "DATE",
    }
    resp = requests.get(GAMELOG_URL, params=params, headers=STATS_HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    result = resp.json()["resultSets"][0]
    cols = {name: i for i, name in enumerate(result["headers"])}
    rows = result["rowSet"]

    # Each game has two rows (one per team); MATCHUP 'vs.' marks the home row.
    by_game: dict[str, dict] = {}
    for row in rows:
        gid = row[cols["GAME_ID"]]
        side = "home" if "vs." in row[cols["MATCHUP"]] else "away"
        by_game.setdefault(gid, {"date": row[cols["GAME_DATE"]]})[side] = {
            "team": row[cols["TEAM_NAME"]],
            "pts": row[cols["PTS"]],
        }

    games = []
    for gid, g in by_game.items():
        if "home" not in g or "away" not in g:
            continue
        if g["home"]["pts"] is None or g["away"]["pts"] is None:
            continue  # game not finished
        games.append(Game(
            sport="nba",
            game_id=gid,
            date=g["date"][:10],
            season=season,
            home_team=g["home"]["team"],
            away_team=g["away"]["team"],
            home_score=int(g["home"]["pts"]),
            away_score=int(g["away"]["pts"]),
        ))
    games.sort(key=lambda g: (g.date, g.game_id))
    return games


def todays_games(day: date | None = None) -> list[dict]:
    """Today's NBA slate, from the nba.com CDN with ESPN fallback."""
    try:
        resp = requests.get(SCOREBOARD_URL, headers=BROWSER_HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        espn, cfg = _espn()
        return espn.todays_games(cfg, day)
    board = resp.json().get("scoreboard", {})
    games = []
    for g in board.get("games", []):
        home, away = g["homeTeam"], g["awayTeam"]
        games.append({
            "home_team": f"{home['teamCity']} {home['teamName']}",
            "away_team": f"{away['teamCity']} {away['teamName']}",
            "date": board.get("gameDate", date.today().isoformat()),
        })
    return games
