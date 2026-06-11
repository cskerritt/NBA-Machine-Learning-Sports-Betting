"""Moneyline odds from the-odds-api.com (free key: https://the-odds-api.com).

Set THE_ODDS_API_KEY in your environment, or pass --api-key on the CLI.
"""

import os

import requests

from sports_edge.config import SportConfig

BASE_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
TIMEOUT = 30

# Map odds-api team names to the names used by our data sources.
TEAM_ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "Oakland Athletics": "Athletics",
    "Los Angeles Angels of Anaheim": "Los Angeles Angels",
}


def normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def fetch_moneylines(cfg: SportConfig, api_key: str | None = None,
                     bookmaker: str | None = None) -> dict[tuple[str, str], dict]:
    """Best available (or single-book) moneylines for upcoming games.

    Returns {(home_team, away_team): {"home_ml": -150, "away_ml": +130,
    "home_book": "fanduel", "away_book": "draftkings"}}.
    When `bookmaker` is None, the best price across US books is used per side.
    """
    api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No odds API key. Get a free key at https://the-odds-api.com and "
            "set THE_ODDS_API_KEY, or pass --api-key."
        )
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }
    if bookmaker:
        params["bookmakers"] = bookmaker
    resp = requests.get(BASE_URL.format(sport=cfg.odds_api_key), params=params,
                        timeout=TIMEOUT)
    resp.raise_for_status()

    odds = {}
    for event in resp.json():
        home = normalize_team(event["home_team"])
        away = normalize_team(event["away_team"])
        best = {"home_ml": None, "away_ml": None, "home_book": None, "away_book": None}
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market["outcomes"]:
                    team = normalize_team(outcome["name"])
                    price = outcome["price"]
                    if team == home and (best["home_ml"] is None or price > best["home_ml"]):
                        best["home_ml"], best["home_book"] = price, book["key"]
                    elif team == away and (best["away_ml"] is None or price > best["away_ml"]):
                        best["away_ml"], best["away_book"] = price, book["key"]
        if best["home_ml"] is not None and best["away_ml"] is not None:
            odds[(home, away)] = best
    return odds
