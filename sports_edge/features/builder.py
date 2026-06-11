"""Build a leakage-free training matrix by replaying history chronologically."""

import pandas as pd

from sports_edge.config import SportConfig
from sports_edge.data.store import Game
from sports_edge.features.state import FEATURE_NAMES, LeagueState


def build_dataset(games: list[Game], cfg: SportConfig) -> pd.DataFrame:
    """One row per game with pre-game features and the home_win label.

    Games where either team has too little history (season ramp-up for new
    state) are kept but flagged via `warm`, so training can filter them.
    """
    state = LeagueState(cfg)
    rows = []
    for g in games:
        state.rollover_if_new_season(g.season)
        feats = state.features_for(g.home_team, g.away_team, g.date)
        feats.update({
            "date": g.date,
            "season": g.season,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_win": 1 if g.home_win else 0,
            "margin": g.margin,
            "total": g.home_score + g.away_score,
            "warm": int(
                feats["games_played_home"] >= cfg.min_history_games
                and feats["games_played_away"] >= cfg.min_history_games
            ),
        })
        rows.append(feats)
        state.update(g)
    df = pd.DataFrame(rows)
    return df[["date", "season", "home_team", "away_team",
               "home_win", "margin", "total", "warm"] + FEATURE_NAMES]


def replay_state(games: list[Game], cfg: SportConfig) -> LeagueState:
    """Current league state after applying all historical games."""
    state = LeagueState(cfg)
    for g in games:
        state.update(g)
    return state
