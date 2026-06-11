"""Synthetic league generator for testing the full pipeline without network."""

import math
import random
from datetime import date, timedelta

from sports_edge.data.store import Game

TEAMS = [f"Team {c}" for c in "ABCDEFGHIJKL"]


def synthetic_seasons(seasons: list[int], games_per_pair: int = 4,
                      seed: int = 7) -> list[Game]:
    """Round-robin league where latent team strengths drive outcomes."""
    rng = random.Random(seed)
    strengths = {t: rng.gauss(0, 1) for t in TEAMS}
    home_edge = 0.35  # logit bump for home team

    games = []
    gid = 0
    for season in seasons:
        day = date(season, 1, 1)
        matchups = []
        for _ in range(games_per_pair):
            for i, home in enumerate(TEAMS):
                for away in TEAMS[i + 1:]:
                    matchups.append((home, away) if rng.random() < 0.5 else (away, home))
        rng.shuffle(matchups)
        for home, away in matchups:
            p_home = 1 / (1 + math.exp(-(strengths[home] - strengths[away] + home_edge)))
            home_won = rng.random() < p_home
            margin = max(1, round(abs(rng.gauss(8, 5))))
            base = rng.randint(95, 115)
            home_score = base + (margin if home_won else 0)
            away_score = base + (0 if home_won else margin)
            gid += 1
            games.append(Game(
                sport="test", game_id=str(gid), date=day.isoformat(), season=season,
                home_team=home, away_team=away,
                home_score=home_score, away_score=away_score,
            ))
            if gid % 4 == 0:
                day += timedelta(days=1)
    return games
