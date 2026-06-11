"""Elo rating system with margin-of-victory adjustment (FiveThirtyEight-style)."""

MEAN_RATING = 1500.0


class Elo:
    def __init__(self, k: float, home_advantage: float, season_regression: float):
        self.k = k
        self.home_advantage = home_advantage
        self.season_regression = season_regression
        self.ratings: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.ratings.get(team, MEAN_RATING)

    def expected_home_win_prob(self, home: str, away: str) -> float:
        diff = self.rating(home) + self.home_advantage - self.rating(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    @staticmethod
    def _mov_multiplier(margin: int, winner_elo_diff: float) -> float:
        """Dampens K for blowouts by strong favorites (autocorrelation guard)."""
        return ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * winner_elo_diff)

    def update(self, home: str, away: str, home_score: int, away_score: int):
        p_home = self.expected_home_win_prob(home, away)
        home_won = home_score > away_score
        margin = home_score - away_score

        elo_diff = self.rating(home) + self.home_advantage - self.rating(away)
        winner_elo_diff = elo_diff if home_won else -elo_diff
        mult = self._mov_multiplier(margin, winner_elo_diff) if margin != 0 else 1.0

        shift = self.k * mult * ((1.0 if home_won else 0.0) - p_home)
        self.ratings[home] = self.rating(home) + shift
        self.ratings[away] = self.rating(away) - shift

    def new_season(self):
        """Regress every team toward the mean between seasons."""
        r = self.season_regression
        for team in self.ratings:
            self.ratings[team] = self.ratings[team] * (1 - r) + MEAN_RATING * r
