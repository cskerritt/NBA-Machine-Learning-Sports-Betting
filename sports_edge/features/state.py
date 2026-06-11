"""Incremental league state: walks games chronologically and produces
pre-game features with no lookahead leakage. The same code path is used
for building training data and for predicting upcoming games."""

from collections import deque
from datetime import date, datetime

from sports_edge.config import SportConfig
from sports_edge.data.store import Game
from sports_edge.features.elo import Elo

FEATURE_NAMES = [
    "elo_home", "elo_away", "elo_diff", "elo_prob_home",
    "home_winpct", "away_winpct",
    "home_margin_avg", "away_margin_avg",
    "home_home_winpct", "away_away_winpct",
    "rest_home", "rest_away",
    "b2b_home", "b2b_away",
    "games_played_home", "games_played_away",
]


def _parse_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


class _TeamForm:
    def __init__(self, window: int):
        self.results = deque(maxlen=window)        # (won: bool, margin: int)
        self.venue_results = {"home": deque(maxlen=window), "away": deque(maxlen=window)}
        self.last_game: date | None = None
        self.games_played = 0

    def winpct(self, default=0.5) -> float:
        if not self.results:
            return default
        return sum(1 for won, _ in self.results if won) / len(self.results)

    def margin_avg(self) -> float:
        if not self.results:
            return 0.0
        return sum(m for _, m in self.results) / len(self.results)

    def venue_winpct(self, venue: str, default=0.5) -> float:
        results = self.venue_results[venue]
        if not results:
            return default
        return sum(results) / len(results)


class LeagueState:
    def __init__(self, cfg: SportConfig):
        self.cfg = cfg
        self.elo = Elo(cfg.elo_k, cfg.elo_home_advantage, cfg.elo_season_regression)
        self.forms: dict[str, _TeamForm] = {}
        self.current_season: int | None = None

    def _form(self, team: str) -> _TeamForm:
        if team not in self.forms:
            self.forms[team] = _TeamForm(self.cfg.rolling_window)
        return self.forms[team]

    def _rest_days(self, form: _TeamForm, game_date: date) -> int:
        if form.last_game is None:
            return self.cfg.rest_cap_days
        return min((game_date - form.last_game).days, self.cfg.rest_cap_days)

    def features_for(self, home: str, away: str, game_date: str) -> dict[str, float]:
        """Pre-game features. Must be called BEFORE update() for that game."""
        d = _parse_date(game_date)
        hf, af = self._form(home), self._form(away)
        rest_h, rest_a = self._rest_days(hf, d), self._rest_days(af, d)
        return {
            "elo_home": self.elo.rating(home),
            "elo_away": self.elo.rating(away),
            "elo_diff": self.elo.rating(home) - self.elo.rating(away),
            "elo_prob_home": self.elo.expected_home_win_prob(home, away),
            "home_winpct": hf.winpct(),
            "away_winpct": af.winpct(),
            "home_margin_avg": hf.margin_avg(),
            "away_margin_avg": af.margin_avg(),
            "home_home_winpct": hf.venue_winpct("home"),
            "away_away_winpct": af.venue_winpct("away"),
            "rest_home": float(rest_h),
            "rest_away": float(rest_a),
            "b2b_home": 1.0 if rest_h <= 1 else 0.0,
            "b2b_away": 1.0 if rest_a <= 1 else 0.0,
            "games_played_home": float(hf.games_played),
            "games_played_away": float(af.games_played),
        }

    def update(self, game: Game):
        """Apply a finished game's result to the state."""
        if self.current_season is not None and game.season != self.current_season:
            self.elo.new_season()
        self.current_season = game.season

        self.elo.update(game.home_team, game.away_team, game.home_score, game.away_score)

        d = _parse_date(game.date)
        hf, af = self._form(game.home_team), self._form(game.away_team)
        hf.results.append((game.home_win, game.margin))
        af.results.append((not game.home_win, -game.margin))
        hf.venue_results["home"].append(1 if game.home_win else 0)
        af.venue_results["away"].append(0 if game.home_win else 1)
        hf.last_game = af.last_game = d
        hf.games_played += 1
        af.games_played += 1
