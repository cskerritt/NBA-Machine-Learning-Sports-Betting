"""Incremental league state: walks games chronologically and produces
pre-game features with no lookahead leakage. The same code path is used
for building training data and for predicting upcoming games."""

from collections import deque
from datetime import date, datetime

from sports_edge.config import SportConfig
from sports_edge.data.store import Game
from sports_edge.features.elo import Elo
from sports_edge.teams import team_key

FEATURE_NAMES = [
    "elo_home", "elo_away", "elo_diff", "elo_prob_home",
    "home_winpct", "away_winpct",
    "home_margin_avg", "away_margin_avg",
    "home_pf_avg", "away_pf_avg",          # rolling points scored
    "home_pa_avg", "away_pa_avg",          # rolling points allowed
    "home_home_winpct", "away_away_winpct",
    "home_season_winpct", "away_season_winpct",
    "home_streak", "away_streak",          # signed win/lose streak
    "rest_home", "rest_away", "rest_diff",
    "b2b_home", "b2b_away",
    "games_played_home", "games_played_away",
]


def parse_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


class _TeamForm:
    def __init__(self, window: int):
        self.results = deque(maxlen=window)        # (won, margin, scored, allowed)
        self.venue_results = {"home": deque(maxlen=window), "away": deque(maxlen=window)}
        self.last_game: date | None = None
        self.games_played = 0
        self.streak = 0                            # +n on a win streak, -n losing
        self.season_wins = 0
        self.season_games = 0

    def winpct(self, default=0.5) -> float:
        if not self.results:
            return default
        return sum(1 for r in self.results if r[0]) / len(self.results)

    def margin_avg(self) -> float:
        if not self.results:
            return 0.0
        return sum(r[1] for r in self.results) / len(self.results)

    def points_avg(self, idx: int) -> float:
        """idx 2 = scored, 3 = allowed."""
        if not self.results:
            return 0.0
        return sum(r[idx] for r in self.results) / len(self.results)

    def venue_winpct(self, venue: str, default=0.5) -> float:
        results = self.venue_results[venue]
        if not results:
            return default
        return sum(results) / len(results)

    def season_winpct(self, default=0.5) -> float:
        if not self.season_games:
            return default
        return self.season_wins / self.season_games

    def record(self, won: bool, margin: int, scored: int, allowed: int):
        self.results.append((won, margin, scored, allowed))
        if won:
            self.streak = self.streak + 1 if self.streak > 0 else 1
        else:
            self.streak = self.streak - 1 if self.streak < 0 else -1
        self.season_wins += 1 if won else 0
        self.season_games += 1
        self.games_played += 1

    def new_season(self):
        self.season_wins = 0
        self.season_games = 0
        self.streak = 0


class LeagueState:
    def __init__(self, cfg: SportConfig):
        self.cfg = cfg
        self.elo = Elo(cfg.elo_k, cfg.elo_home_advantage, cfg.elo_season_regression)
        self.forms: dict[str, _TeamForm] = {}
        self.current_season: int | None = None

    def _form(self, team: str) -> _TeamForm:
        # Forms and Elo are keyed by canonical name so that data sources
        # with different spellings (ESPN "LA Clippers" vs stats.nba.com
        # "Los Angeles Clippers") update the same team.
        key = team_key(team)
        if key not in self.forms:
            self.forms[key] = _TeamForm(self.cfg.rolling_window)
        return self.forms[key]

    def _rest_days(self, form: _TeamForm, game_date: date) -> int:
        if form.last_game is None:
            return self.cfg.rest_cap_days
        return min((game_date - form.last_game).days, self.cfg.rest_cap_days)

    def features_for(self, home: str, away: str, game_date: str) -> dict[str, float]:
        """Pre-game features. Must be called BEFORE update() for that game."""
        d = parse_date(game_date)
        home, away = team_key(home), team_key(away)
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
            "home_pf_avg": hf.points_avg(2),
            "away_pf_avg": af.points_avg(2),
            "home_pa_avg": hf.points_avg(3),
            "away_pa_avg": af.points_avg(3),
            "home_home_winpct": hf.venue_winpct("home"),
            "away_away_winpct": af.venue_winpct("away"),
            "home_season_winpct": hf.season_winpct(),
            "away_season_winpct": af.season_winpct(),
            "home_streak": float(hf.streak),
            "away_streak": float(af.streak),
            "rest_home": float(rest_h),
            "rest_away": float(rest_a),
            "rest_diff": float(rest_h - rest_a),
            "b2b_home": 1.0 if rest_h <= 1 else 0.0,
            "b2b_away": 1.0 if rest_a <= 1 else 0.0,
            "games_played_home": float(hf.games_played),
            "games_played_away": float(af.games_played),
        }

    def team_summary(self, team: str) -> dict:
        """Display-oriented snapshot used in reports (not model features)."""
        f = self._form(team)
        wins = sum(1 for r in f.results if r[0])
        ratings = sorted(self.elo.ratings.values(), reverse=True)
        rating = self.elo.rating(team_key(team))
        rank = 1 + sum(1 for r in ratings if r > rating) if self.elo.ratings else None
        return {
            "elo": round(rating, 1),
            "elo_rank": rank,
            "last_n": f"{wins}-{len(f.results) - wins}",
            "streak": (f"W{f.streak}" if f.streak > 0
                       else f"L{-f.streak}" if f.streak < 0 else "-"),
            "season_record": f"{f.season_wins}-{f.season_games - f.season_wins}",
            "ppg": round(f.points_avg(2), 1),
            "papg": round(f.points_avg(3), 1),
        }

    def rollover_if_new_season(self, season: int):
        """Apply between-season regression/resets. Call before computing
        features for the first games of a new season."""
        if self.current_season is not None and season != self.current_season:
            self.elo.new_season()
            for form in self.forms.values():
                form.new_season()
        self.current_season = season

    def update(self, game: Game):
        """Apply a finished game's result to the state."""
        self.rollover_if_new_season(game.season)

        self.elo.update(team_key(game.home_team), team_key(game.away_team),
                        game.home_score, game.away_score)

        d = parse_date(game.date)
        hf, af = self._form(game.home_team), self._form(game.away_team)
        hf.record(game.home_win, game.margin, game.home_score, game.away_score)
        af.record(not game.home_win, -game.margin, game.away_score, game.home_score)
        hf.venue_results["home"].append(1 if game.home_win else 0)
        af.venue_results["away"].append(0 if game.home_win else 1)
        hf.last_game = af.last_game = d
