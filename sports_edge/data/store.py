"""SQLite storage for historical game results."""

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sports_edge.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    sport      TEXT NOT NULL,
    game_id    TEXT NOT NULL,
    date       TEXT NOT NULL,         -- ISO yyyy-mm-dd
    season     INTEGER NOT NULL,
    home_team  TEXT NOT NULL,
    away_team  TEXT NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    PRIMARY KEY (sport, game_id)
);
CREATE INDEX IF NOT EXISTS idx_games_sport_date ON games (sport, date);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Game:
    sport: str
    game_id: str
    date: str
    season: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int

    @property
    def home_win(self) -> bool:
        return self.home_score > self.away_score

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score


class GameStore:
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(SCHEMA)

    def upsert_games(self, games: list[Game]) -> int:
        rows = [
            (g.sport, g.game_id, g.date, g.season, g.home_team, g.away_team,
             g.home_score, g.away_score)
            for g in games
        ]
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?)", rows
            )
        return len(rows)

    def load_games(self, sport: str, before: date | None = None) -> list[Game]:
        """All finished games for a sport, oldest first."""
        query = "SELECT * FROM games WHERE sport = ?"
        params: list = [sport]
        if before is not None:
            query += " AND date < ?"
            params.append(before.isoformat())
        query += " ORDER BY date, game_id"
        cur = self.conn.execute(query, params)
        return [Game(*row) for row in cur.fetchall()]

    def seasons(self, sport: str) -> list[int]:
        cur = self.conn.execute(
            "SELECT DISTINCT season FROM games WHERE sport = ? ORDER BY season", (sport,)
        )
        return [r[0] for r in cur.fetchall()]

    def count(self, sport: str) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM games WHERE sport = ?", (sport,))
        return cur.fetchone()[0]

    def ensure_source(self, sport: str, source: str) -> bool:
        """Record which data source a sport's games came from. If it changed
        (game ids are source-specific, so mixing would duplicate games),
        purge that sport's games for a clean refetch. Returns True if purged."""
        cur = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (f"source:{sport}",))
        row = cur.fetchone()
        purged = False
        with self.conn:
            if row is None or row[0] != source:
                if self.count(sport):
                    self.conn.execute("DELETE FROM games WHERE sport = ?", (sport,))
                    purged = True
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta VALUES (?, ?)",
                    (f"source:{sport}", source))
        return purged

    def close(self):
        self.conn.close()
