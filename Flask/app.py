import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import date
from functools import lru_cache

from flask import Flask, render_template

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ALLOWED_SPORTSBOOKS = frozenset({
    "fanduel", "draftkings", "betmgm", "pointsbet", "caesars", "wynn", "bet_rivers_ny",
})
SUBPROCESS_TIMEOUT_SECONDS = 60
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

DATA_RE = re.compile(
    r'\n(?P<home_team>[\w ]+)(\((?P<home_confidence>[\d+\.]+)%\))? vs '
    r'(?P<away_team>[\w ]+)(\((?P<away_confidence>[\d+\.]+)%\))?: '
    r'(?P<ou_pick>OVER|UNDER) (?P<ou_value>[\d+\.]+) '
    r'(\((?P<ou_confidence>[\d+\.]+)%\))?',
    re.MULTILINE,
)
EV_RE = re.compile(r'(?P<team>[\w ]+) EV: (?P<ev>[-\d+\.]+)', re.MULTILINE)
ODDS_RE = re.compile(
    r'(?P<away_team>[\w ]+) \((?P<away_team_odds>-?\d+)\) @ '
    r'(?P<home_team>[\w ]+) \((?P<home_team_odds>-?\d+)\)',
    re.MULTILINE,
)


@lru_cache()
def fetch_fanduel(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="fanduel")


@lru_cache()
def fetch_draftkings(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="draftkings")


@lru_cache()
def fetch_betmgm(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="betmgm")


def fetch_game_data(sportsbook="fanduel"):
    if sportsbook not in ALLOWED_SPORTSBOOKS:
        raise ValueError(f"Unsupported sportsbook: {sportsbook!r}")

    cmd = [sys.executable, "main.py", "-xgb", f"-odds={sportsbook}"]
    try:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        logger.exception("main.py timed out for sportsbook=%s", sportsbook)
        return {}
    except subprocess.CalledProcessError as exc:
        logger.exception("main.py failed for sportsbook=%s: %s", sportsbook, exc.stderr)
        return {}

    stdout = completed.stdout
    ev_matches = list(EV_RE.finditer(stdout))
    odds_matches = list(ODDS_RE.finditer(stdout))

    games = {}
    for match in DATA_RE.finditer(stdout):
        game_dict = {
            'away_team': match.group('away_team').strip(),
            'home_team': match.group('home_team').strip(),
            'away_confidence': match.group('away_confidence'),
            'home_confidence': match.group('home_confidence'),
            'ou_pick': match.group('ou_pick'),
            'ou_value': match.group('ou_value'),
            'ou_confidence': match.group('ou_confidence'),
        }
        for ev_match in ev_matches:
            team = ev_match.group('team')
            if team == game_dict['away_team']:
                game_dict['away_team_ev'] = ev_match.group('ev')
            elif team == game_dict['home_team']:
                game_dict['home_team_ev'] = ev_match.group('ev')
        for odds_match in odds_matches:
            if odds_match.group('away_team') == game_dict['away_team']:
                game_dict['away_team_odds'] = odds_match.group('away_team_odds')
            if odds_match.group('home_team') == game_dict['home_team']:
                game_dict['home_team_odds'] = odds_match.group('home_team_odds')

        logger.debug(json.dumps(game_dict, sort_keys=True))
        games[f"{game_dict['away_team']}:{game_dict['home_team']}"] = game_dict
    return games


def get_ttl_hash(seconds=600):
    """Return the same value within `seconds` time period."""
    return round(time.time() / seconds)


app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')


@app.route("/")
def index():
    fanduel = fetch_fanduel(ttl_hash=get_ttl_hash())
    draftkings = fetch_draftkings(ttl_hash=get_ttl_hash())
    betmgm = fetch_betmgm(ttl_hash=get_ttl_hash())

    return render_template(
        'index.html',
        today=date.today(),
        data={"fanduel": fanduel, "draftkings": draftkings, "betmgm": betmgm},
    )
