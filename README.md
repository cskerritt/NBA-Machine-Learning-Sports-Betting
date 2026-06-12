# Sports Edge ⚾🏀🏈🏒

Machine-learning predictions for **MLB, NBA, NFL, and NHL** games, built for
finding positive-expected-value bets. The pipeline fetches historical results
from free public APIs, builds leakage-free features, trains calibrated models
per sport, and compares its win probabilities against live sportsbook
moneylines to surface +EV bets with Kelly-criterion stake sizing — with full
automation for a daily run.

> This repo was previously `NBA-Machine-Learning-Sports-Betting` (NBA only,
> TensorFlow 2.6 / Python 3.8). It has been rewritten from scratch. The old
> code lives in git history (`master`). Feel free to rename the repo to
> `sports-edge` under GitHub → Settings.

## What you get per game

- **Pick + win probability** from a blend of a calibrated XGBoost classifier
  and an Elo baseline (blend weight tuned on holdout log loss)
- **Projected final score** — separate margin and total regressors
- **Confidence tier** (STRONG / LEAN / PASS) based on the model's edge over
  the market's no-vig probability
- **Team form** — Elo rating and league rank, last-N record, streak, season
  record, points for/against
- **Betting analysis** — best moneyline across US books (or one book of your
  choice), market implied probability with the vig removed, expected value
  per $100, and capped fractional-Kelly stakes for your bankroll

## How it works

1. **Data** — MLB from the MLB Stats API, NBA from stats.nba.com, NFL & NHL
   from ESPN's public scoreboard API. All free, no keys. Results land in a
   local SQLite database; daily runs only re-download the in-progress season.
2. **Features** — games are replayed chronologically through a `LeagueState`:
   Elo ratings (FiveThirtyEight-style margin-of-victory multiplier, per-sport
   K-factor and home advantage, between-season regression), rolling win% and
   scoring averages, venue splits, season record, win/loss streaks, rest days,
   and back-to-back flags. Features are always computed *before* a game's
   result (and before season rollovers), so there is no lookahead leakage.
3. **Models** — per sport: an isotonic-calibrated XGBoost win classifier
   blended with Elo, plus margin and total regressors, evaluated on a
   time-ordered holdout (most recent 15% of games).
4. **Betting** — live moneylines via [The Odds API](https://the-odds-api.com)
   (free key), robust team-name matching across sources, EV and quarter-Kelly
   sizing capped at 10% of bankroll.
5. **Backtest** — walk-forward by season (train only on prior seasons), with
   an optional odds CSV to simulate real betting ROI.

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# 1. Download history (defaults to recent seasons)
python main.py fetch --sport mlb        # also: nba, nfl, nhl

# 2. Train models
python main.py train --sport mlb

# 3. Predict today's games with live odds and bet sizing
export THE_ODDS_API_KEY=your_key        # free at https://the-odds-api.com
python main.py predict --sport mlb --bankroll 1000
python main.py predict --sport nhl --book fanduel
python main.py predict --sport nba --no-odds   # probabilities only, no key needed
```

Example output:

```
Miami Heat @ Boston Celtics
  Pick:  Boston Celtics (64.2%)  [STRONG]  (model 65.0% / elo 61.8%, home side)
  Score: Boston Celtics 112.3 - 106.1 Miami Heat  (margin +6.2, total 218.4)
  Form:  Boston Celtics: Elo 1671.2 (#2), last 8-2, streak W4, season 52-18, 117.4/109.2 for/against
  Form:  Miami Heat: Elo 1524.8 (#14), last 5-5, streak L1, season 38-32, 110.1/110.8 for/against
  Odds:  Boston Celtics -150 (fanduel)  |  Miami Heat +135 (draftkings)  [market home prob 58.6%, edge +5.6%]
  +EV:  Boston Celtics  EV $+7.00/100  Kelly 3.5% of bankroll ($35)
```

## Daily automation

One command updates data, retrains, predicts every in-season sport, and writes
JSON + Markdown reports to `reports/<date>/` (and `reports/latest/`):

```bash
python main.py daily                       # all four sports
python main.py daily --sports mlb,nba      # subset
```

**GitHub Actions** (included): `.github/workflows/daily-predictions.yml` runs
every day at 14:00 UTC, caches the game database between runs, writes the
reports into the repo, and publishes them to the workflow job summary. To
enable it, add your The Odds API key as a repository secret named
`THE_ODDS_API_KEY` (Settings → Secrets and variables → Actions). You can also
trigger it manually from the Actions tab.

**Local cron** alternative:

```cron
0 9 * * * cd /path/to/repo && python main.py daily >> daily.log 2>&1
```

## Backtesting

```bash
python main.py backtest --sport nba
python main.py backtest --sport mlb --odds-csv my_odds.csv --min-ev 3
```

The odds CSV needs columns `date,home_team,away_team,home_ml,away_ml`
(American odds). Without it, the backtest reports accuracy / log-loss / Brier
score per season versus the Elo baseline; with it, it also simulates flat $100
bets on every side clearing the `--min-ev` threshold and reports profit and ROI.

## Web dashboard

```bash
python main.py web            # http://127.0.0.1:5000
```

A dark-mode dashboard for all four sports: win-probability bars, projected
scores, confidence tiers, team form, live odds with the best book per side,
market edge, and highlighted +EV bets sized to your bankroll.

## Project layout

```
main.py                     CLI (fetch / train / predict / backtest / daily / web)
sports_edge/
  config.py                 per-sport Elo + feature parameters, season calendar
  data/                     MLB, NBA, ESPN (NFL/NHL) fetchers; SQLite store
  features/                 Elo engine, incremental LeagueState, dataset builder
  models/                   training (calibrated XGBoost + Elo blend, score
                            regressors) and prediction
  betting/                  odds math (EV, no-vig, Kelly) + The Odds API client
  backtest.py               walk-forward season backtests
  daily.py / reports.py     daily pipeline and JSON/Markdown report writer
web/                        Flask dashboard
.github/workflows/          scheduled daily-predictions workflow
tests/                      38 tests on a synthetic league (no network needed)
```

## Running tests

```bash
pip install pytest
python -m pytest tests/
```

## Honest expectations & responsible gambling

Sportsbook closing lines are extremely efficient. Realistic moneyline accuracy
is roughly: NBA mid-60s%, NFL low-60s%, NHL high-50s%, MLB high-50s% — about
on par with the market itself. **Beating the vig consistently is hard.** Treat
the EV output as a screening tool, validate with backtests against real odds
before risking money, prefer quarter-Kelly or less, and never bet more than
you can afford to lose.

If gambling stops being fun: **1-800-GAMBLER** (US) or https://www.ncpgambling.org.
