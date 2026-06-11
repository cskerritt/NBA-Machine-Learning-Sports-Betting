# Sports Edge ⚾🏀

Machine-learning predictions for **MLB** and **NBA** games, built for finding
positive-expected-value bets. The pipeline fetches historical results from free
public APIs, builds leakage-free features (margin-of-victory-adjusted Elo
ratings, rolling form, rest/back-to-backs, home-away splits), trains a
calibrated gradient-boosted model per sport, and compares its win probabilities
against live sportsbook moneylines to surface +EV bets with Kelly-criterion
stake sizing.

> This repo was previously `NBA-Machine-Learning-Sports-Betting` (NBA only,
> TensorFlow 2.6 / Python 3.8). It has been rewritten from scratch for both
> sports on modern Python. The old code lives in git history (`master`).
> Feel free to rename the repo to `sports-edge` under GitHub → Settings.

## How it works

1. **Fetch** — game results come from the free MLB Stats API
   (`statsapi.mlb.com`) and the NBA stats API (`stats.nba.com`), stored in a
   local SQLite database. No API keys needed.
2. **Features** — games are replayed chronologically through a `LeagueState`
   that maintains Elo ratings (FiveThirtyEight-style MOV multiplier, per-sport
   K-factor and home advantage, between-season regression) plus rolling win%,
   scoring margin, venue splits, rest days, and back-to-back flags. Features
   are always computed *before* a game's result is applied, so there is no
   lookahead leakage.
3. **Train** — an XGBoost classifier with isotonic calibration, evaluated on a
   time-ordered holdout (most recent 15% of games) against a pure-Elo baseline.
4. **Predict** — today's slate is pulled from the league APIs, current league
   state is replayed, and the model emits home-win probabilities. With a free
   [The Odds API](https://the-odds-api.com) key, live moneylines from US books
   (FanDuel, DraftKings, BetMGM, Caesars, ...) are attached and the app
   computes the market's no-vig probability, expected value per $100, and a
   capped fractional-Kelly stake.
5. **Backtest** — walk-forward by season (train only on prior seasons), with an
   optional odds CSV to simulate real betting ROI.

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# 1. Download history (defaults to the last ~5 seasons)
python main.py fetch --sport mlb
python main.py fetch --sport nba          # or --seasons 2021,2022,2023,2024

# 2. Train a model per sport
python main.py train --sport mlb
python main.py train --sport nba

# 3. Predict today's games with live odds and bet sizing
export THE_ODDS_API_KEY=your_key          # free at https://the-odds-api.com
python main.py predict --sport mlb --bankroll 1000
python main.py predict --sport nba --book fanduel
python main.py predict --sport nba --no-odds   # probabilities only, no key needed
```

Example output:

```
NBA predictions for 2026-06-11
==============================================================================

Miami Heat @ Boston Celtics
  Model: Boston Celtics (64.2%)   [home 64.2%, elo 61.8%]
  Odds:  Boston Celtics -150 (fanduel)  |  Miami Heat +135 (draftkings)  [market home prob 58.6%]
  +EV:  Boston Celtics  EV $+7.00/100  Kelly 3.5% of bankroll ($35)
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

A dark-mode dashboard showing today's slate for either sport: win-probability
bars, live odds with best book per side, the market's implied probability, and
highlighted +EV bets with suggested stakes for your bankroll.

## Project layout

```
main.py                     CLI entry point (fetch / train / predict / backtest / web)
sports_edge/
  config.py                 per-sport Elo + feature parameters
  data/                     MLB & NBA fetchers, SQLite game store
  features/                 Elo engine, incremental LeagueState, dataset builder
  models/                   training (calibrated XGBoost) and prediction
  betting/                  odds math (EV, no-vig, Kelly) + The Odds API client
  backtest.py               walk-forward season backtests
web/                        Flask dashboard
tests/                      full test suite on a synthetic league (no network)
```

## Running tests

```bash
pip install pytest
python -m pytest tests/
```

## Honest expectations & responsible gambling

Sportsbook closing lines are extremely efficient. A model like this can get
moneyline accuracy in the mid-60s for NBA and high-50s for MLB — roughly on par
with the market — but **beating the vig consistently is hard**. Treat the EV
output as a screening tool, validate with backtests against real odds before
risking money, prefer quarter-Kelly or less, and never bet more than you can
afford to lose.

If gambling stops being fun: **1-800-GAMBLER** (US) or https://www.ncpgambling.org.
