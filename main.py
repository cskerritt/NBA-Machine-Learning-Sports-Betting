"""Sports Edge CLI: fetch data, train models, predict games, backtest.

Examples:
    python main.py fetch    --sport mlb
    python main.py train    --sport mlb
    python main.py predict  --sport mlb --bankroll 1000
    python main.py backtest --sport nba --odds-csv odds.csv
    python main.py web
"""

import argparse
import sys

from colorama import Fore, Style, init as colorama_init

from sports_edge.config import get_sport
from sports_edge.data.store import GameStore


def _fetcher(sport_key: str):
    if sport_key == "mlb":
        from sports_edge.data import mlb as mod
    else:
        from sports_edge.data import nba as mod
    return mod


def cmd_fetch(args):
    cfg = get_sport(args.sport)
    mod = _fetcher(cfg.key)
    seasons = [int(s) for s in args.seasons.split(",")] if args.seasons else cfg.seasons_default
    store = GameStore()
    total = 0
    for season in seasons:
        print(f"Fetching {cfg.name} {season} season...", end=" ", flush=True)
        try:
            games = mod.fetch_season(season)
        except Exception as e:
            print(Fore.RED + f"failed ({e})" + Style.RESET_ALL)
            continue
        n = store.upsert_games(games)
        total += n
        print(f"{n} games")
    print(Fore.GREEN + f"Done. {total} games saved; "
          f"{store.count(cfg.key)} total {cfg.name} games in database." + Style.RESET_ALL)


def cmd_train(args):
    cfg = get_sport(args.sport)
    from sports_edge.features.builder import build_dataset
    from sports_edge.models.train import train

    store = GameStore()
    games = store.load_games(cfg.key)
    if not games:
        sys.exit(f"No {cfg.name} games in database. Run: python main.py fetch --sport {cfg.key}")
    print(f"Building features from {len(games)} games...")
    df = build_dataset(games, cfg)
    print("Training model (holdout = most recent 15% of games)...")
    r = train(df, cfg, fast=args.fast)
    print(f"\n  Train games:        {r.n_train}")
    print(f"  Holdout games:      {r.n_test}")
    print(f"  Accuracy:           {r.accuracy:.3f}  (Elo baseline {r.elo_baseline_accuracy:.3f})")
    print(f"  Log loss:           {r.log_loss:.4f}  (Elo baseline {r.elo_baseline_log_loss:.4f})")
    print(f"  Brier score:        {r.brier:.4f}")
    print(Fore.GREEN + f"\nModel saved to {r.model_path}" + Style.RESET_ALL)


def cmd_predict(args):
    cfg = get_sport(args.sport)
    from sports_edge.models.predict import predict_games

    mod = _fetcher(cfg.key)
    upcoming = mod.todays_games()
    if not upcoming:
        print(f"No {cfg.name} games scheduled today.")
        return

    odds = None
    if not args.no_odds:
        from sports_edge.betting.odds_api import fetch_moneylines
        try:
            odds = fetch_moneylines(cfg, api_key=args.api_key, bookmaker=args.book)
        except Exception as e:
            print(Fore.YELLOW + f"Odds unavailable ({e}); showing probabilities only."
                  + Style.RESET_ALL)

    preds = predict_games(cfg, upcoming, odds=odds, kelly_multiplier=args.kelly)

    print(f"\n{cfg.name} predictions for {preds[0].date}")
    print("=" * 78)
    for p in preds:
        fav_prob = max(p.home_win_prob, p.away_win_prob)
        print(f"\n{p.away_team} @ {p.home_team}")
        print(f"  Model: {Fore.CYAN}{p.pick}{Style.RESET_ALL} "
              f"({fav_prob:.1%})   [home {p.home_win_prob:.1%}, elo {p.elo_prob:.1%}]")
        if p.home_ev is not None:
            print(f"  Odds:  {p.home_team} {p.home_ml:+.0f} ({p.home_book})  |  "
                  f"{p.away_team} {p.away_ml:+.0f} ({p.away_book})  "
                  f"[market home prob {p.home_fair_prob:.1%}]")
            for team, ev, kelly in ((p.home_team, p.home_ev, p.home_kelly),
                                    (p.away_team, p.away_ev, p.away_kelly)):
                if ev > 0:
                    stake = kelly * args.bankroll
                    print(f"  {Fore.GREEN}+EV:  {team}  EV ${ev:+.2f}/100  "
                          f"Kelly {kelly:.1%} of bankroll (${stake:.0f}){Style.RESET_ALL}")
            if p.best_bet is None:
                print(f"  {Fore.YELLOW}No +EV bet at these prices.{Style.RESET_ALL}")
    print()


def cmd_backtest(args):
    cfg = get_sport(args.sport)
    from sports_edge.backtest import run_backtest
    from sports_edge.features.builder import build_dataset

    store = GameStore()
    games = store.load_games(cfg.key)
    if not games:
        sys.exit(f"No {cfg.name} games in database. Run: python main.py fetch --sport {cfg.key}")
    df = build_dataset(games, cfg)
    print(f"Walk-forward backtest over {df['season'].nunique()} seasons "
          f"({len(df)} games)...")
    r = run_backtest(df, cfg, odds_csv=args.odds_csv, min_ev=args.min_ev, fast=args.fast)
    print(f"\n{'Season':>7} {'Games':>6} {'Acc':>6} {'Elo':>6} {'LogLoss':>8} "
          f"{'Brier':>6} {'Bets':>5} {'Profit':>9} {'ROI':>7}")
    for s in r.seasons:
        roi = f"{s.roi:+.1f}%" if s.roi is not None else "-"
        profit = f"${s.profit:+,.0f}" if s.n_bets else "-"
        print(f"{s.season:>7} {s.n_games:>6} {s.accuracy:>6.3f} {s.elo_accuracy:>6.3f} "
              f"{s.log_loss:>8.4f} {s.brier:>6.4f} {s.n_bets:>5} {profit:>9} {roi:>7}")
    print(f"\nOverall accuracy: {r.overall_accuracy:.3f}")
    if r.total_bets:
        print(f"Total: {r.total_bets} bets, profit ${r.total_profit:+,.0f} "
              f"(flat $100 stakes)")


def cmd_web(args):
    from web.app import app
    app.run(host=args.host, port=args.port, debug=args.debug)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sports Edge: MLB & NBA betting predictions")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_sport(p):
        p.add_argument("--sport", required=True, choices=["mlb", "nba"])

    p = sub.add_parser("fetch", help="Download historical game results")
    add_sport(p)
    p.add_argument("--seasons", help="Comma-separated season start years, e.g. 2022,2023,2024")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("train", help="Train the win-probability model")
    add_sport(p)
    p.add_argument("--fast", action="store_true", help="Fewer boosting rounds (for testing)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="Predict today's games with EV and Kelly stakes")
    add_sport(p)
    p.add_argument("--api-key", help="the-odds-api.com key (or set THE_ODDS_API_KEY)")
    p.add_argument("--book", help="Restrict to one bookmaker, e.g. fanduel, draftkings")
    p.add_argument("--no-odds", action="store_true", help="Skip odds; probabilities only")
    p.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll for stake sizing")
    p.add_argument("--kelly", type=float, default=0.25, help="Kelly multiplier (0.25 = quarter)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("backtest", help="Walk-forward backtest by season")
    add_sport(p)
    p.add_argument("--odds-csv", help="CSV with date,home_team,away_team,home_ml,away_ml")
    p.add_argument("--min-ev", type=float, default=2.0, help="Min EV per $100 to place a bet")
    p.add_argument("--fast", action="store_true")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("web", help="Run the web dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    p.set_defaults(func=cmd_web)
    return parser


if __name__ == "__main__":
    colorama_init()
    args = build_parser().parse_args()
    args.func(args)
