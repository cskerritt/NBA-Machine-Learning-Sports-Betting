"""Flask dashboard for today's MLB/NBA predictions."""

from flask import Flask, render_template, request

from sports_edge.config import SPORTS, get_sport

app = Flask(__name__)


def _load_predictions(sport_key: str, bankroll: float):
    cfg = get_sport(sport_key)
    if cfg.key == "mlb":
        from sports_edge.data import mlb as fetcher
    else:
        from sports_edge.data import nba as fetcher
    from sports_edge.models.predict import predict_games

    upcoming = fetcher.todays_games()
    if not upcoming:
        return [], None

    odds, odds_error = None, None
    try:
        from sports_edge.betting.odds_api import fetch_moneylines
        odds = fetch_moneylines(cfg)
    except Exception as e:
        odds_error = str(e)

    preds = predict_games(cfg, upcoming, odds=odds)
    return preds, odds_error


@app.route("/")
def index():
    sport = request.args.get("sport", "nba")
    bankroll = float(request.args.get("bankroll", 1000))
    error = None
    preds, odds_error = [], None
    try:
        preds, odds_error = _load_predictions(sport, bankroll)
    except Exception as e:
        error = str(e)
    return render_template(
        "index.html",
        sports=SPORTS, sport=sport, bankroll=bankroll,
        predictions=preds, error=error, odds_error=odds_error,
    )


if __name__ == "__main__":
    app.run(debug=True)
