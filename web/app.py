"""Flask dashboard for today's MLB / NBA / NFL / NHL predictions."""

from flask import Flask, render_template, request

from sports_edge.config import SPORTS, get_sport

app = Flask(__name__)


def _load_predictions(sport_key: str):
    from sports_edge.pipeline import predict_today
    return predict_today(get_sport(sport_key))


@app.route("/")
def index():
    sport = request.args.get("sport", "nba")
    bankroll = float(request.args.get("bankroll", 1000))
    error = None
    preds, odds_error = [], None
    try:
        preds, odds_error = _load_predictions(sport)
    except Exception as e:
        error = str(e)
    return render_template(
        "index.html",
        sports=SPORTS, sport=sport, bankroll=bankroll,
        predictions=preds, error=error, odds_error=odds_error,
    )


if __name__ == "__main__":
    app.run(debug=True)
