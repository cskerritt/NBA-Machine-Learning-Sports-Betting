"""Write daily prediction reports as JSON (machine-readable) and Markdown."""

import json
from datetime import date
from pathlib import Path

from sports_edge.config import REPORTS_DIR, SportConfig
from sports_edge.models.predict import Prediction


def _fmt_ml(ml: float | None) -> str:
    return f"{ml:+.0f}" if ml is not None else "-"


def _game_md(p: Prediction, bankroll: float) -> str:
    lines = [
        f"### {p.away_team} @ {p.home_team}",
        "",
        f"**Pick: {p.pick} ({p.pick_prob:.1%})** — confidence: **{p.confidence}**",
        "",
        f"- Predicted score: {p.home_team} **{p.pred_home_score:.1f}** — "
        f"{p.away_team} **{p.pred_away_score:.1f}** "
        f"(margin {p.pred_margin:+.1f}, total {p.pred_total:.1f})",
        f"- Win probability: model {p.model_prob:.1%}, Elo {p.elo_prob:.1%}, "
        f"blended {p.home_win_prob:.1%} (home side)",
    ]
    for label, team, form in (("Home", p.home_team, p.home_form),
                              ("Away", p.away_team, p.away_form)):
        lines.append(
            f"- {label} form — {team}: Elo {form.get('elo')} (#{form.get('elo_rank')}), "
            f"last {form.get('last_n')}, streak {form.get('streak')}, "
            f"season {form.get('season_record')}, "
            f"{form.get('ppg')} for / {form.get('papg')} against"
        )
    if p.home_ml is not None:
        lines.append(
            f"- Odds: {p.home_team} {_fmt_ml(p.home_ml)} ({p.home_book}) | "
            f"{p.away_team} {_fmt_ml(p.away_ml)} ({p.away_book}) — "
            f"market home prob {p.home_fair_prob:.1%}, edge {p.edge:+.1%}"
        )
        if p.best_bet:
            ev = p.home_ev if p.best_bet == p.home_team else p.away_ev
            kelly = p.home_kelly if p.best_bet == p.home_team else p.away_kelly
            lines.append(f"- **+EV bet: {p.best_bet}** — EV ${ev:+.2f}/100, "
                         f"Kelly stake ${kelly * bankroll:.0f} "
                         f"({kelly:.1%} of ${bankroll:.0f})")
        else:
            lines.append("- No +EV bet at current prices.")
    lines.append("")
    return "\n".join(lines)


def write_report(cfg: SportConfig, preds: list[Prediction], day: date,
                 bankroll: float = 1000.0,
                 reports_dir: Path | None = None) -> tuple[Path, Path]:
    """Write {date}/{sport}.json and .md; also refresh latest/{sport}.md."""
    reports_dir = reports_dir or REPORTS_DIR
    day_dir = reports_dir / day.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)

    json_path = day_dir / f"{cfg.key}.json"
    with open(json_path, "w") as f:
        json.dump({
            "sport": cfg.key,
            "date": day.isoformat(),
            "games": [p.to_dict() for p in preds],
        }, f, indent=2, default=str)

    md = [f"# {cfg.name} predictions — {day.isoformat()}", ""]
    if not preds:
        md.append("No games scheduled.")
    bets = [p for p in preds if p.best_bet]
    if bets:
        md += ["## Best bets", ""]
        for p in sorted(bets, key=lambda p: -(p.edge or 0)):
            ev = p.home_ev if p.best_bet == p.home_team else p.away_ev
            md.append(f"- **{p.best_bet}** ({p.away_team} @ {p.home_team}) — "
                      f"edge {p.edge:+.1%}, EV ${ev:+.2f}/100, {p.confidence}")
        md.append("")
    md += ["## All games", ""]
    md += [_game_md(p, bankroll) for p in preds]
    md_path = day_dir / f"{cfg.key}.md"
    md_path.write_text("\n".join(md))

    latest = reports_dir / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / f"{cfg.key}.md").write_text("\n".join(md))
    return json_path, md_path
