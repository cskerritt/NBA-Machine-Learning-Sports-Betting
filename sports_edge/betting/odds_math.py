"""Odds conversions, expected value, and Kelly criterion stake sizing."""


def american_to_decimal(american: float) -> float:
    if american >= 100:
        return 1.0 + american / 100.0
    if american <= -100:
        return 1.0 + 100.0 / abs(american)
    raise ValueError(f"Invalid American odds: {american}")


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError(f"Invalid decimal odds: {decimal}")
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100.0)
    return round(-100.0 / (decimal - 1.0))


def american_to_implied_prob(american: float) -> float:
    return 1.0 / american_to_decimal(american)


def no_vig_probs(implied_a: float, implied_b: float) -> tuple[float, float]:
    """Remove the bookmaker margin from a two-way market."""
    total = implied_a + implied_b
    return implied_a / total, implied_b / total


def expected_value(p_win: float, american: float, stake: float = 100.0) -> float:
    """Expected profit on `stake` given our win probability and the offered odds."""
    decimal = american_to_decimal(american)
    return p_win * stake * (decimal - 1.0) - (1.0 - p_win) * stake


def kelly_fraction(p_win: float, american: float, multiplier: float = 0.25,
                   cap: float = 0.10) -> float:
    """Fractional Kelly stake as a share of bankroll (0 when there is no edge).

    `multiplier` scales down full Kelly (quarter-Kelly default — full Kelly
    is far too aggressive given model error), `cap` bounds any single bet.
    """
    b = american_to_decimal(american) - 1.0
    full = (p_win * b - (1.0 - p_win)) / b
    return max(0.0, min(full * multiplier, cap))
