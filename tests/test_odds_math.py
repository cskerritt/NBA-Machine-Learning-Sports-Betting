import pytest

from sports_edge.betting.odds_math import (
    american_to_decimal,
    american_to_implied_prob,
    decimal_to_american,
    expected_value,
    kelly_fraction,
    no_vig_probs,
)


def test_american_to_decimal():
    assert american_to_decimal(100) == 2.0
    assert american_to_decimal(-100) == 2.0
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(-200) == 1.5


def test_invalid_american_odds():
    with pytest.raises(ValueError):
        american_to_decimal(50)


def test_roundtrip():
    for ml in (-350, -110, 100, 125, 600):
        assert decimal_to_american(american_to_decimal(ml)) == ml


def test_implied_prob():
    assert american_to_implied_prob(-100) == pytest.approx(0.5)
    assert american_to_implied_prob(-200) == pytest.approx(2 / 3)
    assert american_to_implied_prob(200) == pytest.approx(1 / 3)


def test_no_vig_sums_to_one():
    a, b = no_vig_probs(american_to_implied_prob(-110), american_to_implied_prob(-110))
    assert a + b == pytest.approx(1.0)
    assert a == pytest.approx(0.5)


def test_expected_value_signs():
    # 60% to win at even money is clearly +EV; 40% is -EV.
    assert expected_value(0.6, 100) == pytest.approx(20.0)
    assert expected_value(0.4, 100) == pytest.approx(-20.0)
    # Fair odds => zero EV.
    assert expected_value(0.5, 100) == pytest.approx(0.0)


def test_kelly():
    # p=0.6 at even money: full Kelly = 0.2, quarter Kelly = 0.05.
    assert kelly_fraction(0.6, 100, multiplier=1.0) == pytest.approx(0.1)  # capped
    assert kelly_fraction(0.6, 100, multiplier=0.25) == pytest.approx(0.05)
    # No edge => no bet.
    assert kelly_fraction(0.5, -110) == 0.0
    assert kelly_fraction(0.3, 100) == 0.0
