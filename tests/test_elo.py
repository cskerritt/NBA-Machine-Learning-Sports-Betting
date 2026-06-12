import pytest

from sports_edge.features.elo import MEAN_RATING, Elo


def make_elo():
    return Elo(k=20.0, home_advantage=80.0, season_regression=0.25)


def test_default_rating():
    assert make_elo().rating("Anyone") == MEAN_RATING


def test_home_advantage():
    elo = make_elo()
    assert elo.expected_home_win_prob("A", "B") > 0.5


def test_update_moves_ratings_in_right_direction():
    elo = make_elo()
    elo.update("A", "B", 110, 100)
    assert elo.rating("A") > MEAN_RATING
    assert elo.rating("B") < MEAN_RATING


def test_zero_sum():
    elo = make_elo()
    elo.update("A", "B", 95, 120)
    elo.update("C", "A", 100, 99)
    total = sum(elo.ratings.values())
    assert total == pytest.approx(MEAN_RATING * 3)


def test_bigger_margin_bigger_shift():
    e1, e2 = make_elo(), make_elo()
    e1.update("A", "B", 101, 100)
    e2.update("A", "B", 130, 100)
    assert e2.rating("A") > e1.rating("A")


def test_upset_shifts_more_than_expected_win():
    e1, e2 = make_elo(), make_elo()
    e1.ratings = {"A": 1600.0, "B": 1400.0}
    e2.ratings = {"A": 1600.0, "B": 1400.0}
    e1.update("A", "B", 110, 100)  # favorite wins: small shift
    e2.update("B", "A", 110, 100)  # underdog wins at home: big shift
    assert abs(e2.rating("B") - 1400.0) > abs(e1.rating("A") - 1600.0)


def test_season_regression():
    elo = make_elo()
    elo.ratings = {"A": 1700.0, "B": 1300.0}
    elo.new_season()
    assert elo.rating("A") == pytest.approx(1650.0)
    assert elo.rating("B") == pytest.approx(1350.0)
