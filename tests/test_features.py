from sports_edge.config import NBA
from sports_edge.features.builder import build_dataset, replay_state
from sports_edge.features.state import FEATURE_NAMES
from tests.synthetic import synthetic_seasons


def test_dataset_shape_and_columns():
    games = synthetic_seasons([2022, 2023])
    df = build_dataset(games, NBA)
    assert len(df) == len(games)
    for col in FEATURE_NAMES + ["home_win", "warm", "date", "season"]:
        assert col in df.columns


def test_no_lookahead_leakage():
    """Features for the first meeting of two new teams must be neutral priors —
    they cannot reflect that game's own result."""
    games = synthetic_seasons([2022])
    df = build_dataset(games, NBA)
    first = df.iloc[0]
    assert first["elo_home"] == 1500.0
    assert first["elo_away"] == 1500.0
    assert first["home_winpct"] == 0.5
    assert first["games_played_home"] == 0


def test_features_predictive_of_outcome():
    """Elo probability should beat coin-flip accuracy on a skill-driven league."""
    games = synthetic_seasons([2022, 2023])
    df = build_dataset(games, NBA)
    warm = df[df["warm"] == 1]
    acc = ((warm["elo_prob_home"] > 0.5) == (warm["home_win"] == 1)).mean()
    assert acc > 0.60


def test_replay_state_matches_incremental_build():
    games = synthetic_seasons([2022])
    state = replay_state(games, NBA)
    # Every team has played and ratings have diverged from the mean.
    assert len(state.elo.ratings) == 12
    assert max(state.elo.ratings.values()) > 1520
    assert min(state.elo.ratings.values()) < 1480


def test_rest_days_capped():
    games = synthetic_seasons([2022])
    df = build_dataset(games, NBA)
    assert df["rest_home"].max() <= NBA.rest_cap_days
    assert df["rest_home"].min() >= 0
