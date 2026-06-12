"""Train calibrated win-probability and score models on the feature matrix.

The final win probability is a blend of the calibrated gradient-boosted
classifier and the Elo baseline, with the blend weight chosen to minimize
log loss on a time-ordered holdout. Margin and total regressors provide
predicted final scores.
"""

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
)

from sports_edge.config import MODELS_DIR, SportConfig
from sports_edge.features.state import FEATURE_NAMES

try:
    from xgboost import XGBClassifier, XGBRegressor

    def _make_estimator(fast: bool):
        return XGBClassifier(
            n_estimators=60 if fast else 400,
            max_depth=3,
            learning_rate=0.2 if fast else 0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )

    def _make_regressor(fast: bool):
        return XGBRegressor(
            n_estimators=60 if fast else 300,
            max_depth=3,
            learning_rate=0.2 if fast else 0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
except ImportError:  # xgboost optional; sklearn fallback
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    def _make_estimator(fast: bool):
        return HistGradientBoostingClassifier(
            max_iter=60 if fast else 400, max_depth=3,
            learning_rate=0.2 if fast else 0.03,
        )

    def _make_regressor(fast: bool):
        return HistGradientBoostingRegressor(
            max_iter=60 if fast else 300, max_depth=3,
            learning_rate=0.2 if fast else 0.05,
        )


@dataclass
class TrainResult:
    model_path: Path
    n_train: int
    n_test: int
    accuracy: float
    log_loss: float
    brier: float
    elo_baseline_accuracy: float
    elo_baseline_log_loss: float
    blend_weight: float          # weight on the ML model (1 - w on Elo)
    margin_mae: float
    total_mae: float


def model_path(cfg: SportConfig) -> Path:
    return MODELS_DIR / f"{cfg.key}_moneyline.pkl"


def blend(p_model, p_elo, w: float):
    """Final win probability: classifier blended with the Elo baseline."""
    return np.clip(w * p_model + (1.0 - w) * p_elo, 1e-6, 1 - 1e-6)


def _best_blend_weight(p_model, p_elo, y) -> float:
    weights = np.arange(0.0, 1.01, 0.05)
    losses = [log_loss(y, blend(p_model, p_elo, w), labels=[0, 1]) for w in weights]
    return float(weights[int(np.argmin(losses))])


def train(df: pd.DataFrame, cfg: SportConfig, fast: bool = False) -> TrainResult:
    """Time-ordered split (last 15% held out); blend weight tuned on the holdout."""
    df = df[df["warm"] == 1].sort_values("date").reset_index(drop=True)
    if len(df) < 200:
        raise ValueError(f"Only {len(df)} usable games; fetch more seasons first.")

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["home_win"].to_numpy()
    p_elo_all = df["elo_prob_home"].to_numpy(dtype=float)
    split = int(len(df) * 0.85)

    model = CalibratedClassifierCV(_make_estimator(fast), method="isotonic", cv=3)
    model.fit(X[:split], y[:split])
    p_test = model.predict_proba(X[split:])[:, 1]
    p_elo = p_elo_all[split:]
    y_test = y[split:]

    w = _best_blend_weight(p_test, p_elo, y_test)
    p_blend = blend(p_test, p_elo, w)

    margin_model = _make_regressor(fast)
    margin_model.fit(X[:split], df["margin"].to_numpy(dtype=float)[:split])
    total_model = _make_regressor(fast)
    total_model.fit(X[:split], df["total"].to_numpy(dtype=float)[:split])

    result = TrainResult(
        model_path=model_path(cfg),
        n_train=split,
        n_test=len(df) - split,
        accuracy=accuracy_score(y_test, p_blend > 0.5),
        log_loss=log_loss(y_test, p_blend, labels=[0, 1]),
        brier=brier_score_loss(y_test, p_blend),
        elo_baseline_accuracy=accuracy_score(y_test, p_elo > 0.5),
        elo_baseline_log_loss=log_loss(y_test, np.clip(p_elo, 1e-6, 1 - 1e-6),
                                       labels=[0, 1]),
        blend_weight=w,
        margin_mae=mean_absolute_error(df["margin"][split:],
                                       margin_model.predict(X[split:])),
        total_mae=mean_absolute_error(df["total"][split:],
                                      total_model.predict(X[split:])),
    )

    # Refit everything on all data before saving; keep the tuned blend weight.
    final = CalibratedClassifierCV(_make_estimator(fast), method="isotonic", cv=3)
    final.fit(X, y)
    final_margin = _make_regressor(fast)
    final_margin.fit(X, df["margin"].to_numpy(dtype=float))
    final_total = _make_regressor(fast)
    final_total.fit(X, df["total"].to_numpy(dtype=float))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result.model_path, "wb") as f:
        pickle.dump({
            "model": final,
            "margin_model": final_margin,
            "total_model": final_total,
            "blend_weight": w,
            "feature_names": FEATURE_NAMES,
            "sport": cfg.key,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_games": len(df),
            "holdout_metrics": {
                "accuracy": result.accuracy,
                "log_loss": result.log_loss,
                "brier": result.brier,
                "margin_mae": result.margin_mae,
                "total_mae": result.total_mae,
            },
        }, f)
    return result


def load_model(cfg: SportConfig) -> dict:
    path = model_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model for {cfg.name} at {path}. "
            f"Run: python main.py train --sport {cfg.key}"
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("feature_names") != FEATURE_NAMES:
        raise ValueError(
            f"Saved {cfg.name} model was trained on a different feature set; "
            f"predictions would be misaligned. "
            f"Retrain with: python main.py train --sport {cfg.key}"
        )
    return bundle
