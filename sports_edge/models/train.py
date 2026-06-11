"""Train a calibrated win-probability model on the historical feature matrix."""

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from sports_edge.config import MODELS_DIR, SportConfig
from sports_edge.features.state import FEATURE_NAMES

try:
    from xgboost import XGBClassifier

    def _make_estimator(fast: bool):
        return XGBClassifier(
            n_estimators=60 if fast else 400,
            max_depth=3,
            learning_rate=0.2 if fast else 0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )
except ImportError:  # xgboost optional; sklearn fallback
    from sklearn.ensemble import HistGradientBoostingClassifier

    def _make_estimator(fast: bool):
        return HistGradientBoostingClassifier(
            max_iter=60 if fast else 400, max_depth=3,
            learning_rate=0.2 if fast else 0.03,
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


def model_path(cfg: SportConfig) -> Path:
    return MODELS_DIR / f"{cfg.key}_moneyline.pkl"


def train(df: pd.DataFrame, cfg: SportConfig, fast: bool = False) -> TrainResult:
    """Time-ordered split (last 15% held out), isotonic-calibrated boosted trees."""
    df = df[df["warm"] == 1].sort_values("date").reset_index(drop=True)
    if len(df) < 200:
        raise ValueError(f"Only {len(df)} usable games; fetch more seasons first.")

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["home_win"].to_numpy()
    split = int(len(df) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = CalibratedClassifierCV(_make_estimator(fast), method="isotonic", cv=3)
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    p_elo = df["elo_prob_home"].to_numpy(dtype=float)[split:]

    result = TrainResult(
        model_path=model_path(cfg),
        n_train=len(X_train),
        n_test=len(X_test),
        accuracy=accuracy_score(y_test, p_test > 0.5),
        log_loss=log_loss(y_test, p_test, labels=[0, 1]),
        brier=brier_score_loss(y_test, p_test),
        elo_baseline_accuracy=accuracy_score(y_test, p_elo > 0.5),
        elo_baseline_log_loss=log_loss(y_test, np.clip(p_elo, 1e-6, 1 - 1e-6), labels=[0, 1]),
    )

    # Refit on everything before saving so predictions use all data.
    final = CalibratedClassifierCV(_make_estimator(fast), method="isotonic", cv=3)
    final.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result.model_path, "wb") as f:
        pickle.dump({
            "model": final,
            "feature_names": FEATURE_NAMES,
            "sport": cfg.key,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_games": len(df),
            "holdout_metrics": {
                "accuracy": result.accuracy,
                "log_loss": result.log_loss,
                "brier": result.brier,
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
        return pickle.load(f)
