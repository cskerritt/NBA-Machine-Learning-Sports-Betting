import copy
import os

import numpy as np
import xgboost as xgb
from colorama import init, deinit

from src.Utils.prediction_display import display_predictions

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ML_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Models', 'XGBoost_Models', 'XGBoost_74.9%_ML-2.json')
UO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Models', 'XGBoost_Models', 'XGBoost_58.9%_UO-6.json')

init()
xgb_ml = xgb.Booster()
xgb_ml.load_model(ML_MODEL_PATH)
xgb_uo = xgb.Booster()
xgb_uo.load_model(UO_MODEL_PATH)


def _batch_to_per_game(predictions):
    return [predictions[i:i + 1] for i in range(predictions.shape[0])]


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds):
    ml_predictions = xgb_ml.predict(xgb.DMatrix(np.asarray(data)))
    ml_predictions_array = _batch_to_per_game(ml_predictions)

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    uo_data = frame_uo.values.astype(float)

    ou_predictions = xgb_uo.predict(xgb.DMatrix(uo_data))
    ou_predictions_array = _batch_to_per_game(ou_predictions)

    display_predictions(games, ml_predictions_array, ou_predictions_array,
                        todays_games_uo, home_team_odds, away_team_odds)

    deinit()
