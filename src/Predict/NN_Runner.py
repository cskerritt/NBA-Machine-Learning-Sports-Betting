import copy
import os

import numpy as np
import tensorflow as tf
from colorama import init, deinit
from tensorflow.keras.models import load_model

from src.Utils.prediction_display import display_predictions

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ML_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Models', 'NN_Models', 'Trained-Model-ML')
UO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Models', 'NN_Models', 'Trained-Model-OU')

init()
model = load_model(ML_MODEL_PATH)
ou_model = load_model(UO_MODEL_PATH)


def _batch_to_per_game(predictions):
    return [predictions[i:i + 1] for i in range(predictions.shape[0])]


def nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds):
    ml_predictions = model.predict(np.asarray(data))
    ml_predictions_array = _batch_to_per_game(ml_predictions)

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    uo_data = frame_uo.values.astype(float)
    uo_data = tf.keras.utils.normalize(uo_data, axis=1)

    ou_predictions = ou_model.predict(np.asarray(uo_data))
    ou_predictions_array = _batch_to_per_game(ou_predictions)

    display_predictions(games, ml_predictions_array, ou_predictions_array,
                        todays_games_uo, home_team_odds, away_team_odds)

    deinit()
