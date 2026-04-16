import copy
import numpy as np
import tensorflow as tf
from colorama import init, deinit
from tensorflow.keras.models import load_model
from src.Utils.prediction_display import display_predictions

init()
model = load_model('Models/NN_Models/Trained-Model-ML')
ou_model = load_model("Models/NN_Models/Trained-Model-OU")


def nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds):
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(model.predict(np.array([row])))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)
    data = tf.keras.utils.normalize(data, axis=1)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(ou_model.predict(np.array([row])))

    display_predictions(games, ml_predictions_array, ou_predictions_array,
                        todays_games_uo, home_team_odds, away_team_odds)

    deinit()
