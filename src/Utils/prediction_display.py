import numpy as np
from colorama import Fore, Style
from src.Utils import Expected_Value


def display_predictions(games, ml_predictions_array, ou_predictions_array, todays_games_uo,
                        home_team_odds, away_team_odds):
    for count, game in enumerate(games):
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        ou_line = str(todays_games_uo[count])

        if under_over == 0:
            ou_label = Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL
            un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
        else:
            ou_label = Fore.BLUE + 'OVER ' + Style.RESET_ALL
            un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)

        if winner == 1:
            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            print(Fore.GREEN + home_team + Style.RESET_ALL +
                  Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL +
                  ' vs ' +
                  Fore.RED + away_team + Style.RESET_ALL + ': ' +
                  ou_label + ou_line + Style.RESET_ALL +
                  Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        else:
            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            print(Fore.RED + home_team + Style.RESET_ALL +
                  ' vs ' +
                  Fore.GREEN + away_team + Style.RESET_ALL +
                  Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                  ou_label + ou_line + Style.RESET_ALL +
                  Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)

    print("--------------------Expected Value---------------------")
    for count, game in enumerate(games):
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
        if ev_home > 0:
            print(home_team + ' EV: ' + Fore.GREEN + str(ev_home) + Style.RESET_ALL)
        else:
            print(home_team + ' EV: ' + Fore.RED + str(ev_home) + Style.RESET_ALL)

        if ev_away > 0:
            print(away_team + ' EV: ' + Fore.GREEN + str(ev_away) + Style.RESET_ALL)
        else:
            print(away_team + ' EV: ' + Fore.RED + str(ev_away) + Style.RESET_ALL)
