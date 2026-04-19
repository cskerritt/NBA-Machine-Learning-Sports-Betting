import os
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.Utils.Dictionaries import team_index_07, team_index_08, team_index_12, team_index_13, team_index_14

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

season_array = ["2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16",
                "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]
odds_directory = os.path.join(PROJECT_ROOT, 'Odds-Data', 'Odds-Data-Clean')

season_to_index = {
    '2007-08': team_index_07,
    '2008-09': team_index_08,
    '2009-10': team_index_08,
    '2010-11': team_index_08,
    '2011-12': team_index_08,
    '2012-13': team_index_12,
    '2013-14': team_index_13,
}

scores = []
win_margin = []
OU = []
OU_Cover = []
games = []
for season in tqdm(season_array):
    file = pd.read_excel(os.path.join(odds_directory, '{}.xlsx'.format(season)))

    team_data_directory = os.path.join(PROJECT_ROOT, 'Team-Data', season)

    for row in file.itertuples():
        home_team = row[3]
        away_team = row[4]

        date = row[2]
        date_array = date.split('-')
        year = date_array[0] + '-' + date_array[1]
        parsed_date = datetime.strptime(date_array[2], '%m%d')

        team_data_file = f'{parsed_date.month}-{parsed_date.day}-{year}.xlsx'

        data_frame = pd.read_excel(os.path.join(team_data_directory, team_data_file))

        if len(data_frame.index) == 30:
            scores.append(row[9])
            OU.append(row[5])
            if row[10] > 0:
                win_margin.append(1)
            else:
                win_margin.append(0)

            if row[9] < row[5]:
                OU_Cover.append(0)
            elif row[9] > row[5]:
                OU_Cover.append(1)
            elif row[9] == row[5]:
                OU_Cover.append(2)

            team_index = season_to_index.get(season, team_index_14)
            if home_team not in team_index or away_team not in team_index:
                # Skip games whose teams aren't in the season's index (e.g. relocations).
                continue
            home_team_series = data_frame.iloc[team_index[home_team]]
            away_team_series = data_frame.iloc[team_index[away_team]]

            game = pd.concat([home_team_series, away_team_series])
            games.append(game)
season = pd.concat(games, ignore_index=True, axis=1)
season = season.T

frame = season.drop(columns=['TEAM_ID', 'CFID', 'CFPARAMS', 'Unnamed: 0'])
frame['Score'] = np.asarray(scores)
frame['Home-Team-Win'] = np.asarray(win_margin)
frame['OU'] = np.asarray(OU)
frame['OU-Cover'] = np.asarray(OU_Cover)
frame.to_excel(os.path.join(PROJECT_ROOT, 'Datasets', 'DataSet-2021-22.xlsx'))
