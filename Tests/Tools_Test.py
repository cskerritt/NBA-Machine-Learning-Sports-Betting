import unittest

from src.Utils.tools import create_todays_games, create_todays_games_from_odds, to_data_frame


class TestCreateTodaysGames(unittest.TestCase):
    def test_create_todays_games_combines_city_and_name(self):
        games = create_todays_games([
            {'h': {'tc': 'Boston', 'tn': 'Celtics'}, 'v': {'tc': 'Los Angeles', 'tn': 'Lakers'}},
            {'h': {'tc': 'Miami', 'tn': 'Heat'}, 'v': {'tc': 'New York', 'tn': 'Knicks'}},
        ])
        self.assertEqual(games, [
            ['Boston Celtics', 'Los Angeles Lakers'],
            ['Miami Heat', 'New York Knicks'],
        ])

    def test_create_todays_games_from_odds_splits_keys(self):
        odds = {
            'Boston Celtics:Los Angeles Lakers': {},
            'Miami Heat:New York Knicks': {},
        }
        games = create_todays_games_from_odds(odds)
        self.assertIn(['Boston Celtics', 'Los Angeles Lakers'], games)
        self.assertIn(['Miami Heat', 'New York Knicks'], games)

    def test_to_data_frame_uses_headers(self):
        result_set = [{
            'headers': ['TEAM_NAME', 'W'],
            'rowSet': [['Atlanta Hawks', 10], ['Boston Celtics', 12]],
        }]
        df = to_data_frame(result_set)
        self.assertEqual(list(df.columns), ['TEAM_NAME', 'W'])
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(df.iloc[1]['TEAM_NAME'], 'Boston Celtics')


if __name__ == '__main__':
    unittest.main()
