import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FLASK_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, 'Flask'))
sys.path.insert(0, FLASK_DIR)

import app as flask_app  # noqa: E402


class TestFetchGameData(unittest.TestCase):
    def test_unknown_sportsbook_raises(self):
        with self.assertRaises(ValueError):
            flask_app.fetch_game_data(sportsbook='not-a-real-book; rm -rf /')

    def test_allowed_sportsbooks_set(self):
        for book in ('fanduel', 'draftkings', 'betmgm'):
            self.assertIn(book, flask_app.ALLOWED_SPORTSBOOKS)


if __name__ == '__main__':
    unittest.main()
