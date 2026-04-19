import unittest

from src.Utils.Dictionaries import (
    team_index_07,
    team_index_08,
    team_index_12,
    team_index_13,
    team_index_14,
    team_index_current,
)

ALL_INDICES = {
    '07': team_index_07,
    '08': team_index_08,
    '12': team_index_12,
    '13': team_index_13,
    '14': team_index_14,
    'current': team_index_current,
}


class TestTeamIndices(unittest.TestCase):
    def test_indices_are_zero_based_and_contiguous(self):
        for label, mapping in ALL_INDICES.items():
            with self.subTest(label=label):
                # `current` includes an alias ('LA Clippers' -> 12), so we look at unique values.
                values = sorted(set(mapping.values()))
                self.assertEqual(values[0], 0, f"{label} indices should start at 0")
                self.assertEqual(
                    values, list(range(len(values))),
                    f"{label} indices should be contiguous",
                )

    def test_current_has_30_unique_indices(self):
        self.assertEqual(len(set(team_index_current.values())), 30)

    def test_la_clippers_alias_maps_same_as_full_name(self):
        self.assertEqual(team_index_current['LA Clippers'], team_index_current['Los Angeles Clippers'])


if __name__ == '__main__':
    unittest.main()
