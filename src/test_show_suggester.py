# test_show_suggester.py

import unittest
from unittest.mock import patch
import numpy as np
from show_suggester import (
    generate_recommendations,
    generate_custom_shows,
    get_tv_shows_input,
    confirm_tv_shows
)


class TestTVShowRecommendationSystem(unittest.TestCase):
    @patch("show_suggester.load_tv_shows")
    @patch("show_suggester.load_embeddings")
    def test_generate_recommendations(self, mock_load_embeddings, mock_load_tv_shows):
        tv_shows_mock = [
            {
                "name": "Breaking Bad",
                "genre": "Drama",
                "description": "A teacher turned criminal",
            },
            {
                "name": "Game of Thrones",
                "genre": "Fantasy",
                "description": "A tale of warring kingdoms",
            },
            {
                "name": "Sherlock",
                "genre": "Mystery",
                "description": "A modern-day detective series",
            },
        ]

        mock_load_tv_shows.return_value = tv_shows_mock
        mock_load_embeddings.return_value = {
            "Breaking Bad": np.random.rand(768),
            "Game of Thrones": np.random.rand(768),
            "Sherlock": np.random.rand(768),
        }

        user_shows = ["breaking bad", "game of thrones"]

        recommendations = generate_recommendations(
            user_shows, tv_shows_mock, mock_load_embeddings.return_value
        )

        self.assertEqual(recommendations[0][0], "Breaking Bad")
        self.assertEqual(recommendations[1][0], "Game of Thrones")

    @patch("show_suggester.load_tv_shows")
    @patch("show_suggester.load_embeddings")
    def test_generate_custom_shows(self, mock_load_embeddings, mock_load_tv_shows):
        # Sample test data for recommendations
        tv_shows_mock = [
            {
                "name": "Breaking Bad",
                "genre": "Drama",
                "description": "A teacher turned criminal",
            },
            {
                "name": "Game of Thrones",
                "genre": "Fantasy",
                "description": "A tale of warring kingdoms",
            },
            {
                "name": "Sherlock",
                "genre": "Mystery",
                "description": "A modern-day detective series",
            },
        ]

        mock_load_embeddings.return_value = {
            "Breaking Bad": np.random.rand(768),
            "Game of Thrones": np.random.rand(768),
            "Sherlock": np.random.rand(768),
        }

        recommendations = [("Breaking Bad", 95), ("Sherlock", 85)]

        custom_show1, custom_show2 = generate_custom_shows(
            ["breaking bad", "game of thrones"], recommendations, tv_shows_mock
        )

        # Check if the custom shows are correct
        self.assertIn("Breaking Bad", custom_show1)
        self.assertIn("Sherlock", custom_show2)
    
    @patch("builtins.input", side_effect=["breaking bad, game of thrones"])
    def test_get_tv_shows_input_valid(self, mock_input):
        # Test for valid input (more than one show)
        shows = get_tv_shows_input()
        self.assertEqual(shows, ["breaking bad", "game of thrones"])

    @patch(
        "builtins.input", side_effect=["breaking bad", "breaking bad, game of thrones"]
    )
    def test_get_tv_shows_input_invalid(self, mock_input):
        # Test for invalid input (only one show)
        shows = get_tv_shows_input()
        self.assertEqual(shows, ["breaking bad", "game of thrones"])

    @patch("builtins.input", side_effect=["y"])
    def test_confirm_tv_shows_yes(self, mock_input):
        # Test for correct confirmation
        shows = ["breaking bad", "game of thrones"]
        confirmed = confirm_tv_shows(shows)
        self.assertTrue(confirmed)

    @patch("builtins.input", side_effect=["n", "n"])
    def test_confirm_tv_shows_no_then_no(self, mock_input):
        # Test for both incorrect confirmations
        shows = ["breaking bad", "game of thrones"]
        confirmed = confirm_tv_shows(shows)
        self.assertFalse(confirmed)


if __name__ == "__main__":
    unittest.main()
