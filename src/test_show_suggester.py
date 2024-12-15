# test_show_suggester.py 
import unittest
from unittest.mock import patch, mock_open
import numpy as np
import pickle
from show_suggester import (
    get_tv_shows_input,
    confirm_tv_shows,
    fuzzy_match,
    load_embeddings,
    load_tv_shows,
    shows_creator,
    cosine_similarity,
)

class TestTVShowRecommendationSystem(unittest.TestCase):
    @patch(
        "builtins.input", side_effect=["breaking bad", "breaking bad, game of thrones"]
    )
    def test_get_tv_shows_input_invalid(self, mock_input):
        shows = get_tv_shows_input()
        self.assertEqual(shows, ["breaking bad", "game of thrones"])

    # Test for confirm_tv_shows function
    @patch("builtins.input", side_effect=["y"])
    def test_confirm_tv_shows_yes(self, mock_input):
        shows = ["breaking bad", "game of thrones"]
        confirmed = confirm_tv_shows(shows)
        self.assertTrue(confirmed)

    @patch("builtins.input", side_effect=["n", "n"])
    def test_confirm_tv_shows_no_then_no(self, mock_input):
        shows = ["breaking bad", "game of thrones"]
        confirmed = confirm_tv_shows(shows)
        self.assertFalse(confirmed)

    # Test for fuzzy_match function
    @patch("show_suggester.process.extractOne")
    def test_fuzzy_match(self, mock_extract_one):
        mock_extract_one.return_value = ("Game of Thrones", 90)
        result = fuzzy_match("game of thrones", [{"name": "Game of Thrones"}])
        self.assertEqual(result, ("Game of Thrones", 90))

    # Test for load_embeddings function
    @patch(
        "builtins.open",
        mock_open(read_data=pickle.dumps({"Breaking Bad": np.random.rand(768)})),
    )
    def test_load_embeddings(self):
        embeddings = load_embeddings("dummy_file.pkl")
        self.assertIn("Breaking Bad", embeddings)

    # Test for load_tv_shows function
    @patch(
        "builtins.open",
        mock_open(
            read_data="Title,Description,Genres\nBreaking Bad,A teacher turned criminal,Drama\n"
        ),
    )
    def test_load_tv_shows(self):
        tv_shows = load_tv_shows("dummy_file.csv")
        self.assertEqual(tv_shows[0]["name"], "Breaking Bad")

    patch("show_suggester.process.extractOne")
    # Test for shows_creator function
    @patch("openai.ChatCompletion.create")
    def test_shows_creator(self, mock_create):
        mock_create.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Show name: Game of Thrones  Description: It is about a fantasy drama"
                    }
                }
            ]
        }
        recommendations = [("Game of Thrones", 90)]
        user_input = ["breaking bad"]
        message = shows_creator(recommendations, user_input, "dummy_openai_api_key")
        self.assertIn("Game of Thrones", message)

    # Test for cosine_similarity function
    def test_cosine_similarity(self):
        vec1 = np.array([1, 0])
        vec2 = np.array([0, 1])
        similarity = cosine_similarity(vec1, vec2)
        self.assertEqual(similarity, 0)

    # Test for edge case in cosine_similarity (same vectors)
    def test_cosine_similarity_same(self):
        vec1 = np.array([1, 0])
        vec2 = np.array([1, 0])
        similarity = cosine_similarity(vec1, vec2)
        self.assertEqual(similarity, 1)


if __name__ == "__main__":
    unittest.main()
