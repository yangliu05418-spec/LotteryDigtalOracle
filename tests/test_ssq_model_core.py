import math
import tempfile
import unittest
from pathlib import Path


class TestSSQModelCore(unittest.TestCase):
    def test_validate_draws_rejects_duplicate_red_balls(self):
        from ssq_model.data import Draw, validate_draws

        draws = [Draw("2026001", "2026-01-01", (1, 2, 3, 4, 5, 5), 8)]

        with self.assertRaises(ValueError):
            validate_draws(draws)

    def test_load_draws_accepts_utf8_sig_csv_and_sorts_by_issue(self):
        from ssq_model.data import load_draws

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            path.write_text(
                "期号,日期,红球-1,红球-2,红球-3,红球-4,红球-5,红球-6,蓝球\n"
                "2026002,2026/1/4,2,4,6,8,10,12,7\n"
                "2026001,2026/1/1,1,3,5,7,9,11,16\n",
                encoding="utf-8-sig",
            )

            draws = load_draws(path)

        self.assertEqual([d.issue for d in draws], ["2026001", "2026002"])
        self.assertEqual(draws[0].red_balls, (1, 3, 5, 7, 9, 11))
        self.assertEqual(draws[1].blue_ball, 7)

    def test_theory_combination_space_and_uniform_probabilities(self):
        from ssq_model.theory import TOTAL_COMBINATIONS, blue_uniform, red_marginal_uniform

        self.assertEqual(TOTAL_COMBINATIONS, 17_721_088)
        self.assertAlmostEqual(sum(blue_uniform().values()), 1.0)
        self.assertAlmostEqual(sum(red_marginal_uniform().values()), 6.0)
        self.assertAlmostEqual(red_marginal_uniform()[1], 6 / 33)

    def test_random_combination_is_legal_and_seeded(self):
        from ssq_model.theory import random_combination

        combo_a = random_combination(seed=42)
        combo_b = random_combination(seed=42)

        self.assertEqual(combo_a, combo_b)
        self.assertEqual(len(combo_a.red_balls), 6)
        self.assertEqual(len(set(combo_a.red_balls)), 6)
        self.assertTrue(all(1 <= n <= 33 for n in combo_a.red_balls))
        self.assertTrue(1 <= combo_a.blue_ball <= 16)

    def test_features_are_computed_for_draw_sequence(self):
        from ssq_model.data import Draw
        from ssq_model.features import compute_features

        previous = Draw("2026001", "2026-01-01", (1, 2, 3, 10, 20, 30), 1)
        current = Draw("2026002", "2026-01-04", (2, 4, 6, 8, 10, 12), 2)

        features = compute_features(current, previous)

        self.assertEqual(features["sum"], 42)
        self.assertEqual(features["span"], 10)
        self.assertEqual(features["odd_count"], 0)
        self.assertEqual(features["small_count"], 6)
        self.assertEqual(features["consecutive_pairs"], 0)
        self.assertEqual(features["repeat_count"], 2)
        self.assertEqual(features["zone_1_11"], 5)
        self.assertEqual(features["zone_12_22"], 1)
        self.assertEqual(features["zone_23_33"], 0)
        self.assertEqual(features["ac_value"], 0)

    def test_frequency_and_smoothed_models_return_valid_probabilities(self):
        from ssq_model.data import Draw
        from ssq_model.models import BayesianSmoothingModel, FrequencyModel

        draws = [
            Draw("1", "2026-01-01", (1, 2, 3, 4, 5, 6), 1),
            Draw("2", "2026-01-04", (1, 2, 7, 8, 9, 10), 2),
        ]

        for model in (FrequencyModel(), BayesianSmoothingModel(alpha=1.0)):
            prediction = model.fit(draws).predict_proba()
            self.assertEqual(set(prediction.red_probs), set(range(1, 34)))
            self.assertEqual(set(prediction.blue_probs), set(range(1, 17)))
            self.assertTrue(all(0 <= p <= 1 for p in prediction.red_probs.values()))
            self.assertAlmostEqual(sum(prediction.red_probs.values()), 6.0)
            self.assertAlmostEqual(sum(prediction.blue_probs.values()), 1.0)

    def test_candidate_generation_is_seeded_unique_and_scored(self):
        from ssq_model.generator import generate_candidates
        from ssq_model.models.base import Prediction

        prediction = Prediction(
            red_probs={i: (0.6 if i <= 6 else 0.01) for i in range(1, 34)},
            blue_probs={i: (0.5 if i == 1 else 0.5 / 15) for i in range(1, 17)},
            model_name="test",
        )

        first = generate_candidates(prediction, top_k=5, seed=123, pool_size=50)
        second = generate_candidates(prediction, top_k=5, seed=123, pool_size=50)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len({(c.red_balls, c.blue_ball) for c in first}), 5)
        self.assertTrue(all(len(c.red_balls) == 6 for c in first))
        self.assertTrue(all(c.score > 0 for c in first))

    def test_backtest_uses_only_past_draws(self):
        from ssq_model.backtest import rolling_backtest
        from ssq_model.data import Draw
        from ssq_model.models import FrequencyModel

        draws = [
            Draw("1", "2026-01-01", (1, 2, 3, 4, 5, 6), 1),
            Draw("2", "2026-01-04", (1, 2, 3, 4, 5, 7), 1),
            Draw("3", "2026-01-07", (8, 9, 10, 11, 12, 13), 16),
        ]

        result = rolling_backtest(draws, lambda: FrequencyModel(), min_train_size=2)

        self.assertEqual(result.n_predictions, 1)
        self.assertEqual(result.rows[0]["issue"], "3")
        self.assertAlmostEqual(result.rows[0]["red_log_loss"], -math.log(1e-12))
        self.assertAlmostEqual(result.rows[0]["blue_log_loss"], -math.log(1e-12))


if __name__ == "__main__":
    unittest.main()
