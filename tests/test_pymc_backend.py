import unittest

from ssq_model.data import Draw


class TestPyMCBackend(unittest.TestCase):
    def test_pymc_not_installed_error_mentions_mac_python3_install(self):
        from ssq_model.models.pymc_models import PyMCNotInstalledError

        message = str(PyMCNotInstalledError())

        self.assertIn("python3 -m pip install", message)
        self.assertIn("macos-arm64", message)
        self.assertIn("PyMC", message)

    def test_blue_dirichlet_analytic_posterior_without_sampling(self):
        from ssq_model.models.pymc_models import PyMCBlueDirichletModel

        draws = [
            Draw("1", "2026-01-01", (1, 2, 3, 4, 5, 6), 1),
            Draw("2", "2026-01-04", (2, 3, 4, 5, 6, 7), 1),
            Draw("3", "2026-01-07", (3, 4, 5, 6, 7, 8), 16),
        ]

        prediction = PyMCBlueDirichletModel(alpha=1.0).fit(draws).predict_proba()

        self.assertEqual(prediction.model_name, "pymc_blue_dirichlet")
        self.assertAlmostEqual(sum(prediction.blue_probs.values()), 1.0)
        self.assertAlmostEqual(sum(prediction.red_probs.values()), 6.0)
        self.assertGreater(prediction.blue_probs[1], prediction.blue_probs[2])
        self.assertGreater(prediction.blue_probs[16], prediction.blue_probs[2])

    def test_red_beta_binomial_analytic_posterior_without_sampling(self):
        from ssq_model.models.pymc_models import PyMCRedBetaBinomialModel

        draws = [
            Draw("1", "2026-01-01", (1, 2, 3, 4, 5, 6), 1),
            Draw("2", "2026-01-04", (1, 2, 3, 7, 8, 9), 2),
        ]

        prediction = PyMCRedBetaBinomialModel(alpha=1.0, beta=1.0).fit(draws).predict_proba()

        self.assertEqual(prediction.model_name, "pymc_red_beta_binomial")
        self.assertAlmostEqual(sum(prediction.red_probs.values()), 6.0)
        self.assertAlmostEqual(sum(prediction.blue_probs.values()), 1.0)
        self.assertGreater(prediction.red_probs[1], prediction.red_probs[10])

    def test_joint_pymc_model_combines_red_and_blue_posteriors(self):
        from ssq_model.models.pymc_models import PyMCJointBayesianModel

        draws = [
            Draw("1", "2026-01-01", (1, 2, 3, 4, 5, 6), 1),
            Draw("2", "2026-01-04", (1, 2, 3, 7, 8, 9), 2),
        ]

        prediction = PyMCJointBayesianModel(alpha=1.0, beta=1.0).fit(draws).predict_proba()

        self.assertEqual(prediction.model_name, "pymc_joint_bayesian")
        self.assertAlmostEqual(sum(prediction.red_probs.values()), 6.0)
        self.assertAlmostEqual(sum(prediction.blue_probs.values()), 1.0)
        self.assertGreater(prediction.red_probs[1], prediction.red_probs[10])
        self.assertGreater(prediction.blue_probs[1], prediction.blue_probs[16])


if __name__ == "__main__":
    unittest.main()
