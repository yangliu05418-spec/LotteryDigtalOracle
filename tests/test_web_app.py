import unittest

from ssq_model.data import Draw


class TestWebApp(unittest.TestCase):
    def test_dashboard_summary_contains_data_and_environment_status(self):
        from ssq_model.web.state import build_dashboard_summary

        draws = [
            Draw("2026001", "2026-01-01", (1, 2, 3, 4, 5, 6), 1),
            Draw("2026002", "2026-01-04", (2, 3, 4, 5, 6, 7), 2),
        ]

        summary = build_dashboard_summary(draws, data_path="数据/历史数据.csv")

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["latest_issue"], "2026002")
        self.assertEqual(summary["total_combinations"], "17,721,088")
        self.assertIn("python_version", summary)
        self.assertIn("pymc_available", summary)
        self.assertIn("academic_notice", summary)

    def test_frequency_chart_spec_is_testable_without_plotly(self):
        from ssq_model.web.charts import build_frequency_chart_spec

        spec = build_frequency_chart_spec({1: 0.2, 2: 0.3}, title="测试图", x_label="球号", y_label="概率")

        self.assertEqual(spec.title, "测试图")
        self.assertEqual(spec.x, ["1", "2"])
        self.assertEqual(spec.y, [0.2, 0.3])

    def test_candidate_action_returns_table_rows(self):
        from ssq_model.web.actions import generate_candidate_rows

        draws = [
            Draw("1", "2026-01-01", (1, 2, 3, 4, 5, 6), 1),
            Draw("2", "2026-01-04", (1, 2, 3, 7, 8, 9), 2),
        ]

        rows = generate_candidate_rows(draws, model_name="bayesian", top_k=3, seed=7, pool_size=30)

        self.assertEqual(len(rows), 3)
        self.assertIn("红球", rows[0])
        self.assertIn("蓝球", rows[0])
        self.assertIn("score", rows[0])

    def test_streamlit_app_exposes_main_function_without_importing_streamlit(self):
        import ssq_model.web_app as app

        self.assertTrue(callable(app.main))
        self.assertTrue(callable(app.run_streamlit_app))


if __name__ == "__main__":
    unittest.main()
