import unittest
from pathlib import Path


class TestDocumentation(unittest.TestCase):
    def test_readme_states_academic_boundary_and_optional_gpu_pymc(self):
        text = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("学术", text)
        self.assertIn("不构成投注建议", text)
        self.assertIn("GPU 非必需", text)
        self.assertIn("PyMC", text)
        self.assertIn("推荐必装", text)
        self.assertIn("Apple Silicon", text)
        self.assertIn("python3 -m ssq_model analyze", text)
        self.assertIn("python3 -m ssq_model backtest", text)
        self.assertIn("python3 -m ssq_model report", text)


if __name__ == "__main__":
    unittest.main()
