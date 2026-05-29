import unittest
from pathlib import Path


class TestMacOSDocs(unittest.TestCase):
    def test_macos_deployment_assets_are_apple_silicon_oriented(self):
        env = Path("environment-macos-arm64.yml").read_text(encoding="utf-8")
        script = Path("scripts/bootstrap_macos.sh").read_text(encoding="utf-8")
        docs = Path("docs/MACOS_APPLE_SILICON.md").read_text(encoding="utf-8")
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("pymc", env)
        self.assertIn("numpyro", env)
        self.assertIn("blackjax", env)
        self.assertIn("uname -m", script)
        self.assertIn("arm64", script)
        self.assertIn("python3 -m venv", script)
        self.assertIn("python3 -m pip install", script)
        self.assertIn("M4 Mac mini", docs)
        self.assertIn("python3 -m ssq_model pymc-fit", docs)
        self.assertIn("macos-arm64", pyproject)


if __name__ == "__main__":
    unittest.main()
