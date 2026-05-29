import unittest
from pathlib import Path


class TestDeploymentFixes(unittest.TestCase):
    def test_pyproject_has_no_utf8_bom(self):
        self.assertFalse(Path("pyproject.toml").read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_web_app_uses_absolute_ssq_model_imports_for_streamlit_script_mode(self):
        text = Path("ssq_model/web_app.py").read_text(encoding="utf-8")
        self.assertIn("from ssq_model.data import", text)
        self.assertIn("from ssq_model.web.actions import", text)
        self.assertNotIn("from .data import", text)
        self.assertNotIn("from .web.", text)

    def test_pyproject_restricts_setuptools_package_discovery_to_ssq_model(self):
        text = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("[tool.setuptools.packages.find]", text)
        self.assertIn('include = ["ssq_model*"]', text)
        self.assertIn('exclude = ["数据*", "outputs*", "tests*", "docs*", "scripts*"]', text)


if __name__ == "__main__":
    unittest.main()
