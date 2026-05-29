import unittest
from pathlib import Path


class TestWebDeploy(unittest.TestCase):
    def test_cli_parser_accepts_web_and_print_command_uses_streamlit(self):
        from ssq_model.cli import build_parser, streamlit_launch_command

        parser = build_parser()
        parser.parse_args(["web", "--print-command"])
        command = streamlit_launch_command(port=7860)

        self.assertIn("python3", command)
        self.assertIn("streamlit", command)
        self.assertIn("ssq_model/web_app.py", command)
        self.assertIn("--server.port=7860", command)

    def test_macos_web_scripts_are_one_click_and_arm64_specific(self):
        install = Path("scripts/install_and_run_macos.sh").read_text(encoding="utf-8")
        run = Path("scripts/run_web_macos.sh").read_text(encoding="utf-8")

        self.assertIn("uname -m", install)
        self.assertIn("arm64", install)
        self.assertIn("Miniforge3-MacOSX-arm64.sh", install)
        self.assertIn("conda env create", install)
        self.assertIn("python3 -m pip install -e '.[macos-arm64,web]'", install)
        self.assertIn("streamlit run ssq_model/web_app.py", install)
        self.assertIn("conda activate ssq-model-macos-arm64", run)
        self.assertIn("streamlit run ssq_model/web_app.py", run)

    def test_pyproject_and_environment_include_web_dependencies(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        env = Path("environment-macos-arm64.yml").read_text(encoding="utf-8")

        self.assertIn("web = [", pyproject)
        self.assertIn("streamlit", pyproject)
        self.assertIn("plotly", pyproject)
        self.assertIn("streamlit", env)
        self.assertIn("plotly", env)

    def test_readme_contains_one_click_web_deployment(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("scripts/install_and_run_macos.sh", readme)
        self.assertIn("scripts/run_web_macos.sh", readme)
        self.assertIn("python3 -m ssq_model web", readme)
        self.assertIn("Streamlit", readme)


if __name__ == "__main__":
    unittest.main()
