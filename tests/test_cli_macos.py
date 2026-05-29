import unittest


class TestCliMacOS(unittest.TestCase):
    def test_parser_accepts_pymc_and_mac_commands(self):
        from ssq_model.cli import build_parser

        parser = build_parser()

        parser.parse_args(["pymc-fit", "--quick", "--no-sample"])
        parser.parse_args(["pymc-predict", "--top-k", "3", "--no-sample"])
        parser.parse_args(["mac-bootstrap", "--print-commands"])

    def test_mac_bootstrap_commands_use_python3(self):
        from ssq_model.cli import mac_bootstrap_commands

        commands = "\n".join(mac_bootstrap_commands())

        self.assertIn("python3 -m venv", commands)
        self.assertIn("python3 -m pip install", commands)
        self.assertIn(".[macos-arm64]", commands)


if __name__ == "__main__":
    unittest.main()
