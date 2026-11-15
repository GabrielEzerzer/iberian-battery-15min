# Tests for the ib15 import and layer rules: each rule fires on a bad sample and stays quiet on a good one.
import re
import shutil
import unittest
from pathlib import Path

from test_rules import AST_GREP_MISSING, scan_matches

RULES = Path(__file__).resolve().parent.parent / "rules" / "iberian-battery-15min-layers.yml"

CASES = {
    "ib15-layer-omie": (
        "src/ib15/omie.py",
        "from ib15.dataset import BuildError\n",
        "import urllib.request\nfrom datetime import date\n",
    ),
    "ib15-layer-dataset": (
        "src/ib15/dataset.py",
        "from ib15.battery import POWER_MW\n",
        "from ib15.omie import FILE_RE, read_day\nimport ib15.omie\n",
    ),
    "ib15-layer-battery-weather": (
        "src/ib15/weather.py",
        "from ib15.battery import SPECS\n",
        "from ib15.dataset import BuildError\nfrom ib15.omie import MADRID\n",
    ),
    "ib15-layer-eda": (
        "src/ib15/eda.py",
        "from ib15 import forecast\n",
        "from ib15.battery import DURATIONS_H\nfrom ib15.weather import LOCATIONS\n",
    ),
    "ib15-library-imports-no-cli": (
        "src/ib15/forecast.py",
        "from ib15.cli import main\n",
        "from ib15.eda import quarter_deviation\n",
    ),
    "ib15-library-imports-no-tests": (
        "src/ib15/battery.py",
        "import pytest\n",
        "import numpy as np\n",
    ),
    "ib15-network-only-in-omie": (
        "src/ib15/weather.py",
        "import urllib.request\n",
        "from urllib.parse import urlencode\nfrom ib15.omie import http_get\n",
    ),
    "ib15-settings-only-in-cli": (
        "src/ib15/battery.py",
        "import os\n\nDATA = os.environ.get('IB15_DATA_DIR')\n",
        "import os\n\nNAME = os.path.basename('a/b')\n",
    ),
    "ib15-plots-only-in-eda": (
        "src/ib15/forecast.py",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
    ),
    "ib15-lp-only-in-battery": (
        "src/ib15/forecast.py",
        "from scipy.optimize import linprog\n",
        "from scipy.stats import kendalltau\n",
    ),
    "ib15-tests-load-fixtures-through-conftest": (
        "tests/test_parse.py",
        "from pathlib import Path\n\nRAW = Path(__file__).parent / 'fixtures'\n",
        "from conftest import FIXTURES\n\nRAW = FIXTURES / 'marginalpdbc_20251001.1'\n",
    ),
}

ALLOWED_HOME = {
    "ib15-network-only-in-omie": ("src/ib15/omie.py", "import urllib.request\nimport urllib.error\n"),
    "ib15-settings-only-in-cli": ("src/ib15/cli.py", "import argparse\nimport os\n\nD = os.environ.get('X')\n"),
    "ib15-plots-only-in-eda": ("src/ib15/eda.py", "import matplotlib\nimport matplotlib.pyplot as plt\n"),
    "ib15-lp-only-in-battery": ("src/ib15/battery.py", "from scipy.optimize import linprog\n"),
    "ib15-library-imports-no-cli": ("src/ib15/__main__.py", "from ib15.cli import main\n"),
    "ib15-tests-load-fixtures-through-conftest": (
        "tests/conftest.py",
        "from pathlib import Path\n\nFIXTURES = Path(__file__).parent / 'fixtures'\n",
    ),
}


def setUpModule():
    if shutil.which("ast-grep") is None:
        raise RuntimeError(AST_GREP_MISSING)


def rules_hit(path, text):
    return {m["ruleId"] for m in scan_matches({path: text}) if m["ruleId"].startswith("ib15-")}


class Layers(unittest.TestCase):
    def test_each_rule_fires_on_a_bad_sample(self):
        for rule, (path, bad, _) in CASES.items():
            with self.subTest(rule=rule):
                self.assertIn(rule, rules_hit(path, bad))

    def test_no_rule_fires_on_a_good_sample(self):
        for rule, (path, _, good) in CASES.items():
            with self.subTest(rule=rule):
                self.assertEqual(rules_hit(path, good), set())

    def test_the_allowed_home_of_each_import_passes(self):
        for rule, (path, text) in ALLOWED_HOME.items():
            with self.subTest(rule=rule):
                self.assertEqual(rules_hit(path, text), set())

    def test_every_rule_has_a_case(self):
        ids = set(re.findall(r"(?m)^id: (\S+)$", RULES.read_text(encoding="utf-8")))
        self.assertEqual(ids, set(CASES))

    def test_messages_say_what_to_do_instead(self):
        for rule, (path, bad, _) in CASES.items():
            with self.subTest(rule=rule):
                messages = [m["message"] for m in scan_matches({path: bad}) if m["ruleId"] == rule]
                self.assertTrue(messages)
                for message in messages:
                    self.assertRegex(message, r"(?:^|[.] )(Move|Return|Put|Take|Pass|Call|Import) ")


if __name__ == "__main__":
    unittest.main()
