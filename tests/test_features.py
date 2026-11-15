import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest import mock

import pytest

from ib15 import cli

ROOT = Path(__file__).resolve().parent.parent
FEATURES = json.loads((ROOT / "features.json").read_text(encoding="utf-8"))
FIELDS = {"name", "description", "command", "inputs", "outputs", "test"}


class Captured(Exception):
    pass


def cli_parser() -> argparse.ArgumentParser:
    def capture(self, args=None, namespace=None):
        raise Captured(self)

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
        with pytest.raises(Captured) as caught:
            cli.main([])
    return caught.value.args[0]


def leaf_commands(parser: argparse.ArgumentParser, prefix: tuple[str, ...]) -> set[str]:
    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if subs:
        return {
            cmd
            for name, child in subs[0].choices.items()
            for cmd in leaf_commands(child, (*prefix, name))
        }
    commands = {" ".join(prefix)}
    for action in parser._actions:
        if not action.option_strings and action.choices:
            commands |= {" ".join((*prefix, c)) for c in action.choices if c != action.default}
    return commands


def script_names() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return project["scripts"]


def test_every_entry_has_the_fields_of_the_map():
    for entry in FEATURES:
        assert set(entry) == FIELDS, f"{entry.get('name')}: fields must be exactly {sorted(FIELDS)}"
        assert entry["inputs"] and entry["outputs"], f"{entry['name']}: list inputs and outputs"


def test_every_cli_command_is_in_the_feature_map():
    mapped = {e["command"] for e in FEATURES}
    missing = sorted(leaf_commands(cli_parser(), ("ib15",)) - mapped)
    assert not missing, f"commands missing from features.json: {missing}; add an entry for each"


def test_every_mapped_command_exists():
    unknown = sorted({e["command"] for e in FEATURES} - leaf_commands(cli_parser(), ("ib15",)))
    assert not unknown, f"features.json maps commands the CLI does not have: {unknown}"
    for script, target in script_names().items():
        module, _, func = target.partition(":")
        assert importlib.util.find_spec(module), f"script {script}: module {module} does not exist"
        assert callable(getattr(importlib.import_module(module), func, None)), target


def test_every_project_script_is_in_the_feature_map():
    mapped = {e["command"].split()[0] for e in FEATURES}
    missing = sorted(set(script_names()) - mapped)
    assert not missing, f"scripts missing from features.json: {missing}"


def test_every_mapped_test_file_exists():
    for entry in FEATURES:
        test = entry["test"]
        if test.startswith("none"):
            assert "#" in test, f"{entry['name']}: an untested feature names its debt issue"
            continue
        assert (ROOT / test).is_file(), f"{entry['name']}: test file {test} does not exist"


@pytest.mark.parametrize("command", sorted({e["command"] for e in FEATURES}))
def test_help_of_every_mapped_command_exits_zero(command):
    proc = subprocess.run(
        [sys.executable, "-m", "ib15", *command.split()[1:], "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"`{command} --help` exited {proc.returncode}: {proc.stderr}"
