import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_text(name):
    return (FIXTURES / name).read_bytes().decode("latin-1")


@pytest.fixture
def data_dir(tmp_path):
    """An empty data dir with raw/omie. data_dir.add(name, text=None) copies a fixture into
    raw/omie, or writes the given text under that name."""
    raw = tmp_path / "raw" / "omie"
    raw.mkdir(parents=True)

    class DataDir(type(tmp_path)):
        pass

    def add(name, text=None):
        if text is None:
            shutil.copy(FIXTURES / name, raw / name)
        else:
            (raw / name).write_bytes(text.encode("latin-1"))
        return raw / name

    d = DataDir(tmp_path)
    d.add = add
    d.raw = raw
    return d
