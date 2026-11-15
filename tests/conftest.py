import shutil
from datetime import timedelta
from pathlib import Path

import pytest
from hypothesis import settings

from ib15.omie import GO_LIVE, expected_periods, file_name

FIXTURES = Path(__file__).parent / "fixtures"

settings.register_profile("ib15", derandomize=True, deadline=None, max_examples=60)
settings.load_profile("ib15")


def fixture_text(name):
    return (FIXTURES / name).read_bytes().decode("latin-1")


def day_text(day):
    assert expected_periods(day) == 96
    return fixture_text("marginalpdbc_20251001.1").replace("2025;10;01;", f"{day:%Y;%m;%d};")


@pytest.fixture
def data_dir(tmp_path):
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

    def fill(until):
        day = GO_LIVE
        while day < until:
            name = file_name(day, 1)
            if (FIXTURES / name).exists():
                add(name)
            else:
                add(name, day_text(day))
            day += timedelta(days=1)

    d = DataDir(tmp_path)
    d.add = add
    d.fill = fill
    d.raw = raw
    return d
