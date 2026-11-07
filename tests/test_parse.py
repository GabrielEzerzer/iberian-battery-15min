from datetime import date

import pytest
from conftest import FIXTURES, fixture_text

from ib15.omie import OmieFormatError, expected_periods, read_day


@pytest.mark.parametrize(
    "name, periods",
    [
        ("marginalpdbc_20251001.1", 96),
        ("marginalpdbc_20251026.1", 100),  # last Sunday of October, 25 h
        ("marginalpdbc_20260329.1", 92),  # last Sunday of March, 23 h
    ],
)
def test_parses_normal_and_dst_days(name, periods):
    d = read_day(FIXTURES / name)
    assert [r[0] for r in d.rows] == list(range(1, periods + 1))
    assert expected_periods(d.day) == periods
    assert d.version == 1


def test_reads_values_and_version_from_file():
    d = read_day(FIXTURES / "marginalpdbc_20251001.1")
    assert d.day == date(2025, 10, 1)
    assert d.rows[0] == (1, 105.1, 105.1)
    assert d.rows[-1] == (96, 101.52, 101.52)
    assert read_day(FIXTURES / "marginalpdbc_20251030.3").version == 3


def test_rejects_hourly_file_on_or_after_go_live(tmp_path):
    # The real 2025-09-30 hourly file, relabelled as 2025-10-01.
    text = fixture_text("marginalpdbc_20250930.1").replace("2025;09;30;", "2025;10;01;")
    p = tmp_path / "marginalpdbc_20251001.1"
    p.write_bytes(text.encode("latin-1"))
    with pytest.raises(OmieFormatError, match=r"hourly file \(24 periods\)"):
        read_day(p)


def test_rejects_day_before_go_live():
    with pytest.raises(OmieFormatError, match="before 2025-10-01, out of scope"):
        read_day(FIXTURES / "marginalpdbc_20250930.1")


def test_rejects_missing_trailer(tmp_path):
    text = fixture_text("marginalpdbc_20251001.1").replace("*", "")
    p = tmp_path / "marginalpdbc_20251001.1"
    p.write_bytes(text.encode("latin-1"))
    with pytest.raises(OmieFormatError, match="missing '\\*' trailer"):
        read_day(p)
