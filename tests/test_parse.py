from datetime import date

import pytest
from conftest import FIXTURES, fixture_text

from ib15.omie import OmieFormatError, expected_periods, read_day


@pytest.mark.parametrize(
    "name, periods",
    [
        pytest.param("marginalpdbc_20251001.1", 96, id="normal-day"),
        pytest.param("marginalpdbc_20251026.1", 100, id="last-sunday-of-october-25h"),
        pytest.param("marginalpdbc_20260329.1", 92, id="last-sunday-of-march-23h"),
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
    pt_and_es_differ = (
        "Portugal and Spain differ here, so the order is (period, price_pt, price_es)"
    )
    assert d.rows[39] == (40, 60.87, 60.0), pt_and_es_differ
    assert d.rows[72] == (73, 60.0, 59.07), pt_and_es_differ
    assert read_day(FIXTURES / "marginalpdbc_20251030.3").version == 3


def test_rejects_hourly_file_on_or_after_go_live(tmp_path):
    real_2025_09_30_hourly_file_relabelled = fixture_text("marginalpdbc_20250930.1").replace(
        "2025;09;30;", "2025;10;01;"
    )
    p = tmp_path / "marginalpdbc_20251001.1"
    p.write_bytes(real_2025_09_30_hourly_file_relabelled.encode("latin-1"))
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
