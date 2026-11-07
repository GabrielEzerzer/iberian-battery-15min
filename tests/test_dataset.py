import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from conftest import fixture_text

from ib15.cli import main
from ib15.dataset import BuildError, build

OCT = ["marginalpdbc_20251025.1", "marginalpdbc_20251026.1", "marginalpdbc_20251027.1"]
MAR = ["marginalpdbc_20260328.1", "marginalpdbc_20260329.1", "marginalpdbc_20260330.1"]
GOOD = fixture_text("marginalpdbc_20251001.1")


def built(data_dir, names):
    for n in names:
        data_dir.add(n)
    build(data_dir)
    out = data_dir / "dataset"
    return (
        pd.read_parquet(out / "prices_15min.parquet"),
        pd.read_parquet(out / "prices_hourly.parquet"),
        json.loads((out / "coverage.json").read_text()),
    )


@pytest.mark.parametrize(
    "names, dst_day, n, first_utc",
    [
        (OCT, date(2025, 10, 26), 100, datetime(2025, 10, 25, 22, tzinfo=UTC)),
        (MAR, date(2026, 3, 29), 92, datetime(2026, 3, 28, 23, tzinfo=UTC)),
    ],
)
def test_utc_starts_unique_and_15min_apart_across_dst_days(data_dir, names, dst_day, n, first_utc):
    q, _, _ = built(data_dir, names)
    ts = q["delivery_start_utc"]
    assert str(ts.dt.tz) == "UTC"
    assert ts.is_unique
    assert (ts.diff().dropna() == pd.Timedelta(minutes=15)).all()  # across both midnights
    day = q[q["delivery_date"] == dst_day]
    assert day["period"].tolist() == list(range(1, n + 1))
    assert day["delivery_start_utc"].iloc[0] == first_utc
    assert day["delivery_start_utc"].iloc[-1] == first_utc + timedelta(minutes=15 * (n - 1))


@pytest.mark.parametrize("names", [OCT, MAR])
def test_hourly_is_mean_of_its_four_quarter_hours(data_dir, names):
    q, h, _ = built(data_dir, names)
    assert list(h.columns) == list(q.columns)
    for d, qd in q.groupby("delivery_date"):
        hd = h[h["delivery_date"] == d].reset_index(drop=True)
        assert len(hd) == len(qd) / 4
        assert hd["period"].tolist() == list(range(1, len(hd) + 1))
        for i, row in hd.iterrows():
            four = qd.iloc[4 * i : 4 * i + 4]
            assert row["price_es"] == pytest.approx(four["price_es"].mean())
            assert row["price_pt"] == pytest.approx(four["price_pt"].mean())
            assert row["delivery_start_utc"] == four["delivery_start_utc"].iloc[0]


def test_coverage(data_dir):
    q, _, cov = built(data_dir, OCT)
    assert cov["date_range"] == {"start": "2025-10-25", "end": "2025-10-27"}
    assert (cov["days"], cov["periods_15min"], cov["hours"]) == (3, 292, 73)
    assert cov["days_not_96_periods"] == {"2025-10-26": {"expected": 100, "actual": 100}}
    assert cov["missing_periods"] == 0 and cov["duplicate_periods"] == 0
    assert cov["days_not_version_1"] == {}
    assert cov["price_es"] == {
        "min": q["price_es"].min(),
        "max": q["price_es"].max(),
        "mean": pytest.approx(q["price_es"].mean()),
    }
    assert cov["units"] == "EUR/MWh"
    assert "git_commit" in cov


def test_coverage_records_non_v1_day(data_dir):
    q, _, cov = built(data_dir, ["marginalpdbc_20251030.3"])
    assert cov["days_not_version_1"] == {"2025-10-30": 3}
    assert set(q["source_version"]) == {3}


def _with_period_5(edit):
    lines = GOOD.splitlines(keepends=True)
    i = next(i for i, ln in enumerate(lines) if ln.startswith("2025;10;01;5;"))
    return "".join(lines[:i] + edit(lines[i]) + lines[i + 1 :])


@pytest.mark.parametrize(
    "text, message",
    [
        (
            _with_period_5(lambda ln: []),
            r"wrong period count, expected 96 for this day, got 95 \(missing \[5\]",
        ),
        (_with_period_5(lambda ln: [ln, ln]), "duplicate period 5"),
        (
            GOOD.replace("2025;10;01;7;", "2025;10;02;7;"),
            "row date 2025-10-02 differs from file date",
        ),
        (
            GOOD.replace("2025;10;01;3;102.28;102.28;", "2025;10;01;3;102.28;abc;"),
            "non-numeric price 'abc'",
        ),
        (
            GOOD.replace("2025;10;01;3;102.28;102.28;", "2025;10;01;3;nan;102.28;"),
            "non-numeric price 'nan'",
        ),
        (GOOD.replace("MARGINALPDBC;", "<!DOCTYPE html>", 1), "unexpected header"),
    ],
    ids=["missing", "duplicate", "date-mismatch", "non-numeric", "nan", "header"],
)
def test_bad_file_fails_build(data_dir, text, message):
    data_dir.add("marginalpdbc_20251001.1", text)
    with pytest.raises(BuildError, match=message):
        build(data_dir)
    assert not (data_dir / "dataset").exists()


def test_gap_in_days_fails_build(data_dir):
    data_dir.add("marginalpdbc_20251025.1")
    data_dir.add("marginalpdbc_20251027.1")
    with pytest.raises(
        BuildError, match="gap in day sequence: 2025-10-25 is followed by 2025-10-27"
    ):
        build(data_dir)


def test_two_versions_of_one_day_fail_build(data_dir):
    data_dir.add("marginalpdbc_20251001.1")
    data_dir.add("marginalpdbc_20251001.2", GOOD)
    with pytest.raises(BuildError, match="more than one version"):
        build(data_dir)


def test_pre_go_live_file_fails_build(data_dir):
    data_dir.add("marginalpdbc_20250930.1")
    data_dir.add("marginalpdbc_20251001.1")
    with pytest.raises(BuildError, match="before 2025-10-01, out of scope"):
        build(data_dir)


def test_cli_build_exits_nonzero_with_message(data_dir, capsys):
    data_dir.add("marginalpdbc_20251001.1", _with_period_5(lambda ln: []))
    assert main(["build", "--data-dir", str(data_dir)]) == 1
    assert "wrong period count" in capsys.readouterr().err


def test_cli_build_reads_data_dir_from_env(data_dir, monkeypatch):
    data_dir.add("marginalpdbc_20251001.1")
    monkeypatch.setenv("IB15_DATA_DIR", str(data_dir))
    assert main(["build"]) == 0
    assert (data_dir / "dataset" / "prices_15min.parquet").exists()


def test_cli_requires_data_dir(monkeypatch):
    monkeypatch.delenv("IB15_DATA_DIR", raising=False)
    with pytest.raises(SystemExit, match="pass --data-dir or set IB15_DATA_DIR"):
        main(["build"])
