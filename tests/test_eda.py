from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ib15 import cli, eda
from ib15.dataset import hourly_benchmark
from ib15.omie import expected_periods, local_midnight_utc
from ib15.weather import LOCATIONS, VARIABLES


def prices(days, price_fn=None):
    """A prices_15min frame for consecutive local delivery days starting at ``days[0]``, with
    each day's real period count (92/96/100) and prices from price_fn(i_global) or random."""
    rows, i = [], 0
    rng = np.random.default_rng(0)
    for day in days:
        start = pd.Timestamp(local_midnight_utc(day))
        for k in range(expected_periods(day)):
            p = price_fn(i) if price_fn else float(np.round(rng.normal(60, 30), 2))
            rows.append(
                {
                    "delivery_start_utc": start + pd.Timedelta(minutes=15 * k),
                    "delivery_date": day,
                    "period": k + 1,
                    "price_es": p,
                    "price_pt": p,
                    "source_version": 1,
                }
            )
            i += 1
    df = pd.DataFrame(rows)
    df["delivery_start_utc"] = pd.to_datetime(df["delivery_start_utc"], utc=True)
    return df


def test_within_hour_spread_and_flat_share_on_dst_day():
    # 2025-10-26 is the autumn DST day: 100 periods, 25 hours of the local delivery date.
    day = date(2025, 10, 26)
    q = prices([day], price_fn=lambda i: 50.0)
    q.loc[q["period"].between(1, 4), "price_es"] = [10.0, 30.0, 20.0, 40.0]  # spread 30
    q.loc[q["period"].between(97, 100), "price_es"] = [5.0, 5.0, 5.0, 9.0]  # 25th hour, spread 4
    h = eda.hour_table(q)
    assert len(h) == 25
    assert h.loc[0, "spread"] == 30.0
    assert h.loc[0, "slope"] == 30.0  # 40 - 10
    assert h.loc[24, "spread"] == 4.0
    assert h.loc[24, "hour"] == 24
    s = eda.spread_summary(h)
    assert s["hours"] == 25
    assert s["share_flat_hours"] == pytest.approx(23 / 25)
    assert s["mean_spread"] == pytest.approx(34 / 25)
    # first hour: mean 25, |dev| = 15, 5, 5, 15 -> 10
    assert h.loc[0, "abs_dev"] == pytest.approx(10.0)


def test_quarter_deviations_sum_to_zero_within_each_hour():
    q = prices([date(2025, 10, 25), date(2025, 10, 26)])
    dev = eda.quarter_deviation(q)
    sums = dev.groupby([q["delivery_date"], (q["period"] - 1) // 4]).sum()
    assert len(sums) == 24 + 25
    assert np.allclose(sums, 0, atol=1e-9)
    assert dev.abs().sum() > 0


def test_concentration_shares_on_hand_made_vector():
    # 20 days: two big days, four medium, ten small, four zero.
    u = [50, 30] + [10] * 4 + [1] * 10 + [0] * 4
    c = eda.concentration(u)
    total = 130
    assert c["share_from_top_10pct_days"] == pytest.approx(80 / total)  # 2 days
    assert c["share_from_top_25pct_days"] == pytest.approx(110 / total)  # 5 days
    assert c["share_days_zero_uplift"] == pytest.approx(4 / 20)
    assert c["median_daily_uplift_eur_per_mw"] == 1.0
    assert c["mean_daily_uplift_eur_per_mw"] == pytest.approx(130 / 20)


def test_concentration_rounds_top_share_up_to_whole_days():
    c = eda.concentration([5, 4, 3, 2, 1])  # 10% of 5 days -> 1 day, 25% -> 2 days
    assert c["share_from_top_10pct_days"] == pytest.approx(5 / 15)
    assert c["share_from_top_25pct_days"] == pytest.approx(9 / 15)


def test_autocorr_periodic_and_sign_flipping():
    idx = pd.date_range("2025-10-01", periods=96 * 20, freq="15min", tz="UTC")
    rng = np.random.default_rng(1)
    day = rng.normal(size=96)
    periodic = pd.Series(np.tile(day, 20), index=idx)
    r, n = eda.autocorr(periodic, pd.Timedelta(days=1))
    assert r == pytest.approx(1.0)
    assert n == 96 * 19
    flip = pd.Series(np.concatenate([day * (-1) ** k for k in range(20)]), index=idx)
    r, _ = eda.autocorr(flip, pd.Timedelta(days=1))
    assert r == pytest.approx(-1.0)
    r7, n7 = eda.autocorr(flip, pd.Timedelta(days=7))
    assert r7 == pytest.approx(-1.0)  # odd lag in days
    assert n7 == 96 * 13


def weather_frame(hours):
    """Hourly weather for the 10 locations: location j has value j + hour index for every
    variable, so the national mean at hour i is 4.5 + i."""
    rows = []
    for i, t in enumerate(hours):
        for j, loc in enumerate(LOCATIONS):
            row = {"valid_time_utc": t, "location": loc, "model": "ecmwf_ifs025"}
            row.update({v: float(j + i) for v in VARIABLES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_national_mean_averages_the_ten_locations_at_each_hour():
    hours = pd.date_range("2025-10-01", periods=5, freq="h", tz="UTC")
    w = weather_frame(hours).sample(frac=1, random_state=0)  # order must not matter
    m = eda.national_mean(w)
    assert list(m.index) == list(hours)
    for i, t in enumerate(hours):
        for v in VARIABLES:
            assert m.loc[t, v] == pytest.approx(4.5 + i)


def test_national_mean_rejects_a_missing_location_hour():
    hours = pd.date_range("2025-10-01", periods=3, freq="h", tz="UTC")
    w = weather_frame(hours)
    w = w.drop(w.index[(w["location"] == "lugo") & (w["valid_time_utc"] == hours[1])])
    with pytest.raises(eda.BuildError):
        eda.national_mean(w)


@pytest.mark.parametrize(
    "month,expected",
    [(11, "autumn"), (12, "winter"), (2, "winter"), (3, "spring"), (5, "spring"), (6, "summer")]
    + [(8, "summer"), (9, "autumn"), (1, "winter")],
)
def test_season_boundaries(month, expected):
    assert eda.season(month) == expected


def write_data_dir(root, with_sensitivity):
    days = [date(2025, 10, 20) + timedelta(days=i) for i in range(12)]  # includes 10-26 DST
    q = prices(days)
    h = hourly_benchmark(q)
    ds, res = root / "dataset", root / "results"
    ds.mkdir(parents=True)
    res.mkdir()
    q.to_parquet(ds / "prices_15min.parquet", index=False)
    h.to_parquet(ds / "prices_hourly.parquet", index=False)
    hours = pd.DatetimeIndex(q["delivery_start_utc"].dt.floor("h").unique())
    weather_frame(hours).to_parquet(ds / "weather_hourly.parquet", index=False)
    rows = []
    for i, d in enumerate(days):
        for dur in (1, 2, 4):
            for res_, net in (("15min", 100.0 + i + dur), ("hourly", 95.0 + dur)):
                rows.append(
                    {
                        "delivery_date": d,
                        "duration_h": dur,
                        "spec": "main",
                        "resolution": res_,
                        "net_eur_per_mw": net,
                    }
                )
    pd.DataFrame(rows).to_parquet(res / "perfect_foresight_daily.parquet", index=False)
    if with_sensitivity:
        s = pd.DataFrame(
            [
                {
                    "duration_h": dur,
                    "cost_eur_per_mwh": c,
                    "net_uplift_keur_per_mw": 1.0,
                    "net_uplift_eur_per_mw_day": 10.0 - c / 10,
                    "net_uplift_pct": 5.0 + c / 10,
                }
                for dur in (1, 2, 4)
                for c in (0, 10, 20, 30, 40)
            ]
        )
        s.to_csv(res / "perfect_foresight_cost_sensitivity.csv", index=False)


TABLES = [
    "price_summary.csv",
    "within_hour_spread.csv",
    "uplift_by_month.csv",
    "uplift_concentration.csv",
    "weather_summary.csv",
    "forecast_motivation.csv",
]
FIGURES = [
    "fig_a_daily_price_and_spread",
    "fig_b_profile_by_season",
    "fig_c_quarter_deviation_by_hour",
    "fig_d_spread_distribution",
    "fig_e_uplift_by_month",
    "fig_f_uplift_vs_spread",
    "fig_h_price_vs_weather",
]


@pytest.mark.parametrize("with_sensitivity", [False, True])
def test_eda_command_writes_every_output(tmp_path, capsys, with_sensitivity):
    write_data_dir(tmp_path, with_sensitivity)
    assert cli.main(["eda", "--data-dir", str(tmp_path)]) == 0
    out = tmp_path / "results" / "eda"
    figures = FIGURES + (["fig_g_cost_sensitivity"] if with_sensitivity else [])
    expected = set(TABLES) | {f"{f}.{e}" for f in figures for e in ("png", "pdf")}
    expected.add("eda_meta.json")
    assert {p.name for p in out.iterdir()} == expected
    assert all((out / name).stat().st_size > 0 for name in expected)
    printed = capsys.readouterr().out
    assert ("skipped fig_g_cost_sensitivity" in printed) is not with_sensitivity
    conc = pd.read_csv(out / "uplift_concentration.csv")
    assert list(conc["duration_h"]) == [1, 2, 4]
    assert conc["mean_daily_uplift_eur_per_mw"].tolist() == pytest.approx([10.5] * 3)
    ps = pd.read_csv(out / "price_summary.csv").set_index("series")
    assert ps.loc["15min", "periods"] == 12 * 96 + 4
    assert ps.loc["15min", "days"] == 12


def test_eda_command_fails_clearly_on_missing_input(tmp_path, capsys):
    write_data_dir(tmp_path, False)
    (tmp_path / "dataset" / "weather_hourly.parquet").unlink()
    assert cli.main(["eda", "--data-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "weather_hourly.parquet" in err
    assert not (tmp_path / "results" / "eda").exists()
