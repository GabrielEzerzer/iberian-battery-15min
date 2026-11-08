import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ib15 import battery, cli, forecast
from ib15.dataset import BuildError, hourly_benchmark
from ib15.forecast import MODELS
from ib15.omie import expected_periods, local_midnight_utc
from ib15.weather import LOCATIONS, VARIABLES

WX_COLS = [f"{prefix}_{loc}" for prefix, _ in VARIABLES.values() for loc in LOCATIONS]
OUTPUTS = {
    "forecasts_15min.parquet",
    "forecast_errors.csv",
    "forecast_dm_tests.csv",
    "forecast_value_daily.parquet",
    "forecast_value_summary.csv",
    "forecast_meta.json",
}


def frame(start: date, ndays: int, seed: int = 0) -> pd.DataFrame:
    """A prices_weather_15min frame for consecutive local delivery days from ``start``, each with
    its real period count, random prices and random weather."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(ndays):
        day = start + timedelta(days=i)
        t0 = pd.Timestamp(local_midnight_utc(day))
        for k in range(expected_periods(day)):
            rows.append(
                {
                    "delivery_start_utc": t0 + pd.Timedelta(minutes=15 * k),
                    "delivery_date": day,
                    "period": k + 1,
                }
            )
    df = pd.DataFrame(rows)
    df["delivery_start_utc"] = pd.to_datetime(df["delivery_start_utc"], utc=True)
    df["price_es"] = np.round(rng.normal(60, 30, len(df)), 2)
    df["price_pt"] = df["price_es"]
    df["source_version"] = 1
    wx = rng.uniform(0, 30, (len(df), len(WX_COLS)))
    return pd.concat([df, pd.DataFrame(wx, columns=WX_COLS)], axis=1)


@pytest.fixture(scope="module")
def long_frame():
    # Covers both DST changes with more than 63 days before each.
    return frame(date(2025, 8, 15), 240)


def forecast_for(pw: pd.DataFrame, day: date) -> dict[str, np.ndarray]:
    """Each model's forecast for ``day``, mapped back to its real periods."""
    dates, p, w, slots = forecast.slot_arrays(pw)
    t = dates.index(day)
    fc = forecast.forecast_slots(dates, p, w, [t])
    return {m: fc[m][0][slots[t]] for m in MODELS}


def day_prices(pw: pd.DataFrame, day: date) -> np.ndarray:
    return pw.loc[pw["delivery_date"] == day, "price_es"].to_numpy()


# ---------------------------------------------------------------- 1, 2: information set


@pytest.mark.parametrize(
    "day",
    [date(2025, 10, 26), date(2025, 10, 27), date(2026, 3, 29), date(2026, 1, 14)],
    ids=["100-period Sunday", "Monday", "92-period Sunday", "Wednesday"],
)
def test_no_look_ahead(long_frame, day):
    base = forecast_for(long_frame, day)
    assert all(len(base[m]) == expected_periods(day) for m in MODELS)
    rng = np.random.default_rng(1)
    changed = long_frame.copy()
    on_or_after = changed["delivery_date"] >= day
    changed.loc[on_or_after, "price_es"] += rng.normal(0, 50, on_or_after.sum())
    after = changed["delivery_date"] > day
    changed.loc[after, WX_COLS] += rng.normal(0, 5, (after.sum(), len(WX_COLS)))
    new = forecast_for(changed, day)
    for m in MODELS:
        assert np.array_equal(new[m], base[m]), m

    # The information that is allowed does move the forecast.
    d1 = changed.copy()
    d1.loc[d1["delivery_date"] == day - timedelta(days=1), "price_es"] += 10.0
    assert not np.allclose(forecast_for(d1, day)["expert"], base["expert"])
    wx = long_frame.copy()
    wx.loc[wx["delivery_date"] == day, WX_COLS] += 3.0
    assert not np.allclose(forecast_for(wx, day)["expert_weather"], base["expert_weather"])


def test_window_reaches_back_exactly_63_days(long_frame):
    day = date(2026, 1, 14)
    base = forecast_for(long_frame, day)
    old = long_frame.copy()
    older = old["delivery_date"] < day - timedelta(days=63)
    old.loc[older, "price_es"] += 100.0
    new = forecast_for(old, day)
    for m in MODELS:
        assert np.array_equal(new[m], base[m]), m
    edge = long_frame.copy()
    edge.loc[edge["delivery_date"] == day - timedelta(days=63), "price_es"] += 100.0
    assert not np.allclose(forecast_for(edge, day)["expert"], base["expert"])


# ---------------------------------------------------------------- 3, 4: models


@pytest.mark.parametrize(
    "day, lag",
    [
        (date(2026, 1, 12), 7),  # Monday
        (date(2026, 1, 17), 7),  # Saturday
        (date(2026, 1, 18), 7),  # Sunday
        (date(2026, 1, 14), 1),  # Wednesday
    ],
)
def test_naive_rule(long_frame, day, lag):
    got = forecast_for(long_frame, day)["naive"]
    assert np.array_equal(got, day_prices(long_frame, day - timedelta(days=lag)))


def noise_free(with_weather: bool):
    """Prices generated without noise from the expert equation (plus weather terms), per slot."""
    rng = np.random.default_rng(1)
    n = 70
    dates = [date(2025, 11, 3) + timedelta(days=i) for i in range(n)]
    p = np.zeros((n, 96))
    p[:7] = rng.uniform(20, 120, (7, 96))
    beta = np.column_stack(
        [
            rng.uniform(-5, 5, 96),
            rng.uniform(0.3, 0.5, 96),
            rng.uniform(-0.2, 0.2, 96),
            rng.uniform(0.3, 0.45, 96),
            rng.uniform(-0.1, 0.1, (96, 3)),
            rng.uniform(-10, 10, (96, 3)),
        ]
    )
    w = rng.normal(size=(n, 96, 3)) * [5, 3, 200] + [15, 6, 200]
    gamma = rng.uniform(-1, 1, (96, 3)) * [1, 1, 0.05]
    for j in range(7, n):
        prev, wd = p[j - 1], dates[j].weekday()
        x = np.column_stack(
            [np.ones(96), prev, p[j - 2], p[j - 7]]
            + [np.full(96, v) for v in (prev.min(), prev.max(), prev[-1])]
            + [np.full(96, float(wd == k)) for k in (0, 5, 6)]
        )
        p[j] = (x * beta).sum(axis=1) + ((w[j] * gamma).sum(axis=1) if with_weather else 0)
    return dates, p, w


@pytest.mark.parametrize("model", ["expert", "expert_weather"])
def test_expert_recovers_a_noise_free_equation(model):
    dates, p, w = noise_free(model == "expert_weather")
    targets = [forecast.FIRST_EVAL, len(dates) - 1]
    got = forecast.forecast_slots(dates, p, w, targets)[model]
    np.testing.assert_allclose(got, p[targets], rtol=1e-9, atol=1e-8)


# ---------------------------------------------------------------- 5: DST slots


def test_dst_days_map_to_96_slots_and_back():
    pw = frame(date(2025, 10, 25), 3)
    pw = pd.concat([pw, frame(date(2026, 3, 28), 3)], ignore_index=True)
    for day, n in ((date(2026, 3, 29), 92), (date(2025, 10, 26), 100)):
        g = pw[pw["delivery_date"] == day]
        assert len(g) == n
        s = forecast.local_slots(g["delivery_start_utc"])
        values = np.column_stack([g["price_es"].to_numpy(), g[WX_COLS[0]].to_numpy()])
        slots = forecast.to_slots(values, s)
        assert slots.shape == (96, 2)
        back = slots[s]
        assert back.shape == (n, 2)
        if n == 92:
            # 02:00-02:59 does not exist: it takes 01:00-01:59.
            np.testing.assert_array_equal(slots[8:12], slots[4:8])
            np.testing.assert_array_equal(back, values)
        else:
            # 02:00-02:59 happens twice (periods 9-12 and 13-16): the slot is their mean, and
            # both occurrences get it back.
            np.testing.assert_allclose(slots[8:12], (values[8:12] + values[12:16]) / 2)
            np.testing.assert_allclose(back[8:12], slots[8:12])
            np.testing.assert_allclose(back[12:16], slots[8:12])
            keep = np.r_[0:8, 16:100]
            np.testing.assert_array_equal(back[keep], values[keep])


# ---------------------------------------------------------------- 6: battery value


def test_a_perfect_forecast_earns_the_perfect_foresight_value():
    rng = np.random.default_rng(3)
    q = np.round(rng.normal(60, 40, 96), 2)
    h = q.reshape(-1, 4).mean(axis=1)
    rows = pd.DataFrame(forecast.day_values(date(2026, 1, 1), q, h, {"exact": q.copy()}))
    v = rows.set_index(["duration_h", "model", "resolution"])["net_eur_per_mw"]
    for hours in battery.DURATIONS_H:
        for res, prices, dt in (("15min", q, 0.25), ("hourly", h, 1.0)):
            pf = battery.solve_day(prices, dt, hours, forecast.SPEC).net
            assert v[(hours, "perfect", res)] == pytest.approx(pf, abs=1e-9)
            assert v[(hours, "exact", res)] == pytest.approx(pf, abs=1e-9)


def test_an_hourly_schedule_settles_the_same_at_quarter_hours():
    rng = np.random.default_rng(4)
    q = np.round(rng.normal(60, 40, 96), 2)
    h = q.reshape(-1, 4).mean(axis=1)
    plan = h + rng.normal(0, 20, 24)
    r = battery.solve_day(plan, 1.0, 2.0, forecast.SPEC)
    hourly = forecast.settle(r.charge, r.discharge, 1.0, h)
    quarters = forecast.settle(*r.quarter_hourly(), 0.25, q)
    assert hourly["discharged_mwh_per_mw"] > 0
    for k, v in hourly.items():
        assert quarters[k] == pytest.approx(v, abs=1e-9)


def test_a_forecast_above_perfect_foresight_stops_the_run():
    daily = pd.DataFrame(
        [
            {
                "delivery_date": date(2026, 1, 1),
                "duration_h": 1,
                "resolution": "15min",
                "model": m,
                "net_eur_per_mw": v,
            }
            for m, v in (("perfect", 100.0), ("naive", 90.0), ("expert", 100.00001))
        ]
    )
    with pytest.raises(BuildError, match="beats perfect foresight"):
        forecast.check_perfect_bound(daily)
    daily.loc[daily["model"] == "expert", "net_eur_per_mw"] = 100.0000005
    forecast.check_perfect_bound(daily)


# ---------------------------------------------------------------- 7: metrics and DM test


def test_error_metrics_match_hand_computed_values():
    actual = [10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0, 0.0]
    fc = pd.DataFrame(
        {
            "delivery_date": date(2026, 1, 1),
            "period": range(1, 9),
            "price_es": actual,
            "naive": [12.0, 18.0, 30.0, 44.0, 1.0, 1.0, 1.0, 1.0],
            "expert": actual,
            "expert_weather": [a + 5 for a in actual],
        }
    )
    e = forecast.errors(fc).set_index("model")
    # naive errors 2, -2, 0, 4, 1, 1, 1, 1; hourly means 26 and 1 against 25 and 0; within-hour
    # deviations -14, -8, 4, 18 against -15, -5, 5, 15 in the first hour, 0 in the second.
    assert e.loc["naive", "mae_15min"] == pytest.approx(1.5)
    assert e.loc["naive", "rmse_15min"] == pytest.approx(np.sqrt(3.5))
    assert e.loc["naive", "mae_hourly"] == pytest.approx(1.0)
    assert e.loc["naive", "rmse_hourly"] == pytest.approx(1.0)
    assert e.loc["naive", "mae_within_hour_deviation"] == pytest.approx(1.0)
    assert e.loc["expert", ["mae_15min", "rmse_hourly"]].tolist() == [0, 0]
    assert e.loc["expert_weather", "mae_hourly"] == pytest.approx(5.0)
    assert e.loc["expert_weather", "mae_within_hour_deviation"] == pytest.approx(0.0)
    assert e.loc["naive", ["days", "periods", "hours"]].tolist() == [1, 8, 2]


def test_diebold_mariano_matches_hand_computed_values():
    # d = -1, 0, 1, 2: mean 0.5, gamma0 1.25, gamma1 0.3125; lag 1 weight 1/2 gives 1.5625,
    # so the statistic is 0.5 / sqrt(1.5625 / 4) = 0.8.
    r = forecast.dm_test([1, 2, 3, 4], [2, 2, 2, 2], lag=1)
    assert r["mean_loss_diff"] == pytest.approx(0.5)
    assert r["dm_stat"] == pytest.approx(0.8)
    assert r["p_value"] == pytest.approx(0.4237107971, abs=1e-9)
    assert r["n_days"] == 4
    assert forecast.newey_west_variance([-1, 0, 1, 2], 1) == pytest.approx(1.5625)


def test_newey_west_lag_0_is_the_plain_variance():
    d = np.random.default_rng(5).normal(size=50)
    assert forecast.newey_west_variance(d, 0) == pytest.approx(np.var(d), rel=1e-12)


# ---------------------------------------------------------------- 8: inputs and writing


# The three evaluation days of write_data_dir are Friday 5, Saturday 6 and Sunday 7 December.
START = date(2025, 10, 3)


def write_data_dir(root: Path, ndays: int = forecast.FIRST_EVAL + 3) -> None:
    pw = frame(START, ndays, seed=7)
    q = pw[["delivery_start_utc", "delivery_date", "period", "price_es", "price_pt"]].assign(
        source_version=1
    )
    ds = root / "dataset"
    ds.mkdir(parents=True)
    q.to_parquet(ds / "prices_15min.parquet", index=False)
    hourly_benchmark(q).to_parquet(ds / "prices_hourly.parquet", index=False)
    pw.to_parquet(ds / "prices_weather_15min.parquet", index=False)


def files_in(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def test_run_writes_every_output(tmp_path):
    write_data_dir(tmp_path)
    assert cli.main(["forecast", "--data-dir", str(tmp_path)]) == 0
    out = tmp_path / "results" / "forecast"
    assert set(files_in(out)) == OUTPUTS
    meta = json.loads((out / "forecast_meta.json").read_text(encoding="utf-8"))
    assert meta["evaluation"]["start"] == str(START + timedelta(days=63))
    assert meta["evaluation"]["days"] == 3
    fc = pd.read_parquet(out / "forecasts_15min.parquet")
    assert len(fc) == 3 * 96 and set(MODELS) <= set(fc.columns)
    # Each day's naive column is the input price of D-1 on Friday and of D-7 at the weekend.
    pw = pd.read_parquet(tmp_path / "dataset" / "prices_weather_15min.parquet")
    for day, lag in ((date(2025, 12, 5), 1), (date(2025, 12, 6), 7), (date(2025, 12, 7), 7)):
        got = fc.loc[fc["delivery_date"] == day, "naive"].to_numpy()
        assert np.array_equal(got, day_prices(pw, day - timedelta(days=lag))), day
    s = pd.read_csv(out / "forecast_value_summary.csv")
    assert len(s) == len(battery.DURATIONS_H) * 4
    perfect = s[s["model"] == "perfect"]
    assert np.allclose(perfect[["net_capture_15min_pct", "net_capture_hourly_pct"]], 100)
    assert np.allclose(perfect["net_uplift_capture_pct"], 100)
    assert (s["net_capture_15min_pct"] <= 100 + 1e-6).all()
    daily = pd.read_parquet(out / "forecast_value_daily.parquet")
    assert len(daily) == 3 * len(battery.DURATIONS_H) * 4 * 2
    # The uplift columns, recomputed from the daily values.
    net = daily.pivot_table(
        index=["duration_h", "model"], columns="resolution", values="net_eur_per_mw", aggfunc="sum"
    )
    up = net["15min"] - net["hourly"]
    for hours in battery.DURATIONS_H:
        row = s[(s["duration_h"] == hours) & (s["model"] == "expert")].iloc[0]
        u = up[(hours, "expert")]
        assert row["net_uplift_keur_per_mw"] == pytest.approx(u / 1000)
        assert row["net_uplift_eur_per_mw_day"] == pytest.approx(u / 3)
        assert row["net_uplift_capture_pct"] == pytest.approx(100 * u / up[(hours, "perfect")])


@pytest.mark.parametrize(
    "missing",
    ["prices_15min.parquet", "prices_hourly.parquet", "prices_weather_15min.parquet"],
)
def test_a_missing_input_stops_before_anything_is_written(tmp_path, capsys, missing):
    write_data_dir(tmp_path)
    (tmp_path / "dataset" / missing).unlink()
    assert cli.main(["forecast", "--data-dir", str(tmp_path)]) == 1
    assert missing in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_a_missing_price_column_in_the_weather_file_stops_the_run(tmp_path, capsys):
    write_data_dir(tmp_path)
    path = tmp_path / "dataset" / "prices_weather_15min.parquet"
    pd.read_parquet(path).drop(columns="delivery_date").to_parquet(path, index=False)
    assert cli.main(["forecast", "--data-dir", str(tmp_path)]) == 1
    assert "has no column delivery_date" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_too_few_days_for_one_evaluation_day_stops_the_run(tmp_path, capsys):
    write_data_dir(tmp_path, ndays=forecast.FIRST_EVAL)
    assert cli.main(["forecast", "--data-dir", str(tmp_path)]) == 1
    assert "at least 64" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_weather_rows_that_differ_from_the_prices_stop_the_run(tmp_path, capsys):
    write_data_dir(tmp_path)
    path = tmp_path / "dataset" / "prices_weather_15min.parquet"
    pw = pd.read_parquet(path)
    pw.loc[5, "price_es"] += 1.0
    pw.to_parquet(path, index=False)
    assert cli.main(["forecast", "--data-dir", str(tmp_path)]) == 1
    assert "does not hold the rows" in capsys.readouterr().err


def test_a_failure_while_writing_leaves_the_previous_outputs(tmp_path, monkeypatch):
    write_data_dir(tmp_path)
    out = tmp_path / "results" / "forecast"
    out.mkdir(parents=True)
    for name in OUTPUTS:
        (out / name).write_text(f"previous {name}")
    previous = files_in(out)
    to_csv = pd.DataFrame.to_csv

    def fail_at_summary(self, path=None, *args, **kwargs):
        # By now the forecasts, errors, DM tests and daily values are in their .part files.
        if path is not None and Path(path).name.startswith("forecast_value_summary"):
            raise OSError("disk full")
        return to_csv(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_at_summary)
    with pytest.raises(OSError, match="disk full"):
        cli.main(["forecast", "--data-dir", str(tmp_path)])
    assert files_in(out) == previous  # byte for byte, and no .part or .prev left

    monkeypatch.undo()
    assert cli.main(["forecast", "--data-dir", str(tmp_path)]) == 0
    new = files_in(out)
    assert set(new) == set(previous)
    assert all(new[name] != previous[name] for name in previous)
