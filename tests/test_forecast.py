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
    "forecast_window_sensitivity.csv",
    "forecast_uncertainty.csv",
    "forecast_regret_drivers.csv",
    "forecast_meta.json",
    "fig_example_day.png",
    "fig_example_day.pdf",
}


def frame(start: date, ndays: int, seed: int = 0) -> pd.DataFrame:
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
    covers_both_dst_changes_with_more_than_63_days_before_each = frame(date(2025, 8, 15), 240)
    return covers_both_dst_changes_with_more_than_63_days_before_each


def forecast_for(pw: pd.DataFrame, day: date) -> dict[str, np.ndarray]:
    dates, p, w, z, slots = forecast.slot_arrays(pw)
    t = dates.index(day)
    fc = forecast.forecast_slots(dates, p, w, z, [t])
    return {m: fc[m][0][slots[t]] for m in MODELS}


def day_prices(pw: pd.DataFrame, day: date) -> np.ndarray:
    return pw.loc[pw["delivery_date"] == day, "price_es"].to_numpy()


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

    d1 = changed.copy()
    d1.loc[d1["delivery_date"] == day - timedelta(days=1), "price_es"] += 10.0
    assert not np.allclose(forecast_for(d1, day)["expert"], base["expert"]), (
        "the information that is allowed does move the forecast"
    )
    wx = long_frame.copy()
    wx.loc[wx["delivery_date"] == day, WX_COLS] += 3.0
    assert not np.allclose(forecast_for(wx, day)["expert_weather"], base["expert_weather"]), (
        "the information that is allowed does move the forecast"
    )


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


@pytest.mark.parametrize("window", forecast.SENSITIVITY_WINDOWS)
def test_each_sensitivity_window_reaches_back_exactly_its_length_plus_7_days(long_frame, window):
    dates, p, w, z, _ = forecast.slot_arrays(long_frame)
    t = dates.index(date(2026, 1, 14))
    reach = window + 7

    def expert(prices):
        fc = forecast.forecast_slots(dates, prices, w, z, [t], window)
        return fc["expert"][0], fc["expert_weather"][0]

    base = expert(p)
    older = p.copy()
    older[: t - reach] += 100.0
    assert all(np.array_equal(a, b) for a, b in zip(expert(older), base, strict=True))
    edge = p.copy()
    edge[t - reach] += 100.0
    assert not any(np.allclose(a, b) for a, b in zip(expert(edge), base, strict=True))


@pytest.mark.parametrize(
    "day, lag",
    [
        pytest.param(date(2026, 1, 12), 7, id="monday"),
        pytest.param(date(2026, 1, 17), 7, id="saturday"),
        pytest.param(date(2026, 1, 18), 7, id="sunday"),
        pytest.param(date(2026, 1, 14), 1, id="wednesday"),
        pytest.param(date(2025, 12, 8), 1, id="monday-holiday-copies-last-sunday-7-dec"),
        pytest.param(date(2025, 12, 6), 6, id="saturday-holiday-copies-last-sunday-30-nov"),
        pytest.param(date(2025, 12, 25), 4, id="thursday-holiday-copies-last-sunday-21-dec"),
        pytest.param(date(2026, 1, 6), 2, id="tuesday-holiday-copies-last-sunday-4-jan"),
        pytest.param(date(2025, 12, 9), 1, id="tuesday-after-holiday-follows-ordinary-rule"),
        pytest.param(date(2025, 12, 26), 1, id="friday-after-holiday-follows-ordinary-rule"),
        pytest.param(date(2026, 1, 7), 1, id="wednesday-after-holiday-follows-ordinary-rule"),
    ],
)
def test_naive_rule(long_frame, day, lag):
    got = forecast_for(long_frame, day)["naive"]
    assert np.array_equal(got, day_prices(long_frame, day - timedelta(days=lag)))


@pytest.mark.parametrize(
    "day, dummies",
    [
        pytest.param(date(2025, 12, 8), (0, 0, 1), id="monday-holiday"),
        pytest.param(date(2025, 12, 25), (0, 0, 1), id="thursday-holiday"),
        pytest.param(date(2025, 12, 15), (1, 0, 0), id="ordinary-monday"),
        pytest.param(date(2025, 12, 18), (0, 0, 0), id="ordinary-thursday"),
    ],
)
def test_a_holiday_has_the_sunday_dummies(long_frame, day, dummies):
    dates, p, w, _, _ = forecast.slot_arrays(long_frame)
    cols = [forecast.EXPERT_FEATURES.index(k) for k in ("Monday", "Saturday", "Sunday")]
    for f in (forecast.features(dates, p), forecast.features(dates, p, w)):
        row = f[dates.index(day)][:, cols]
        assert (row == dummies).all()


def test_the_holiday_list():
    hol = forecast.HOLIDAYS
    assert all(isinstance(d, date) and 2025 <= d.year <= 2026 for d in hol)
    assert len(set(hol)) == len(hol)
    assert list(hol) == sorted(hol)
    for d in ("2025-01-01", "2026-01-01", "2025-05-01", "2026-05-01", "2025-10-12", "2026-10-12"):
        assert date.fromisoformat(d) in hol, d
    assert date(2026, 4, 2) not in hol, (
        "Holy Thursday is not held in Cataluña or the Comunitat Valenciana"
    )


def per_hour(a: np.ndarray) -> np.ndarray:
    return np.repeat(a, 4, axis=0)


def noise_free(with_weather: bool):
    rng = np.random.default_rng(1)
    n = 70
    dates = [date(2025, 11, 3) + timedelta(days=i) for i in range(n)]
    p = np.zeros((n, 96))
    p[:7] = rng.uniform(20, 120, (7, 96))
    one_coefficient_per_hour_for_its_four_quarters = per_hour(
        np.column_stack(
            [
                rng.uniform(-5, 5, 24),
                rng.uniform(0.3, 0.5, 24),
                rng.uniform(-0.2, 0.2, 24),
                rng.uniform(0.3, 0.45, 24),
                rng.uniform(-0.1, 0.1, (24, 3)),
                rng.uniform(-10, 10, (24, 3)),
                rng.uniform(-8, 8, (24, 3)),
            ]
        )
    )
    w = rng.normal(size=(n, 96, 2)) * [5, 3] + [15, 6]
    gamma = per_hour(rng.uniform(-1, 1, (24, 2)))
    z = rng.normal(size=(n, 96, 2)) * [200, 50] + [200, 0]
    one_radiation_coefficient_for_all_hours = np.array([-0.05, 0.02])
    quarter = np.arange(96) % 4
    for j in range(7, n):
        prev = p[j - 1]
        weekday_a_holiday_has_the_sunday_dummies = (
            6 if dates[j] in forecast.HOLIDAYS else dates[j].weekday()
        )
        x = np.column_stack(
            [np.ones(96), prev, p[j - 2], p[j - 7]]
            + [np.full(96, v) for v in (prev.min(), prev.max(), prev[-1])]
            + [np.full(96, float(weekday_a_holiday_has_the_sunday_dummies == k)) for k in (0, 5, 6)]
            + [(quarter == k).astype(float) for k in (1, 2, 3)]
        )
        p[j] = (x * one_coefficient_per_hour_for_its_four_quarters).sum(axis=1)
        if with_weather:
            p[j] += (w[j] * gamma).sum(axis=1) + z[j] @ one_radiation_coefficient_for_all_hours
    return dates, p, w, z


@pytest.mark.parametrize("model", ["expert", "expert_weather"])
def test_expert_recovers_a_noise_free_equation(model):
    dates, p, w, z = noise_free(model == "expert_weather")
    targets = [forecast.FIRST_EVAL, len(dates) - 1]
    got = forecast.forecast_slots(dates, p, w, z, targets)[model]
    np.testing.assert_allclose(got, p[targets], rtol=1e-9, atol=1e-8)


def test_an_hour_dark_through_the_window_does_not_extrapolate_when_the_clock_brings_sun():
    rng = np.random.default_rng(4)
    n = forecast.FIRST_EVAL + 1
    dates = [date(2026, 1, 5) + timedelta(days=i) for i in range(n)]
    hour_19 = slice(76, 80)
    hourly_sun = np.clip(np.sin(np.pi * (np.arange(96) - 32) / 40), 0, None)
    rad = 600 * hourly_sun * rng.uniform(0.3, 1, (n, 1))
    rad[:, hour_19] = rng.uniform(0, 0.5, (n, 4))
    spring_forward_puts_19_00_an_hour_earlier_in_the_sun = 147.0
    rad[-1, hour_19] = spring_forward_puts_19_00_an_hour_earlier_in_the_sun
    slope = -0.1
    base = per_hour(rng.uniform(40, 80, 24))
    p = base + slope * rad + rng.normal(0, 5, (n, 96))
    w = rng.normal(size=(n, 96, 2)) * [5, 3] + [15, 6]
    z = np.stack([rad, np.zeros_like(rad)], axis=2)
    got = forecast.forecast_slots(dates, p, w, z, [n - 1])["expert_weather"][0]
    expected = base[hour_19] + slope * spring_forward_puts_19_00_an_hour_earlier_in_the_sun
    np.testing.assert_allclose(
        got[hour_19],
        expected,
        atol=15,
        err_msg="a radiation slope fitted on one hour's near-zero values would extrapolate to"
        " hundreds of EUR/MWh; the slope shared by all hours gives about base - 0.1 * 147",
    )


def test_the_radiation_change_is_centred_inside_the_day_and_one_sided_at_its_edges():
    hourly = np.array([0.0, 100.0, 300.0, 300.0, 250.0])
    got = forecast.radiation_ramp(np.repeat(hourly, 4))
    np.testing.assert_allclose(got, np.repeat([100.0, 150.0, 100.0, -25.0, -50.0], 4))


def test_the_radiation_change_is_tilted_by_the_quarters_position_in_the_hour(long_frame):
    day = date(2026, 1, 14)
    pw = long_frame.copy()
    rows = pw["delivery_date"] == day
    ghi = [c for c in WX_COLS if c.startswith("ghi_")]
    hour = (pw.loc[rows, "period"].to_numpy() - 1) // 4
    pw.loc[rows, ghi] = np.repeat((10.0 * hour**2)[:, None], len(ghi), axis=1)
    dates, _, _, z, _ = forecast.slot_arrays(pw)
    rad, tilt = z[dates.index(day)].T
    np.testing.assert_allclose(rad, np.repeat(10.0 * np.arange(24) ** 2, 4))
    change = np.gradient(10.0 * np.arange(24) ** 2)
    np.testing.assert_allclose(tilt, np.repeat(change, 4) * np.tile([-1.5, -0.5, 0.5, 1.5], 24))


def test_the_example_day_has_the_median_perfect_foresight_1_h_uplift():
    uplifts = {date(2026, 1, d): u for d, u in ((1, 9.0), (2, 1.0), (3, 5.0), (4, 3.0), (5, 7.0))}
    rows = [
        {
            "delivery_date": day,
            "duration_h": hours,
            "model": model,
            "resolution": res,
            "net_eur_per_mw": 100.0 + (u if res == "15min" else 0.0) * (model == "perfect"),
        }
        for day, u in uplifts.items()
        for hours in battery.DURATIONS_H
        for model in ("perfect", *MODELS)
        for res in ("15min", "hourly")
    ]
    assert forecast.example_day(pd.DataFrame(rows)) == date(2026, 1, 3), (
        "uplifts 9, 1, 5, 3, 7: the median is 5, on 3 January"
    )
    even = pd.DataFrame([r for r in rows if r["delivery_date"] != date(2026, 1, 5)])
    assert forecast.example_day(even) == date(2026, 1, 4), (
        "uplifts 9, 1, 5, 3: the lower middle day is the one at 3"
    )


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
            np.testing.assert_array_equal(
                slots[8:12],
                slots[4:8],
                err_msg="02:00-02:59 does not exist: it takes 01:00-01:59",
            )
            np.testing.assert_array_equal(back, values)
        else:
            np.testing.assert_allclose(
                slots[8:12],
                (values[8:12] + values[12:16]) / 2,
                err_msg="02:00-02:59 happens twice (periods 9-12 and 13-16): the slot is their "
                "mean",
            )
            both_get_it_back = "both occurrences of 02:00-02:59 get the slot back"
            np.testing.assert_allclose(back[8:12], slots[8:12], err_msg=both_get_it_back)
            np.testing.assert_allclose(back[12:16], slots[8:12], err_msg=both_get_it_back)
            keep = np.r_[0:8, 16:100]
            np.testing.assert_array_equal(back[keep], values[keep])


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
    naive_errors = "naive errors 2, -2, 0, 4, 1, 1, 1, 1"
    hourly_means = "hourly means 26 and 1 against 25 and 0"
    assert e.loc["naive", "mae_15min"] == pytest.approx(1.5), naive_errors
    assert e.loc["naive", "rmse_15min"] == pytest.approx(np.sqrt(3.5)), naive_errors
    assert e.loc["naive", "mae_hourly"] == pytest.approx(1.0), hourly_means
    assert e.loc["naive", "rmse_hourly"] == pytest.approx(1.0), hourly_means
    assert e.loc["naive", "mae_within_hour_deviation"] == pytest.approx(1.0), (
        "within-hour deviations -14, -8, 4, 18 against -15, -5, 5, 15 in the first hour, "
        "0 in the second"
    )
    assert e.loc["expert", ["mae_15min", "rmse_hourly"]].tolist() == [0, 0]
    assert e.loc["expert_weather", "mae_hourly"] == pytest.approx(5.0)
    assert e.loc["expert_weather", "mae_within_hour_deviation"] == pytest.approx(0.0)
    assert e.loc["naive", ["days", "periods", "hours"]].tolist() == [1, 8, 2]


def test_diebold_mariano_matches_hand_computed_values():
    r = forecast.dm_test([1, 2, 3, 4], [2, 2, 2, 2], lag=1)
    assert r["mean_loss_diff"] == pytest.approx(0.5)
    assert r["dm_stat"] == pytest.approx(0.8), (
        "d = -1, 0, 1, 2: mean 0.5, gamma0 1.25, gamma1 0.3125; lag 1 weight 1/2 gives variance "
        "1.5625, so the statistic is 0.5 / sqrt(1.5625 / 4) = 0.8"
    )
    assert r["p_value"] == pytest.approx(0.4237107971, abs=1e-9)
    assert r["n_days"] == 4
    assert forecast.newey_west_variance([-1, 0, 1, 2], 1) == pytest.approx(1.5625)


def test_newey_west_lag_0_is_the_plain_variance():
    d = np.random.default_rng(5).normal(size=50)
    assert forecast.newey_west_variance(d, 0) == pytest.approx(np.var(d), rel=1e-12)


def test_dm_tests_report_every_lag_and_the_main_lag_is_the_main_test():
    rng = np.random.default_rng(6)
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(60)]
    fc = pd.DataFrame({"delivery_date": np.repeat(days, 4)})
    fc["price_es"] = rng.normal(60, 30, len(fc))
    for m in MODELS:
        fc[m] = fc["price_es"] + rng.normal(0, 10, len(fc))
    dm = forecast.dm_tests(fc)
    assert forecast.DM_LAG == 7
    for a, b in (("expert", "naive"), ("expert_weather", "expert")):
        rows = dm[(dm["model_a"] == a) & (dm["model_b"] == b)].set_index("newey_west_lag")
        assert list(rows.index) == [7, 14, 28]
        main = forecast.dm_test(forecast.daily_mae(fc, a), forecast.daily_mae(fc, b), lag=7)
        assert main.pop("newey_west_lag") == 7
        for k, v in main.items():
            assert rows.loc[7, k] == pytest.approx(v, rel=1e-12), k
        assert rows.loc[14, "dm_stat"] != pytest.approx(rows.loc[7, "dm_stat"])


def hourly_prices(pw: pd.DataFrame) -> dict:
    cols = ["delivery_start_utc", "delivery_date", "period", "price_es", "price_pt"]
    h = hourly_benchmark(pw[cols + ["source_version"]])
    return {d: g["price_es"].to_numpy() for d, g in h.groupby("delivery_date")}


def test_the_56_day_sensitivity_row_is_the_main_model_on_the_common_days():
    first = forecast.first_eval(max(forecast.SENSITIVITY_WINDOWS))
    pw = frame(date(2025, 9, 1), first + 2, seed=8)
    hourly = hourly_prices(pw)
    sens, days = forecast.window_sensitivity(pw, hourly)
    assert days == sorted(pw["delivery_date"].unique())[first:]
    assert len(sens) == len(battery.DURATIONS_H) * 2 * len(forecast.SENSITIVITY_WINDOWS)

    main_56_day_forecasts = forecast.forecast_periods(pw)
    fc = main_56_day_forecasts[main_56_day_forecasts["delivery_date"] >= days[0]]
    fc = fc.reset_index(drop=True)
    rows = []
    for day, g in fc.groupby("delivery_date", sort=True):
        plans = {m: g[m].to_numpy() for m in MODELS}
        rows += forecast.day_values(day, g["price_es"].to_numpy(), hourly[day], plans)
    main = forecast.summarise(pd.DataFrame(rows)).set_index(["duration_h", "model"])
    s = sens.set_index(["duration_h", "model", "window_days"])
    cols = [
        "days",
        "net_capture_15min_pct",
        "net_capture_hourly_pct",
        "net_uplift_eur_per_mw_day",
        "net_uplift_capture_pct",
    ]
    for hours in battery.DURATIONS_H:
        for m in forecast.SENSITIVITY_MODELS:
            mae = (fc[m] - fc["price_es"]).abs().mean()
            assert s.loc[(hours, m, 56), "mae_15min"] == pytest.approx(mae, rel=1e-12)
            for c in cols:
                assert s.loc[(hours, m, 56), c] == pytest.approx(main.loc[(hours, m), c]), (
                    "the main 56-day forecasts, cut to the common days and valued as the main "
                    "run does",
                    c,
                )
            for win in (28, 112):
                assert s.loc[(hours, m, win), "mae_15min"] != pytest.approx(mae), (
                    "the other windows are really other fits"
                )


START = date(2025, 8, 6)
NDAYS = forecast.first_eval(max(forecast.SENSITIVITY_WINDOWS)) + 5


def write_data_dir(root: Path, ndays: int = NDAYS, start: date = START) -> None:
    pw = frame(start, ndays, seed=7)
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
    days = NDAYS - forecast.FIRST_EVAL
    assert (meta["evaluation"]["start"], meta["evaluation"]["days"]) == (
        str(START + timedelta(days=63)),
        days,
    ), "the main evaluation runs from 8 October to Sunday 7 December 2025 (61 days)"
    assert meta["holidays"]["dates"] == [str(d) for d in forecast.HOLIDAYS]
    assert meta["dm_test"]["lag_days"] == 7
    assert meta["dm_test"]["sensitivity_lags_days"] == [7, 14, 28]
    sens_meta = meta["window_sensitivity"]
    assert sens_meta["windows_days"] == [28, 56, 112]
    assert (sens_meta["start"], sens_meta["days"]) == ("2025-12-03", 5), (
        "the window sensitivity runs from Wednesday 3 December (5 days)"
    )
    fc = pd.read_parquet(out / "forecasts_15min.parquet")
    assert fc["delivery_date"].nunique() == days and set(MODELS) <= set(fc.columns)
    daily = pd.read_parquet(out / "forecast_value_daily.parquet")
    assert meta["example_day"]["date"] == str(forecast.example_day(daily))
    pw = pd.read_parquet(tmp_path / "dataset" / "prices_weather_15min.parquet")
    for day, lag in ((date(2025, 12, 5), 1), (date(2025, 12, 6), 6), (date(2025, 12, 7), 7)):
        got = fc.loc[fc["delivery_date"] == day, "naive"].to_numpy()
        assert np.array_equal(got, day_prices(pw, day - timedelta(days=lag))), (
            f"{day}: D-1 on Friday, the last Sunday on the Saturday holiday 6 December, "
            "D-7 on Sunday"
        )
    s = pd.read_csv(out / "forecast_value_summary.csv")
    assert len(s) == len(battery.DURATIONS_H) * 4
    perfect = s[s["model"] == "perfect"]
    assert np.allclose(perfect[["net_capture_15min_pct", "net_capture_hourly_pct"]], 100)
    assert np.allclose(perfect["net_uplift_capture_pct"], 100)
    assert (s["net_capture_15min_pct"] <= 100 + 1e-6).all()
    sens = pd.read_csv(out / "forecast_window_sensitivity.csv")
    assert len(sens) == len(battery.DURATIONS_H) * 2 * 3 and (sens["days"] == 5).all()
    assert (sens["net_capture_15min_pct"] <= 100 + 1e-6).all()
    dm = pd.read_csv(out / "forecast_dm_tests.csv")
    assert sorted(dm["newey_west_lag"]) == [7, 7, 14, 14, 28, 28]
    daily = pd.read_parquet(out / "forecast_value_daily.parquet")
    assert len(daily) == days * len(battery.DURATIONS_H) * 4 * 2
    net = daily.pivot_table(
        index=["duration_h", "model"], columns="resolution", values="net_eur_per_mw", aggfunc="sum"
    )
    up = net["15min"] - net["hourly"]
    for hours in battery.DURATIONS_H:
        row = s[(s["duration_h"] == hours) & (s["model"] == "expert")].iloc[0]
        u = up[(hours, "expert")]
        recomputed = "the uplift columns, recomputed from the daily values"
        assert row["net_uplift_keur_per_mw"] == pytest.approx(u / 1000), recomputed
        assert row["net_uplift_eur_per_mw_day"] == pytest.approx(u / days), recomputed
        assert row["net_uplift_capture_pct"] == pytest.approx(100 * u / up[(hours, "perfect")]), (
            recomputed
        )
    unc = pd.read_csv(out / "forecast_uncertainty.csv")
    assert (unc["days"] == days).all()
    assert len(unc) == len(battery.DURATIONS_H) * (4 * 4 + 2)
    si = s.set_index(["duration_h", "model"])
    for _, r in unc.iterrows():
        hours, stat = r["duration_h"], r["statistic"]
        if " minus " in r["model"]:
            a, b = r["model"].split(" minus ")
            want = si.loc[(hours, a), stat] - si.loc[(hours, b), stat]
        else:
            want = si.loc[(hours, r["model"]), stat]
        assert r["estimate"] == want, (
            "the uncertainty estimates are the summary values, and the differences their "
            "differences",
            tuple(r),
        )
    assert meta["uncertainty"]["seed"] == battery.BOOT_SEED
    assert "tau-b" in meta["kendall_tau"]
    err = pd.read_csv(out / "forecast_errors.csv")
    assert err.columns[-1] == "mean_daily_kendall_tau"
    assert err["mean_daily_kendall_tau"].between(-1, 1).all()
    drivers = pd.read_csv(out / "forecast_regret_drivers.csv")
    assert len(drivers) == len(battery.DURATIONS_H) * 4
    pooled = drivers[drivers["model"] == "pooled"]
    assert (pooled["observations"] == 3 * days).all() and (drivers["days"] == days).all()


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
    write_data_dir(tmp_path, ndays=forecast.first_eval(max(forecast.SENSITIVITY_WINDOWS)))
    assert cli.main(["forecast", "--data-dir", str(tmp_path)]) == 1
    assert "at least 120" in capsys.readouterr().err, (
        "the longest sensitivity window (112 days, with lags back 7) needs 120 days"
    )
    assert not (tmp_path / "results").exists()


def test_days_outside_the_holiday_list_stop_the_run(tmp_path, capsys):
    write_data_dir(tmp_path, start=date(2024, 9, 1))
    assert cli.main(["forecast", "--data-dir", str(tmp_path)]) == 1
    assert "holiday list covers 2025-2026 only" in capsys.readouterr().err
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
        if path is not None and Path(path).name.startswith("forecast_value_summary"):
            raise OSError("disk full")
        return to_csv(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_at_summary)
    with pytest.raises(OSError, match="disk full"):
        cli.main(["forecast", "--data-dir", str(tmp_path)])
    assert files_in(out) == previous, (
        "by the summary the forecasts, errors, DM tests and daily values are in their .part "
        "files; byte for byte, and no .part or .prev left"
    )

    monkeypatch.undo()
    assert cli.main(["forecast", "--data-dir", str(tmp_path)]) == 0
    new = files_in(out)
    assert set(new) == set(previous)
    assert all(new[name] != previous[name] for name in previous)


def value_daily(values: dict[str, tuple]) -> pd.DataFrame:
    n = len(next(iter(values.values()))[0])
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(n)]
    rows = [
        {
            "delivery_date": d,
            "duration_h": hours,
            "model": m,
            "resolution": res,
            "gross_eur_per_mw": v[i],
            "wear_cost_eur_per_mw": 0.0,
            "net_eur_per_mw": v[i],
        }
        for hours in battery.DURATIONS_H
        for m, (v15, vh) in values.items()
        for res, v in (("15min", v15), ("hourly", vh))
        for i, d in enumerate(days)
    ]
    return pd.DataFrame(rows)


def random_values(rng, n):
    vh = rng.uniform(50, 150, n)
    return vh + rng.uniform(0, 40, n), vh


def test_resampled_statistics_on_the_original_days_are_the_summary():
    rng = np.random.default_rng(12)
    daily = value_daily({m: random_values(rng, 40) for m in ("perfect", *MODELS)})
    s = forecast.summarise(daily).set_index(["duration_h", "model"])
    stats = forecast.resampled_statistics(daily, np.arange(40)[None])
    assert len(stats) == len(battery.DURATIONS_H) * (4 * 4 + 2)
    for (hours, model, stat), v in stats.items():
        if " minus " in model:
            a, b = model.split(" minus ")
            want = s.loc[(hours, a), stat] - s.loc[(hours, b), stat]
        else:
            want = s.loc[(hours, model), stat]
        assert v[0] == pytest.approx(want, rel=1e-12), (hours, model, stat)


def test_the_interval_collapses_when_every_day_is_the_same():
    daily = value_daily(
        {
            "perfect": (np.full(30, 200.0), np.full(30, 150.0)),
            "naive": (np.full(30, 120.0), np.full(30, 100.0)),
            "expert": (np.full(30, 160.0), np.full(30, 130.0)),
            "expert_weather": (np.full(30, 170.0), np.full(30, 135.0)),
        }
    )
    unc = forecast.uncertainty(daily, forecast.summarise(daily))
    assert np.allclose(unc["ci_low"], unc["estimate"], rtol=1e-12, atol=1e-12)
    assert np.allclose(unc["ci_high"], unc["estimate"], rtol=1e-12, atol=1e-12)
    u = unc.set_index(["duration_h", "model", "statistic"])["estimate"]
    assert u[(1, "naive", "net_capture_15min_pct")] == pytest.approx(60)
    assert u[(1, "expert minus naive", "net_uplift_eur_per_mw_day")] == pytest.approx(10)


def test_every_model_is_resampled_on_the_same_days():
    rng = np.random.default_rng(13)
    n = 60
    v15, vh = random_values(rng, n)
    daily = value_daily(
        {
            "perfect": (2 * v15, 2 * vh),
            "naive": (v15, vh),
            "expert": (v15 + 3, vh),
            "expert_weather": random_values(rng, n),
        }
    )
    unc = forecast.uncertainty(daily, forecast.summarise(daily)).set_index(
        ["duration_h", "model", "statistic"]
    )
    diff = unc.loc[(2, "expert minus naive", "net_uplift_eur_per_mw_day")]
    assert diff[["estimate", "ci_low", "ci_high"]].tolist() == pytest.approx([3, 3, 3]), (
        "expert = naive + 3 every day, so the paired difference is 3 on every resample only if "
        "both are resampled on the same days"
    )
    own = unc.loc[(2, "naive", "net_uplift_eur_per_mw_day")]
    assert own["ci_high"] - own["ci_low"] > 2, (
        "each model's own uplift varies a lot from day to day"
    )
    for stat in ("net_capture_15min_pct", "net_capture_hourly_pct", "net_uplift_capture_pct"):
        row = unc.loc[(2, "naive", stat)]
        assert row[["estimate", "ci_low", "ci_high"]].tolist() == pytest.approx([50, 50, 50]), (
            f"{stat}: perfect = 2 x naive, so naive's share is 50 on every resample only if the "
            "benchmark is resampled on the same days"
        )


def test_kendall_tau_matches_hand_computed_values():
    actual = [1.0, 2.0, 3.0, 4.0] * 2
    fc = pd.DataFrame(
        {
            "delivery_date": [date(2026, 1, 1)] * 4 + [date(2026, 1, 2)] * 4,
            "period": [1, 2, 3, 4] * 2,
            "price_es": actual,
            "naive": [1.0, 3.0, 2.0, 4.0, 1.0, 1.0, 2.0, 2.0],
            "expert": actual,
            "expert_weather": [4.0, 3.0, 2.0, 1.0] * 2,
        }
    )
    tau = forecast.daily_tau(fc, "naive")
    assert tau.tolist() == pytest.approx([4 / 6, 4 / np.sqrt(24)]), (
        "day 1: 5 concordant pairs and 1 discordant give 4/6; day 2: 4 concordant and 2 tied in "
        "the forecast give tau-b 4 / sqrt(6 * 4)"
    )
    e = forecast.errors(fc).set_index("model")["mean_daily_kendall_tau"]
    assert e["naive"] == pytest.approx((4 / 6 + 4 / np.sqrt(24)) / 2)
    assert e[["expert", "expert_weather"]].tolist() == pytest.approx([1, -1])


def test_regret_drivers_match_hand_computed_values():
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(4)]
    actual = np.array([0.0, 10.0, 20.0, 30.0])
    daily_mae_1_5_2_20_and_tau_1_4_6_1_minus_1_for_every_model = [
        actual + 1,
        [10.0, 0.0, 20.0, 30.0],
        actual + 2,
        actual[::-1],
    ]
    forecasts = daily_mae_1_5_2_20_and_tau_1_4_6_1_minus_1_for_every_model
    fc = pd.DataFrame(
        {
            "delivery_date": np.repeat(days, 4),
            "period": [1, 2, 3, 4] * 4,
            "price_es": np.tile(actual, 4),
        }
    )
    for m in MODELS:
        fc[m] = np.concatenate(forecasts)
    perfect = np.full(4, 100.0)
    naive_regret = np.array([3.0, 1.0, 2.0, 4.0])
    expert_10_higher_expert_weather_20_higher = 10
    daily = value_daily(
        {"perfect": (perfect, perfect)}
        | {
            m: (perfect - naive_regret - expert_10_higher_expert_weather_20_higher * k, perfect)
            for k, m in enumerate(MODELS)
        }
    )
    d = forecast.regret_drivers(fc, daily).set_index(["duration_h", "model"])
    for hours in battery.DURATIONS_H:
        for m in MODELS:
            assert d.loc[(hours, m), "spearman_regret_mae"] == pytest.approx(0.2), (
                "ranks of regret 3, 1, 2, 4 and of daily MAE 1, 5, 2, 20 (ranks 1, 3, 2, 4): "
                "1 - 6 * 8 / (4 * 15) = 0.2"
            )
            assert d.loc[(hours, m), "spearman_regret_tau"] == pytest.approx(-1 / np.sqrt(10)), (
                "daily tau 1, 4/6, 1, -1 has ranks 3.5, 2, 3.5, 1: Pearson of the ranks "
                "-1.5 / sqrt(5 * 4.5)"
            )
            assert d.loc[(hours, m), ["days", "observations"]].tolist() == [4, 4]
        pooled = d.loc[(hours, "pooled")]
        assert pooled["spearman_regret_mae"] == pytest.approx(9 / np.sqrt(143 * 135)), (
            "pooled over 12 days: regret ranks 1..12 (sum of squared deviations 143); MAE ranks "
            "2, 8, 5, 11 for each model (deviations -4.5, 1.5, -1.5, 4.5; squares 135 over 12); "
            "cross sum 3 * (3 * -4.5 + 1 * 1.5 + 2 * -1.5 + 4 * 4.5) = 9"
        )
        assert pooled["spearman_regret_tau"] == pytest.approx(-13.5 / np.sqrt(143 * 121.5)), (
            "pooled tau ranks 9.5, 5, 9.5, 2 (deviations 3, -1.5, 3, -4.5; squares 121.5): "
            "cross sum 3 * -4.5 = -13.5"
        )
        assert pooled[["days", "observations"]].tolist() == [4, 12]
