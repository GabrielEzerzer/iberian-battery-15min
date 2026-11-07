import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ib15.battery import (
    DT_15MIN,
    DT_HOURLY,
    DURATIONS_H,
    ETA,
    POWER_MW,
    RTE,
    SOC_START,
    SPECS,
    SolveError,
    Spec,
    solve_day,
)
from ib15.cli import main

MAIN, NO_COST, CAP2 = SPECS
TOL = 1e-6


def random_day(rng, n=96):
    """Quarter-hour prices with a daily shape, noise and some negative and zero prices."""
    t = np.arange(n)
    shape = 60 + 40 * np.sin(2 * np.pi * t / n) - 50 * np.exp(-(((t - n / 2) / (n / 8)) ** 2))
    p = shape + rng.normal(0, 25, n)
    p[rng.random(n) < 0.05] = 0.0
    return np.round(p, 2)


def hourly(p):
    return p.reshape(-1, 4).mean(axis=1)


def simulate(c, d, dt, energy):
    """End-of-period energy for a schedule, starting at SOC_START of capacity."""
    return SOC_START * energy + np.cumsum(ETA * c * dt - d * dt / ETA)


def test_two_quarter_hours_hand_solved():
    # A 0.1 MWh battery holds 0.05 MWh at the start; power never binds (0.25 MWh a quarter).
    # Optimal: fill it at 0, then empty it back to 0.05 MWh at 100.
    energy = 0.1
    r = solve_day([0.0, 100.0], DT_15MIN, energy, MAIN)
    stored = energy - SOC_START * energy
    np.testing.assert_allclose(r.charge, [stored / (ETA * DT_15MIN), 0], atol=1e-9)
    np.testing.assert_allclose(r.discharge, [0, stored * ETA / DT_15MIN], atol=1e-9)
    np.testing.assert_allclose(r.energy, [energy, SOC_START * energy], atol=1e-9)
    assert r.discharged_mwh == pytest.approx(stored * ETA)
    assert r.gross == pytest.approx(100 * stored * ETA)
    assert r.degradation == pytest.approx(20 * stored * ETA)
    assert r.net == pytest.approx(80 * stored * ETA)
    assert r.simultaneous == 0


def test_flat_prices_give_no_dispatch_in_main():
    r = solve_day(np.full(96, 55.0), DT_15MIN, 2.0, MAIN)
    assert np.abs(r.charge).max() < TOL and np.abs(r.discharge).max() < TOL
    assert r.gross == pytest.approx(0, abs=TOL) and r.net == pytest.approx(0, abs=TOL)


@pytest.mark.parametrize("sell, trades", [(76.0, False), (78.0, True)])
def test_degradation_cost_blocks_a_cycle_below_20_net_of_losses(sell, trades):
    # Buying at 50 costs 50 / RTE = 56.8 per MWh sold, so the spread net of losses is
    # sell - 56.8: 19.2 at 76 (below the 20 cost), 21.2 at 78 (above it).
    r = solve_day([50.0, sell], DT_15MIN, 0.1, MAIN)
    assert (r.discharged_mwh > TOL) == trades
    assert (r.net > TOL) == trades
    if not trades:
        assert r.net == pytest.approx(0, abs=TOL)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("hours", DURATIONS_H)
def test_bounds_on_a_random_day(spec, hours):
    p = random_day(np.random.default_rng(hours))
    energy = POWER_MW * hours
    for prices, dt in ((p, DT_15MIN), (hourly(p), DT_HOURLY)):
        r = solve_day(prices, dt, energy, spec)
        for x in (r.charge, r.discharge):
            assert x.min() >= -TOL and x.max() <= POWER_MW + TOL
        assert r.energy.min() >= -TOL and r.energy.max() <= energy + TOL
        assert r.energy[-1] == pytest.approx(SOC_START * energy, abs=TOL)
        # The reported energy path is the one the dynamics give for the reported powers.
        np.testing.assert_allclose(r.energy, simulate(r.charge, r.discharge, dt, energy), atol=TOL)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_hourly_schedule_is_constant_within_each_hour_and_feasible_at_15min(spec):
    p = random_day(np.random.default_rng(7))
    energy = 2.0
    r = solve_day(hourly(p), DT_HOURLY, energy, spec)
    c, d = r.quarter_hourly()
    assert len(c) == len(d) == len(p)
    for x in (c, d):
        assert (x.reshape(-1, 4) == x.reshape(-1, 4)[:, :1]).all()
    assert np.abs(c[::4] - r.charge).max() == 0 and np.abs(d[::4] - r.discharge).max() == 0
    # Played at 15 minutes against the quarter-hour prices, it stays feasible and earns the same.
    s = simulate(c, d, DT_15MIN, energy)
    assert s.min() >= -TOL and s.max() <= energy + TOL
    assert s[-1] == pytest.approx(SOC_START * energy, abs=TOL)
    assert np.sum(p * (d - c)) * DT_15MIN == pytest.approx(r.gross, abs=TOL)


def price_days():
    rng = np.random.default_rng(2025)
    days = [random_day(rng) for _ in range(25)]
    days += [random_day(rng, n) for n in (92, 96, 100)]  # DST-length days
    days.append(rng.uniform(-30, 200, 96).round(2))  # no shape, wide spread
    # Nearly flat within each hour, so the 15-minute gain is small, as on most real days.
    days += [np.repeat(hourly(d), 4) + rng.normal(0, 0.5, len(d)) for d in days[:10]]
    return days


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("hours", DURATIONS_H)
def test_15min_value_is_at_least_the_hourly_value(spec, hours):
    for p in price_days():
        net15 = solve_day(p, DT_15MIN, hours, spec).net
        neth = solve_day(hourly(p), DT_HOURLY, hours, spec).net
        assert net15 >= neth - TOL


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("n", [92, 96, 100])
def test_15min_equals_hourly_when_prices_are_flat_within_each_hour(spec, n):
    rng = np.random.default_rng(n)
    for hours in DURATIONS_H:
        ph = hourly(random_day(rng, n))
        net15 = solve_day(np.repeat(ph, 4), DT_15MIN, hours, spec).net
        neth = solve_day(ph, DT_HOURLY, hours, spec).net
        assert net15 == pytest.approx(neth, abs=TOL)


@pytest.mark.parametrize("hours", DURATIONS_H)
def test_cap2_limits_discharge_to_two_cycles(hours):
    # Prices alternating every two hours would pay for about six cycles a day uncapped.
    p = np.tile(np.repeat([10.0, 150.0], 8), 6)
    energy = POWER_MW * hours
    for prices, dt in ((p, DT_15MIN), (hourly(p), DT_HOURLY)):
        assert solve_day(prices, dt, energy, NO_COST).discharged_mwh > 2 * energy + 0.1
        assert solve_day(prices, dt, energy, CAP2).discharged_mwh <= 2 * energy + TOL


@pytest.mark.parametrize("hours", DURATIONS_H)
def test_charge_plus_discharge_stays_within_power_at_negative_prices(hours):
    # At a flat -100 the battery is paid to import, so it burns energy in the losses by charging
    # and discharging at once, keeping sum(d) = RTE * sum(c) to end the day where it started.
    # Without the shared limit, c = 1 and d = RTE in every period earns 100 * (1 - RTE) EUR an
    # hour; with c + d <= 1 the best is sum(c) = periods / (1 + RTE), which earns less.
    energy = POWER_MW * hours
    for n, dt in ((96, DT_15MIN), (24, DT_HOURLY)):
        r = solve_day(np.full(n, -100.0), dt, energy, NO_COST)
        assert (r.charge + r.discharge).max() <= POWER_MW + 1e-9
        unconstrained = 100 * (1 - RTE) * n * dt
        shared = unconstrained / (1 + RTE)
        assert r.net == pytest.approx(shared, abs=TOL)
        assert r.net < unconstrained - 1
        assert r.simultaneous > 0


@pytest.mark.parametrize(
    "hours, net15, neth",
    [
        (1, 216.35462488547944, 71.48959453084461),
        (2, 315.15831395696404, 149.308553553445),
        (4, 388.65213169650485, 174.91324627777712),
    ],
)
def test_main_value_is_unchanged_by_the_shared_power_limit(hours, net15, neth):
    # Values from the model before c + d <= POWER_MW was added. On these days main never
    # charges and discharges in the same period, so the limit cannot bind.
    p = random_day(np.random.default_rng(hours))
    for prices, dt, expected in ((p, DT_15MIN, net15), (hourly(p), DT_HOURLY, neth)):
        r = solve_day(prices, dt, hours, MAIN)
        assert r.net == pytest.approx(expected, abs=1e-6)
        assert r.simultaneous == 0


def test_solver_failure_raises():
    impossible = Spec("negative_cap", 0.0, -1.0)  # discharge cannot be below zero MWh
    with pytest.raises(SolveError, match="status"):
        solve_day(random_day(np.random.default_rng(0)), DT_15MIN, 1.0, impossible)


def dataset_dir(tmp_path, ndays=3):
    """A data dir with a tiny price dataset: ndays 96-period days and one 100-period day."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(ndays + 1):
        day = date(2025, 10, 1) + timedelta(days=i)
        n = 100 if i == ndays else 96
        start = datetime(2025, 9, 30, 22, tzinfo=UTC) + timedelta(days=i)
        for k, price in enumerate(random_day(rng, n)):
            rows.append((start + timedelta(minutes=15 * k), day, k + 1, price, price, 1))
    q = pd.DataFrame(
        rows,
        columns=[
            "delivery_start_utc",
            "delivery_date",
            "period",
            "price_es",
            "price_pt",
            "source_version",
        ],
    )
    h = q.assign(period=(q["period"] - 1) // 4 + 1).groupby(["delivery_date", "period"])
    h = h.agg(price_es=("price_es", "mean")).reset_index()
    ds = tmp_path / "dataset"
    ds.mkdir()
    q.to_parquet(ds / "prices_15min.parquet", index=False)
    h.to_parquet(ds / "prices_hourly.parquet", index=False)
    return tmp_path


def test_cli_writes_daily_summary_and_meta(tmp_path):
    data_dir = dataset_dir(tmp_path)
    assert main(["perfect-foresight", "--data-dir", str(data_dir)]) == 0
    out = data_dir / "results"
    assert sorted(p.name for p in out.iterdir()) == [
        "perfect_foresight_daily.parquet",
        "perfect_foresight_meta.json",
        "perfect_foresight_summary.csv",
    ]
    daily = pd.read_parquet(out / "perfect_foresight_daily.parquet")
    assert len(daily) == 4 * len(DURATIONS_H) * len(SPECS) * 2
    main_rows = daily[daily["spec"] == "main"]
    assert np.allclose(
        main_rows["net_eur_per_mw"],
        main_rows["gross_eur_per_mw"] - main_rows["degradation_eur_per_mw"],
    )
    summary = pd.read_csv(out / "perfect_foresight_summary.csv")
    assert list(zip(summary["duration_h"], summary["spec"], strict=True)) == [
        (h, s.name) for h in DURATIONS_H for s in SPECS
    ]
    assert (summary["days"] == 4).all()
    row = summary[(summary["duration_h"] == 2) & (summary["spec"] == "main")].iloc[0]
    g = daily[(daily["duration_h"] == 2) & (daily["spec"] == "main")]
    v15 = g[g["resolution"] == "15min"]["net_eur_per_mw"].sum()
    vh = g[g["resolution"] == "hourly"]["net_eur_per_mw"].sum()
    assert row["net_uplift_keur_per_mw"] == pytest.approx((v15 - vh) / 1000)
    assert row["net_uplift_pct"] == pytest.approx(100 * (v15 - vh) / vh)
    assert row["net_uplift_eur_per_mw_day"] == pytest.approx((v15 - vh) / 4)
    meta = json.loads((out / "perfect_foresight_meta.json").read_text())
    assert meta["date_range"] == {"start": "2025-10-01", "end": "2025-10-04"}
    assert meta["days"] == 4 and meta["round_trip_efficiency"] == 0.88


def test_hourly_prices_that_are_not_the_mean_are_rejected(tmp_path, capsys):
    data_dir = dataset_dir(tmp_path, ndays=1)
    path = data_dir / "dataset" / "prices_hourly.parquet"
    h = pd.read_parquet(path)
    h.loc[3, "price_es"] += 1
    h.to_parquet(path, index=False)
    assert main(["perfect-foresight", "--data-dir", str(data_dir)]) == 1
    assert "not the mean" in capsys.readouterr().err
    assert not (data_dir / "results").exists()
