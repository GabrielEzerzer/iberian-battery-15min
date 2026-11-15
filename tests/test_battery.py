import errno
import json
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ib15.battery import (
    BOOT_RESAMPLES,
    DT_15MIN,
    DT_HOURLY,
    DURATIONS_H,
    ETA,
    POWER_MW,
    RTE,
    SENSITIVITY_COSTS,
    SOC_START,
    SPECS,
    UNCERTAINTY_STATS,
    SolveError,
    Spec,
    _write_all,
    bootstrap_days,
    cost_sensitivity,
    daily_values,
    interval,
    portugal,
    price_agreement,
    solve_day,
    summarise,
    uncertainty,
    uplift_statistics,
)
from ib15.cli import main

MAIN, NO_COST, CAP2 = SPECS
TOL = 1e-6


def random_day(rng, n=96):
    t = np.arange(n)
    shape = 60 + 40 * np.sin(2 * np.pi * t / n) - 50 * np.exp(-(((t - n / 2) / (n / 8)) ** 2))
    p = shape + rng.normal(0, 25, n)
    p[rng.random(n) < 0.05] = 0.0
    return np.round(p, 2)


def hourly(p):
    return p.reshape(-1, 4).mean(axis=1)


def simulate(c, d, dt, energy):
    return SOC_START * energy + np.cumsum(ETA * c * dt - d * dt / ETA)


def test_two_quarter_hours_hand_solved_fill_at_0_then_empty_back_to_the_start_at_100():
    energy = 0.1
    r = solve_day([0.0, 100.0], DT_15MIN, energy, MAIN)
    stored = energy - SOC_START * energy
    power_never_binds = (
        "a 0.1 MWh battery starts with 0.05 MWh; power never binds (0.25 MWh a quarter)"
    )
    np.testing.assert_allclose(
        r.charge, [stored / (ETA * DT_15MIN), 0], atol=1e-9, err_msg=power_never_binds
    )
    np.testing.assert_allclose(
        r.discharge, [0, stored * ETA / DT_15MIN], atol=1e-9, err_msg=power_never_binds
    )
    np.testing.assert_allclose(
        r.energy,
        [energy, SOC_START * energy],
        atol=1e-9,
        err_msg="full at 0, back to 0.05 MWh at 100",
    )
    assert r.discharged_mwh == pytest.approx(stored * ETA)
    assert r.gross == pytest.approx(100 * stored * ETA)
    assert r.degradation == pytest.approx(20 * stored * ETA)
    assert r.net == pytest.approx(80 * stored * ETA)
    assert r.simultaneous == 0


def test_flat_prices_give_no_dispatch_in_main():
    r = solve_day(np.full(96, 55.0), DT_15MIN, 2.0, MAIN)
    assert np.abs(r.charge).max() < TOL and np.abs(r.discharge).max() < TOL
    assert r.gross == pytest.approx(0, abs=TOL) and r.net == pytest.approx(0, abs=TOL)


@pytest.mark.parametrize(
    "sell, trades",
    [
        pytest.param(76.0, False, id="buy-50-costs-56.8-sold-spread-19.2-below-cost-20"),
        pytest.param(78.0, True, id="buy-50-costs-56.8-sold-spread-21.2-above-cost-20"),
    ],
)
def test_degradation_cost_blocks_a_cycle_below_20_net_of_losses(sell, trades):
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
    s = simulate(c, d, DT_15MIN, energy)
    assert s.min() >= -TOL and s.max() <= energy + TOL
    assert s[-1] == pytest.approx(SOC_START * energy, abs=TOL)
    assert np.sum(p * (d - c)) * DT_15MIN == pytest.approx(r.gross, abs=TOL)


def price_days():
    rng = np.random.default_rng(2025)
    days = [random_day(rng) for _ in range(25)]
    days += [random_day(rng, n) for n in (92, 96, 100)]
    days.append(rng.uniform(-30, 200, 96).round(2))
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
    p = np.tile(np.repeat([10.0, 150.0], 8), 6)
    energy = POWER_MW * hours
    for prices, dt in ((p, DT_15MIN), (hourly(p), DT_HOURLY)):
        assert solve_day(prices, dt, energy, NO_COST).discharged_mwh > 2 * energy + 0.1
        assert solve_day(prices, dt, energy, CAP2).discharged_mwh <= 2 * energy + TOL


@pytest.mark.parametrize("hours", DURATIONS_H)
def test_charge_plus_discharge_stays_within_power_at_negative_prices(hours):
    energy = POWER_MW * hours
    for n, dt in ((96, DT_15MIN), (24, DT_HOURLY)):
        r = solve_day(np.full(n, -100.0), dt, energy, NO_COST)
        assert (r.charge + r.discharge).max() <= POWER_MW + 1e-9
        unconstrained_charge_1_discharge_rte_every_period = 100 * (1 - RTE) * n * dt
        shared_limit_best_charges_periods_over_1_plus_rte = (
            unconstrained_charge_1_discharge_rte_every_period / (1 + RTE)
        )
        assert r.net == pytest.approx(shared_limit_best_charges_periods_over_1_plus_rte, abs=TOL)
        assert r.net < unconstrained_charge_1_discharge_rte_every_period - 1, (
            "at -100 the battery burns energy by charging and discharging at once, keeping "
            "sum(d) = RTE * sum(c); without the shared limit it would earn 100 * (1 - RTE) an hour"
        )
        assert r.simultaneous > 0


@pytest.mark.parametrize(
    "hours, net15, neth",
    [
        (1, 216.35462488547944, 71.48959453084461),
        (2, 315.15831395696404, 149.308553553445),
        (4, 388.65213169650485, 174.91324627777712),
    ],
)
def test_main_values_from_before_the_shared_power_limit_still_hold_as_it_never_binds(
    hours, net15, neth
):
    p = random_day(np.random.default_rng(hours))
    for prices, dt, expected in ((p, DT_15MIN, net15), (hourly(p), DT_HOURLY, neth)):
        r = solve_day(prices, dt, hours, MAIN)
        assert r.net == pytest.approx(expected, rel=1e-7)
        assert r.simultaneous == 0


def test_solver_failure_raises():
    impossible = Spec("negative_cap", 0.0, -1.0)
    with pytest.raises(SolveError, match="status"):
        solve_day(random_day(np.random.default_rng(0)), DT_15MIN, 1.0, impossible)


def price_tables(days, pt_days=None):
    rows = []
    for i, (prices, pt) in enumerate(zip(days, pt_days or days, strict=True)):
        day = date(2025, 10, 1) + timedelta(days=i)
        start = datetime(2025, 9, 30, 22, tzinfo=UTC) + timedelta(days=i)
        for k, (price, price_pt) in enumerate(zip(prices, pt, strict=True)):
            rows.append((start + timedelta(minutes=15 * k), day, k + 1, price, price_pt, 1))
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
    h = h.agg(price_es=("price_es", "mean"), price_pt=("price_pt", "mean")).reset_index()
    return q, h


def dataset_dir(tmp_path, ndays=3):
    rng = np.random.default_rng(1)
    q, h = price_tables([random_day(rng, 100 if i == ndays else 96) for i in range(ndays + 1)])
    ds = tmp_path / "dataset"
    ds.mkdir()
    q.to_parquet(ds / "prices_15min.parquet", index=False)
    h.to_parquet(ds / "prices_hourly.parquet", index=False)
    return tmp_path


def test_cli_writes_daily_summary_and_meta(tmp_path):
    data_dir = dataset_dir(tmp_path)
    assert main(["perfect-foresight", "--data-dir", str(data_dir)]) == 0
    out = data_dir / "results"
    assert sorted(p.name for p in out.iterdir()) == RESULT_FILES
    daily = pd.read_parquet(out / "perfect_foresight_daily.parquet")
    assert len(daily) == 4 * len(DURATIONS_H) * (len(SPECS) + 3) * 2
    wide = daily.pivot_table(
        index=["delivery_date", "duration_h", "spec"], columns="resolution", values="net_eur_per_mw"
    )
    assert len(wide) == 4 * len(DURATIONS_H) * (len(SPECS) + 3)
    assert (wide["15min"] >= wide["hourly"] - 1e-6).all()
    q = pd.read_parquet(data_dir / "dataset" / "prices_15min.parquet")
    first = q[q["delivery_date"] == date(2025, 10, 1)].sort_values("period")["price_es"]
    assert wide.loc[(date(2025, 10, 1), 2, "main"), "15min"] == pytest.approx(
        solve_day(first.to_numpy(), DT_15MIN, 2.0, MAIN).net, rel=1e-9
    )
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
    assert meta["cost_sensitivity_eur_per_mwh_discharged"] == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert set(meta["specs"]) == set(daily["spec"])
    assert meta["specs"]["cost_10"] == {
        "cost_eur_per_mwh_discharged": 10.0,
        "max_cycles_per_day": None,
    }
    sens = pd.read_csv(out / "perfect_foresight_cost_sensitivity.csv")
    assert list(zip(sens["duration_h"], sens["cost_eur_per_mwh"], strict=True)) == [
        (h, c) for h in DURATIONS_H for c in SENSITIVITY_COSTS
    ]


RESULT_FILES = [
    "perfect_foresight_cost_sensitivity.csv",
    "perfect_foresight_daily.parquet",
    "perfect_foresight_meta.json",
    "perfect_foresight_portugal.csv",
    "perfect_foresight_summary.csv",
    "perfect_foresight_uncertainty.csv",
]


def previous_results(out):
    out.mkdir(exist_ok=True)
    for name in RESULT_FILES:
        (out / name).write_text(f"previous {name}")
    return {name: (out / name).read_bytes() for name in RESULT_FILES}


def files_in(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def test_a_failed_write_leaves_the_previous_results_intact(tmp_path, monkeypatch):
    data_dir = dataset_dir(tmp_path, ndays=1)
    out = data_dir / "results"
    previous = previous_results(out)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="disk full"):
        main(["perfect-foresight", "--data-dir", str(data_dir)])
    assert files_in(out) == previous

    monkeypatch.undo()
    assert main(["perfect-foresight", "--data-dir", str(data_dir)]) == 0
    assert sorted(p.name for p in out.iterdir()) == RESULT_FILES
    assert len(pd.read_parquet(out / "perfect_foresight_daily.parquet")) > 0


@pytest.mark.parametrize("locked", RESULT_FILES)
def test_a_file_that_cannot_be_replaced_leaves_the_previous_results(
    tmp_path, monkeypatch, capsys, locked
):
    data_dir = dataset_dir(tmp_path, ndays=1)
    out = data_dir / "results"
    previous = previous_results(out)
    replace = Path.replace

    def locked_replace(self, target):
        if out / locked in (self, Path(target)):
            raise PermissionError(errno.EACCES, "used by another process")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", locked_replace)
    assert main(["perfect-foresight", "--data-dir", str(data_dir)]) == 1
    assert f"cannot replace {out / locked}" in capsys.readouterr().err
    assert files_in(out) == previous


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="only Windows refuses to rename or replace a file another program has open",
)
def test_a_results_file_open_in_another_program_leaves_the_previous_results(tmp_path, capsys):
    data_dir = dataset_dir(tmp_path, ndays=1)
    out = data_dir / "results"
    previous = previous_results(out)
    summary = out / "perfect_foresight_summary.csv"
    with open(summary):
        assert main(["perfect-foresight", "--data-dir", str(data_dir)]) == 1, (
            "Python's open() on Windows blocks rename and delete the way Excel does"
        )
    assert f"cannot replace {summary}" in capsys.readouterr().err
    assert files_in(out) == previous
    assert main(["perfect-foresight", "--data-dir", str(data_dir)]) == 0
    assert sorted(p.name for p in out.iterdir()) == RESULT_FILES
    assert summary.read_bytes() != previous[summary.name]


KILLED = {
    "while writing the parts": ({"a": "old a", "b": "old b", "a.part": "half"}, "old"),
    "while moving the old files aside": (
        {"a.prev": "old a", "b": "old b", "a.part": "new a", "b.part": "new b"},
        "old",
    ),
    "while putting the new files in place": (
        {"a": "new a", "a.prev": "old a", "b.prev": "old b", "b.part": "new b"},
        "old",
    ),
    "while deleting the old files": ({"a": "new a", "b": "new b", "b.prev": "old b"}, "new"),
}


@pytest.mark.parametrize("leftovers, later_failure_keeps", KILLED.values(), ids=KILLED.keys())
def test_files_left_by_a_killed_run_neither_block_nor_become_the_result(
    tmp_path, leftovers, later_failure_keeps
):
    a, b = tmp_path / "a", tmp_path / "b"

    def leave(files):
        for p in tmp_path.iterdir():
            p.unlink()
        for name, text in files.items():
            (tmp_path / name).write_text(text)

    def write(text):
        return lambda t: t.write_text(text)

    def fail(t):
        raise OSError("disk full")

    leave(leftovers)
    with pytest.raises(OSError, match="disk full"):
        _write_all({a: write("newest a"), b: fail})
    assert files_in(tmp_path) == {
        "a": f"{later_failure_keeps} a".encode(),
        "b": f"{later_failure_keeps} b".encode(),
    }, (
        "for a and b written in that order, a later run that fails leaves the files from before "
        "the killed run (old), unless that run had put all its new files in place (new)"
    )

    leave(leftovers)
    _write_all({a: write("newest a"), b: write("newest b")})
    assert files_in(tmp_path) == {"a": b"newest a", "b": b"newest b"}


def test_a_failed_rename_removes_a_new_file_with_no_previous_version(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("old a")
    real_replace = Path.replace

    def replace(self, target):
        if self.name == "a.part" and Path(target) == a:
            raise OSError("rename failed")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="rename failed"):
        _write_all({b: lambda t: t.write_text("new b"), a: lambda t: t.write_text("new a")})
    assert files_in(tmp_path) == {"a": b"old a"}


@pytest.fixture(scope="module")
def sensitivity_daily():
    rng = np.random.default_rng(11)
    days = [random_day(rng) for _ in range(3)] + [random_day(rng, 92)]
    days.append(np.tile(np.repeat([10.0, 150.0], 8), 6))
    return daily_values(*price_tables(days))


def test_sensitivity_at_20_and_0_reproduces_main_and_no_cost(sensitivity_daily):
    summary = summarise(sensitivity_daily)
    sens = cost_sensitivity(sensitivity_daily)
    shared = [c for c in summary.columns if c not in ("duration_h", "spec")]
    assert set(shared) < set(sens.columns)
    for cost, spec in ((20.0, "main"), (0.0, "no_cost")):
        for hours in DURATIONS_H:
            a = summary[(summary["duration_h"] == hours) & (summary["spec"] == spec)][shared]
            b = sens[(sens["duration_h"] == hours) & (sens["cost_eur_per_mwh"] == cost)][shared]
            assert len(a) == len(b) == 1
            assert a.iloc[0].equals(b.iloc[0])


@pytest.mark.parametrize("resolution", ["15min", "hourly"])
def test_net_value_and_cycles_fall_as_the_wear_cost_rises(sensitivity_daily, resolution):
    sens = cost_sensitivity(sensitivity_daily)
    net, cycles = f"net_{resolution}_keur_per_mw", f"cycles_per_day_{resolution}"
    for hours in DURATIONS_H:
        g = sens[sens["duration_h"] == hours].sort_values("cost_eur_per_mwh")
        assert list(g["cost_eur_per_mwh"]) == list(SENSITIVITY_COSTS)
        for column in (net, cycles):
            assert (np.diff(g[column].to_numpy()) <= 1e-9).all(), (hours, column)
        assert g[net].iloc[-1] < g[net].iloc[0] and g[cycles].iloc[-1] < g[cycles].iloc[0]


def test_15min_net_is_at_least_hourly_at_every_cost(sensitivity_daily):
    sens = cost_sensitivity(sensitivity_daily)
    assert (sens["net_15min_keur_per_mw"] >= sens["net_hourly_keur_per_mw"] - 1e-9).all()
    assert (sens["net_uplift_keur_per_mw"] > 0).all()


def test_percent_uplift_is_nan_when_the_hourly_total_is_zero():
    daily = daily_values(*price_tables([np.tile([0.0, 100.0], 48)]))
    summary = summarise(daily)
    assert (summary["net_hourly_keur_per_mw"] == 0).all()
    assert (summary["net_15min_keur_per_mw"] > 0).all()
    assert summary["net_uplift_pct"].map(math.isnan).all()
    assert summary["gross_uplift_pct"].map(math.isnan).all()
    daily.loc[daily["resolution"] == "hourly", ["net_eur_per_mw", "gross_eur_per_mw"]] += 1e-9
    noisy = summarise(daily)
    assert noisy["net_uplift_pct"].map(math.isnan).all()
    assert noisy["gross_uplift_pct"].map(math.isnan).all()


def test_hourly_prices_that_are_not_the_mean_are_rejected(tmp_path, capsys):
    data_dir = dataset_dir(tmp_path, ndays=1)
    path = data_dir / "dataset" / "prices_hourly.parquet"
    h = pd.read_parquet(path)
    h.loc[3, "price_es"] += 1
    h.to_parquet(path, index=False)
    assert main(["perfect-foresight", "--data-dir", str(data_dir)]) == 1
    assert "not the mean" in capsys.readouterr().err
    assert not (data_dir / "results").exists()


def test_the_bootstrap_is_reproducible_with_the_seed():
    a = bootstrap_days(30, resamples=50)
    assert np.array_equal(a, bootstrap_days(30, resamples=50))
    assert not np.array_equal(a, bootstrap_days(30, resamples=50, seed=1))
    assert bootstrap_days(30).shape == (BOOT_RESAMPLES, 30)


def test_bootstrap_blocks_are_contiguous_and_wrap_around():
    n, block = 10, 7
    idx = bootstrap_days(n, block=block, resamples=500)
    assert idx.min() >= 0 and idx.max() < n
    for start, length in ((0, block), (block, n - block)):
        b = idx[:, start : start + length]
        assert (np.diff(b, axis=1) % n == 1).all()
    first = idx[:, :block]
    assert ((first[:, :-1] == n - 1) & (first[:, 1:] == 0)).any()


@pytest.mark.parametrize("n", [1, 6, 7, 10, 404])
def test_every_resample_has_as_many_days_as_the_sample(n):
    assert bootstrap_days(n, resamples=20).shape == (20, n)


def pf_daily(v15, vh) -> pd.DataFrame:
    days = [date(2025, 10, 1) + timedelta(days=i) for i in range(len(v15))]
    rows = [
        {
            "delivery_date": d,
            "duration_h": hours,
            "spec": spec.name,
            "resolution": res,
            "gross_eur_per_mw": v[i],
            "degradation_eur_per_mw": 0.0,
            "net_eur_per_mw": v[i],
            "discharged_mwh_per_mw": 1.0,
            "cycles": 1.0,
            "simultaneous_periods": 0,
        }
        for hours in DURATIONS_H
        for spec in SPECS
        for res, v in (("15min", v15), ("hourly", vh))
        for i, d in enumerate(days)
    ]
    return pd.DataFrame(rows)


def test_uncertainty_estimates_are_the_summary_values(tmp_path):
    data_dir = dataset_dir(tmp_path)
    assert main(["perfect-foresight", "--data-dir", str(data_dir)]) == 0
    out = data_dir / "results"
    summary = pd.read_csv(out / "perfect_foresight_summary.csv")
    unc = pd.read_csv(out / "perfect_foresight_uncertainty.csv")
    assert len(unc) == len(summary) * len(UNCERTAINTY_STATS)
    for _, r in unc.iterrows():
        row = summary[(summary["duration_h"] == r["duration_h"]) & (summary["spec"] == r["spec"])]
        assert r["estimate"] == row[r["statistic"]].iloc[0]
        assert r["days"] == 4
    meta = json.loads((out / "perfect_foresight_meta.json").read_text())
    assert meta["uncertainty"]["block_days"] == 7 and meta["uncertainty"]["resamples"] == 10_000
    daily = pd.read_parquet(out / "perfect_foresight_daily.parquet")
    same = np.arange(4)[None]
    for _, row in summary.iterrows():
        stats = uplift_statistics(daily, row["duration_h"], row["spec"], same)
        for name in UNCERTAINTY_STATS:
            assert stats[name][0] == pytest.approx(row[name], rel=1e-9), (row["spec"], name)


def test_the_interval_collapses_when_every_day_is_the_same():
    daily = pf_daily(np.full(30, 150.0), np.full(30, 120.0))
    unc = uncertainty(daily, summarise(daily))
    assert np.allclose(unc["ci_low"], unc["estimate"], rtol=1e-12, atol=0)
    assert np.allclose(unc["ci_high"], unc["estimate"], rtol=1e-12, atol=0)
    assert unc.groupby("statistic")["estimate"].first().round(9).to_dict() == {
        "net_uplift_eur_per_mw_day": 30.0,
        "net_uplift_pct": 25.0,
    }


def test_the_interval_matches_the_normal_one_for_independent_days():
    n = 1000
    u = np.random.default_rng(3).normal(50, 20, n)
    daily = pf_daily(100 + u, np.full(n, 100.0))
    stats = uplift_statistics(daily, 1, "main", bootstrap_days(n))
    lo, hi = interval(stats["net_uplift_eur_per_mw_day"])
    half = 1.96 * u.std(ddof=1) / math.sqrt(n)
    assert (hi - lo) / 2 == pytest.approx(half, rel=0.15), (
        "sd 20 over 1000 independent days: half-width about 1.96 * 20 / sqrt(1000) = 1.24"
    )
    assert (hi + lo) / 2 == pytest.approx(u.mean(), abs=0.15 * half)


def test_an_undefined_statistic_has_no_bounds():
    assert all(math.isnan(x) for x in interval([1.0, math.nan, 2.0]))


def portugal_table(days, pt_days):
    q, h = price_tables(days, pt_days)
    return portugal(daily_values(q, h), daily_values(q, h, "price_pt", (MAIN,)))


def test_portugal_rows_equal_the_spanish_ones_when_the_prices_are_equal(tmp_path):
    data_dir = dataset_dir(tmp_path)
    assert main(["perfect-foresight", "--data-dir", str(data_dir)]) == 0
    out = data_dir / "results"
    pt = pd.read_csv(out / "perfect_foresight_portugal.csv")
    summary = pd.read_csv(out / "perfect_foresight_summary.csv")
    unc = pd.read_csv(out / "perfect_foresight_uncertainty.csv")
    assert list(zip(pt["duration_h"], pt["market"], strict=True)) == [
        (h, m) for h in DURATIONS_H for m in ("es", "pt")
    ]
    es, pt_rows = pt[pt["market"] == "es"], pt[pt["market"] == "pt"]
    main_rows = summary[summary["spec"] == "main"]
    shared = [c for c in summary.columns if c != "spec"]
    for rows in (es, pt_rows):
        assert rows[shared].reset_index(drop=True).equals(main_rows[shared].reset_index(drop=True))
    for name in UNCERTAINTY_STATS:
        u = unc[(unc["spec"] == "main") & (unc["statistic"] == name)]
        for rows in (es, pt_rows):
            assert list(rows[f"{name}_ci_low"]) == list(u["ci_low"])
            assert list(rows[f"{name}_ci_high"]) == list(u["ci_high"])


def test_portugal_rows_follow_the_portuguese_price():
    rng = np.random.default_rng(5)
    hourly_shape = np.repeat(np.round(rng.uniform(20, 120, 24), 2), 4)
    flat = np.repeat(50.0, 96)
    days = [hourly_shape, flat]
    pt_days = [hourly_shape, flat + np.tile([-50.0, 50.0], 48)]
    t = portugal_table(days, pt_days).set_index(["duration_h", "market"])
    for hours in DURATIONS_H:
        es, pt = t.loc[(hours, "es")], t.loc[(hours, "pt")]
        assert es["net_uplift_eur_per_mw_day"] == pytest.approx(0, abs=1e-6)
        assert pt["net_uplift_eur_per_mw_day"] > 1
        assert pt["net_hourly_keur_per_mw"] == pytest.approx(es["net_hourly_keur_per_mw"])
        direct = sum(solve_day(p, DT_15MIN, hours * POWER_MW, MAIN).net for p in pt_days)
        assert pt["net_15min_keur_per_mw"] == pytest.approx(direct / 1000, rel=1e-9)


def test_price_agreement_on_a_hand_computed_case():
    es = [10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 50.0, 50.0]
    pt = [10.0, 20.0, 30.0, 40.0, 46.0, 50.0, 58.0, 50.0]
    second_day_100_higher_same_spreads = [[p + 100 for p in day] for day in (es, pt)]
    q, _ = price_tables(
        [es, second_day_100_higher_same_spreads[0]], [pt, second_day_100_higher_same_spreads[1]]
    )
    a = price_agreement(q.iloc[::-1])
    assert a["quarter_hours"] == 16
    assert a["share_pt_equal_to_es"] == 12 / 16
    assert a["mean_abs_pt_minus_es_eur_per_mwh"] == pytest.approx(2 * (4 + 8) / 16)
    assert a["mean_within_hour_spread_eur_per_mwh"] == pytest.approx({"es": 15.0, "pt": 21.0}), (
        "spreads each day: Spain 30 and 0, Portugal 30 and 12; pooling the two days by hour of "
        "day would give a spread of 130 in the first hour"
    )


def test_portugal_intervals_are_the_bootstrap_intervals_of_each_market():
    rng = np.random.default_rng(8)
    n = 30
    es, pt = (
        pf_daily(100 + rng.normal(10, 5, n), np.full(n, 100.0)),
        pf_daily(100 + rng.normal(8, 9, n), np.full(n, 100.0)),
    )
    t = portugal(es, pt).set_index(["duration_h", "market"])
    idx = bootstrap_days(n)
    for hours in DURATIONS_H:
        for market, daily in (("es", es), ("pt", pt)):
            stats = uplift_statistics(daily, hours, "main", idx)
            for name in UNCERTAINTY_STATS:
                lo, hi = interval(stats[name])
                assert lo < hi
                assert t.loc[(hours, market), f"{name}_ci_low"] == lo
                assert t.loc[(hours, market), f"{name}_ci_high"] == hi
