# Perfect-foresight battery arbitrage value on the day-ahead prices, at 15-minute and hourly
# resolution. The linear programme is set out in the README, "Method notes".
import contextlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from ib15.dataset import BuildError, _git

POWER_MW = 1.0
DURATIONS_H = (1, 2, 4)
RTE = 0.88
ETA = math.sqrt(RTE)
SOC_START = 0.0
DT_15MIN = 0.25
DT_HOURLY = 1.0
SIMULTANEOUS_TOL_MW = 1e-6


@dataclass(frozen=True)
class Spec:
    name: str
    cost_per_mwh: float
    max_cycles: float | None


SPECS = (
    Spec("main", 20.0, None),
    Spec("no_cost", 0.0, None),
    Spec("cap2", 0.0, 2.0),
)
MAIN_SPEC = SPECS[0]

SENSITIVITY_COSTS = (0.0, 10.0, 20.0, 30.0, 40.0)
SENSITIVITY_SPECS = tuple(
    next(
        (s for s in SPECS if s.max_cycles is None and s.cost_per_mwh == cost),
        Spec(f"cost_{cost:g}", cost, None),
    )
    for cost in SENSITIVITY_COSTS
)
SOLVED_SPECS = SPECS + tuple(s for s in SENSITIVITY_SPECS if s not in SPECS)

BOOT_BLOCK_DAYS = 7
BOOT_RESAMPLES = 10_000
BOOT_SEED = 20251001
BOOT_PERCENTILES = (2.5, 97.5)
UNCERTAINTY_STATS = ("net_uplift_eur_per_mw_day", "net_uplift_pct")
WINDOWS = {
    "oct_nov_2025": (date(2025, 10, 1), date(2025, 11, 30)),
    "full": (None, None),
}


class SolveError(Exception):
    pass


@dataclass
class Dispatch:
    charge: np.ndarray
    discharge: np.ndarray
    energy: np.ndarray
    dt: float
    gross: float
    degradation: float
    net: float
    discharged_mwh: float
    simultaneous: int

    def quarter_hourly(self) -> tuple[np.ndarray, np.ndarray]:
        k = round(self.dt / DT_15MIN)
        return np.repeat(self.charge, k), np.repeat(self.discharge, k)


def solve_day(prices, dt: float, energy_mwh: float, spec: Spec) -> Dispatch:
    p = np.asarray(prices, dtype=float)
    n = len(p)
    s0 = SOC_START * energy_mwh
    eye = sparse.identity(n, format="csr")
    shift = sparse.eye(n, k=-1, format="csr")
    energy_balance = sparse.hstack([-ETA * dt * eye, (dt / ETA) * eye, eye - shift], format="csr")
    energy_balance_rhs = np.zeros(n)
    energy_balance_rhs[0] = s0
    bounds = [(0.0, POWER_MW)] * (2 * n) + [(0.0, energy_mwh)] * (n - 1) + [(s0, s0)]
    cost = np.concatenate([p * dt, -(p - spec.cost_per_mwh) * dt, np.zeros(n)])
    limits = sparse.hstack([eye, eye, sparse.csr_matrix((n, n))], format="csr")
    limits_rhs = np.full(n, POWER_MW)
    if spec.max_cycles is not None:
        discharged = sparse.csr_matrix(np.concatenate([np.zeros(n), np.full(n, dt), np.zeros(n)]))
        limits = sparse.vstack([limits, discharged], format="csr")
        limits_rhs = np.append(limits_rhs, spec.max_cycles * energy_mwh)
    res = linprog(
        cost,
        A_ub=limits,
        b_ub=limits_rhs,
        A_eq=energy_balance,
        b_eq=energy_balance_rhs,
        bounds=bounds,
        method="highs",
    )
    if res.status != 0:
        raise SolveError(
            f"HiGHS did not find an optimum ({spec.name}, {energy_mwh} MWh, dt {dt} h, "
            f"{n} periods): status {res.status}, {res.message}"
        )
    c, d, s = res.x[:n], res.x[n : 2 * n], res.x[2 * n :]
    gross = float(np.sum(p * (d - c)) * dt)
    discharged = float(np.sum(d) * dt)
    degradation = spec.cost_per_mwh * discharged
    return Dispatch(
        charge=c,
        discharge=d,
        energy=s,
        dt=dt,
        gross=gross,
        degradation=degradation,
        net=gross - degradation,
        discharged_mwh=discharged,
        simultaneous=int(np.sum((c > SIMULTANEOUS_TOL_MW) & (d > SIMULTANEOUS_TOL_MW))),
    )


def daily_values(
    q: pd.DataFrame, h: pd.DataFrame, price: str = "price_es", specs=SOLVED_SPECS
) -> pd.DataFrame:
    q = q.sort_values(["delivery_date", "period"])
    h = h.sort_values(["delivery_date", "period"])
    hourly = {d: g[price].to_numpy() for d, g in h.groupby("delivery_date")}
    if set(hourly) != set(q["delivery_date"]):
        raise BuildError("the 15-minute and hourly price files cover different days")
    rows = []
    for day, g in q.groupby("delivery_date"):
        pq = g[price].to_numpy()
        ph = hourly.get(day)
        if ph is None or len(pq) != 4 * len(ph):
            raise BuildError(f"{day}: hourly prices do not match the 15-minute periods")
        if not np.allclose(pq.reshape(-1, 4).mean(axis=1), ph, rtol=0, atol=1e-9):
            raise BuildError(f"{day}: an hourly price is not the mean of its four quarter-hours")
        for hours in DURATIONS_H:
            energy = POWER_MW * hours
            for spec in specs:
                for resolution, prices, dt in (("15min", pq, DT_15MIN), ("hourly", ph, DT_HOURLY)):
                    r = solve_day(prices, dt, energy, spec)
                    rows.append(
                        {
                            "delivery_date": day,
                            "duration_h": hours,
                            "spec": spec.name,
                            "resolution": resolution,
                            "gross_eur_per_mw": r.gross,
                            "degradation_eur_per_mw": r.degradation,
                            "net_eur_per_mw": r.net,
                            "discharged_mwh_per_mw": r.discharged_mwh,
                            "cycles": r.discharged_mwh / energy,
                            "simultaneous_periods": r.simultaneous,
                        }
                    )
    return pd.DataFrame(rows)


def _compare(daily: pd.DataFrame, hours: int, spec: Spec) -> dict:
    g = daily[(daily["duration_h"] == hours) & (daily["spec"] == spec.name)]
    r15 = g[g["resolution"] == "15min"].set_index("delivery_date")
    rh = g[g["resolution"] == "hourly"].set_index("delivery_date")
    if not r15.index.sort_values().equals(rh.index.sort_values()):
        raise BuildError(f"{hours} h {spec.name}: resolutions cover different days")
    days = len(r15)
    row = {"days": days}
    for label in ("net", "gross"):
        v15 = r15[f"{label}_eur_per_mw"].sum()
        vh = rh[f"{label}_eur_per_mw"].sum()
        row[f"{label}_15min_keur_per_mw"] = v15 / 1000
        row[f"{label}_hourly_keur_per_mw"] = vh / 1000
        row[f"{label}_uplift_keur_per_mw"] = (v15 - vh) / 1000
        row[f"{label}_uplift_eur_per_mw_day"] = (v15 - vh) / days
        row[f"{label}_uplift_pct"] = (
            math.nan if math.isclose(vh, 0, abs_tol=1e-6) else 100 * (v15 - vh) / vh
        )
    row["degradation_15min_keur_per_mw"] = r15["degradation_eur_per_mw"].sum() / 1000
    row["degradation_hourly_keur_per_mw"] = rh["degradation_eur_per_mw"].sum() / 1000
    row["cycles_per_day_15min"] = r15["cycles"].mean()
    row["cycles_per_day_hourly"] = rh["cycles"].mean()
    return row


def summarise(daily: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"duration_h": hours, "spec": spec.name} | _compare(daily, hours, spec)
            for hours in DURATIONS_H
            for spec in SPECS
        ]
    )


def cost_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"duration_h": hours, "cost_eur_per_mwh": spec.cost_per_mwh}
            | _compare(daily, hours, spec)
            for hours in DURATIONS_H
            for spec in SENSITIVITY_SPECS
        ]
    )


def bootstrap_days(
    n: int, block: int = BOOT_BLOCK_DAYS, resamples: int = BOOT_RESAMPLES, seed: int = BOOT_SEED
) -> np.ndarray:
    if n < 1:
        raise ValueError("no days to resample")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(resamples, -(-n // block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(resamples, -1)[:, :n]


def percent(x, ref) -> np.ndarray:
    x, ref = np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(ref) <= 1e-6, np.nan, 100 * x / ref)


def interval(stats) -> tuple[float, float]:
    stats = np.asarray(stats, dtype=float)
    if np.isnan(stats).any():
        return math.nan, math.nan
    lo, hi = np.percentile(stats, BOOT_PERCENTILES)
    return float(lo), float(hi)


def bootstrap_meta() -> dict:
    return {
        "method": "circular moving-block bootstrap over days (numpy.random.default_rng)",
        "block_days": BOOT_BLOCK_DAYS,
        "resamples": BOOT_RESAMPLES,
        "seed": BOOT_SEED,
        "interval": f"95% percentile interval, percentiles {list(BOOT_PERCENTILES)}",
        "statistic": "recomputed on each resample as the point estimate is: a ratio of sums or a"
        " mean over the resampled days; every series is resampled on the same days",
        "undefined": "both bounds are empty when the statistic's denominator is zero (within"
        " 1e-6 EUR/MW) on any resample",
    }


def uplift_statistics(daily: pd.DataFrame, hours: int, spec: str, idx) -> dict[str, np.ndarray]:
    g = daily[(daily["duration_h"] == hours) & (daily["spec"] == spec)]
    w = g.pivot(index="delivery_date", columns="resolution", values="net_eur_per_mw").sort_index()
    v15, vh = w["15min"].to_numpy()[idx], w["hourly"].to_numpy()[idx]
    return {
        "net_uplift_eur_per_mw_day": (v15 - vh).mean(axis=1),
        "net_uplift_pct": percent((v15 - vh).sum(axis=1), vh.sum(axis=1)),
    }


def uncertainty(daily: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    idx = bootstrap_days(daily["delivery_date"].nunique())
    s = summary.set_index(["duration_h", "spec"])
    rows = []
    for hours, spec in s.index:
        stats = uplift_statistics(daily, hours, spec, idx)
        for name in UNCERTAINTY_STATS:
            lo, hi = interval(stats[name])
            rows.append(
                {
                    "duration_h": hours,
                    "spec": spec,
                    "statistic": name,
                    "days": s.loc[(hours, spec), "days"],
                    "estimate": s.loc[(hours, spec), name],
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def in_window(daily: pd.DataFrame, window: str) -> pd.DataFrame:
    start, end = WINDOWS[window]
    d = daily["delivery_date"]
    start = d.min() if start is None else start
    end = d.max() if end is None else end
    return daily[(d >= start) & (d <= end)]


def portugal(daily: pd.DataFrame, daily_pt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in ("full", *(w for w in WINDOWS if w != "full")):
        es, pt = in_window(daily, window), in_window(daily_pt, window)
        days = es["delivery_date"]
        if days.empty:
            continue
        idx = bootstrap_days(days.nunique())
        span = {"window": window, "start": days.min(), "end": days.max()}
        for hours in DURATIONS_H:
            for market, frame in (("es", es), ("pt", pt)):
                row = span | {"duration_h": hours, "market": market}
                row |= _compare(frame, hours, MAIN_SPEC)
                stats = uplift_statistics(frame, hours, MAIN_SPEC.name, idx)
                for name in UNCERTAINTY_STATS:
                    row[f"{name}_ci_low"], row[f"{name}_ci_high"] = interval(stats[name])
                rows.append(row)
    return pd.DataFrame(rows)


def price_agreement(q: pd.DataFrame) -> dict:
    q = q.sort_values(["delivery_date", "period"])
    diff = (q["price_pt"] - q["price_es"]).abs()
    hours = q.assign(hour=(q["period"] - 1) // 4).groupby(["delivery_date", "hour"])
    spread = hours[["price_pt", "price_es"]].max() - hours[["price_pt", "price_es"]].min()
    return {
        "quarter_hours": len(q),
        "share_pt_equal_to_es": float((diff <= 1e-9).mean()),
        "mean_abs_pt_minus_es_eur_per_mwh": float(diff.mean()),
        "mean_within_hour_spread_eur_per_mwh": {
            "pt": float(spread["price_pt"].mean()),
            "es": float(spread["price_es"].mean()),
        },
    }


def _write_all(files: dict[Path, Callable[[Path], object]]) -> None:
    part = {p: p.with_name(p.name + ".part") for p in files}
    prev = {p: p.with_name(p.name + ".prev") for p in files}
    killed_before_done = any(t.exists() for t in part.values())
    for p, old in prev.items():
        if not old.exists():
            continue
        if killed_before_done:
            old.replace(p)
        else:
            old.unlink()
    moved, placed = [], []
    try:
        for p, write in files.items():
            write(part[p])
        for p in files:
            if p.exists():
                try:
                    p.replace(prev[p])
                except PermissionError as e:
                    raise BuildError(
                        f"cannot replace {p}: {e.strerror}. If it is open in another program, "
                        "close it and run again. The previous results are unchanged."
                    ) from e
                moved.append(p)
        for p in files:
            part[p].replace(p)
            placed.append(p)
    except BaseException:
        for p in placed:
            if p not in moved:
                p.unlink()
        for p in moved:
            prev[p].replace(p)
        for t in part.values():
            t.unlink(missing_ok=True)
        raise
    for p in moved:
        with contextlib.suppress(OSError):
            prev[p].unlink()


def perfect_foresight(data_dir: Path) -> dict:
    ds = data_dir / "dataset"
    paths = [ds / "prices_15min.parquet", ds / "prices_hourly.parquet"]
    for p in paths:
        if not p.exists():
            raise BuildError(f"{p} does not exist; run `ib15 build` first")
    q, h = (pd.read_parquet(p) for p in paths)
    started = time.perf_counter()
    daily = daily_values(q, h)
    summary = summarise(daily)
    sensitivity = cost_sensitivity(daily)
    unc = uncertainty(daily, summary)
    pt = portugal(daily, daily_values(q, h, "price_pt", (MAIN_SPEC,)))
    status = _git("status", "--porcelain")
    meta = {
        "power_mw": POWER_MW,
        "durations_h": list(DURATIONS_H),
        "round_trip_efficiency": RTE,
        "charge_efficiency": ETA,
        "discharge_efficiency": ETA,
        "soc_start_and_end_fraction": SOC_START,
        "price": "price_es",
        "specs": {
            s.name: {
                "cost_eur_per_mwh_discharged": s.cost_per_mwh,
                "max_cycles_per_day": s.max_cycles,
            }
            for s in SOLVED_SPECS
        },
        "cost_sensitivity_eur_per_mwh_discharged": list(SENSITIVITY_COSTS),
        "hourly": "one charge and discharge power per hour at the hourly price (mean of quarters)",
        "solver": "scipy.optimize.linprog, method highs",
        "simultaneous_tolerance_mw": SIMULTANEOUS_TOL_MW,
        "simultaneous_periods": {
            f"{spec} {res}": int(n)
            for (spec, res), n in daily.groupby(["spec", "resolution"])["simultaneous_periods"]
            .sum()
            .items()
        },
        "date_range": {
            "start": str(min(daily["delivery_date"])),
            "end": str(max(daily["delivery_date"])),
        },
        "days": int(daily["delivery_date"].nunique()),
        "uncertainty": bootstrap_meta() | {"statistics": list(UNCERTAINTY_STATS)},
        "portugal_spec": MAIN_SPEC.name,
        "portugal_windows": {
            w: {"start": None if a is None else str(a), "end": None if b is None else str(b)}
            for w, (a, b) in WINDOWS.items()
        },
        "portugal_price_agreement": price_agreement(q),
        "solve_seconds": round(time.perf_counter() - started, 1),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }
    out = data_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    _write_all(
        {
            out / "perfect_foresight_daily.parquet": lambda t: daily.to_parquet(t, index=False),
            out / "perfect_foresight_summary.csv": lambda t: summary.to_csv(t, index=False),
            out / "perfect_foresight_cost_sensitivity.csv": lambda t: sensitivity.to_csv(
                t, index=False
            ),
            out / "perfect_foresight_uncertainty.csv": lambda t: unc.to_csv(t, index=False),
            out / "perfect_foresight_portugal.csv": lambda t: pt.to_csv(t, index=False),
            out / "perfect_foresight_meta.json": lambda t: t.write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            ),
        }
    )
    return meta
