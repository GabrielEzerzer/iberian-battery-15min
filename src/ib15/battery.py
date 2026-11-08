"""Perfect-foresight battery arbitrage value on the day-ahead prices, at 15-minute and hourly
resolution.

Each delivery day is a separate linear programme solved with HiGHS. Per period t the battery
charges c_t and discharges d_t MW (both in [0, POWER_MW]); the energy in it evolves as
s_t = s_{t-1} + ETA * c_t * dt - d_t * dt / ETA, stays within [0, energy capacity], and starts
and ends the day at SOC_START of capacity. Both c_t and d_t can be positive in the same period:
that stands for the battery alternating between charging and discharging within the period,
which burns energy in the efficiency losses (worth doing at negative prices). A single
converter can alternate only up to its rated power in total, so c_t + d_t <= POWER_MW. The
objective is gross revenue sum(price_t * (d_t - c_t) * dt) minus a degradation cost per MWh
discharged at the grid side.
The hourly case decides one c and d per hour at the hourly price (the mean of the hour's four
quarter-hours), so its schedule is feasible at 15 minutes and earns the same there.
"""

import contextlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from ib15.dataset import BuildError, _git

POWER_MW = 1.0
DURATIONS_H = (1, 2, 4)
RTE = 0.88
ETA = math.sqrt(RTE)  # charge and discharge efficiency each
SOC_START = 0.5  # fraction of energy capacity at the start and end of every delivery day
DT_15MIN = 0.25
DT_HOURLY = 1.0
SIMULTANEOUS_TOL = 1e-6  # MW; a period with both c and d above this counts as simultaneous


@dataclass(frozen=True)
class Spec:
    name: str
    cost_per_mwh: float  # EUR per MWh discharged at the grid side
    max_cycles: float | None  # full cycles a day, None for no cap


SPECS = (
    Spec("main", 20.0, None),
    Spec("no_cost", 0.0, None),
    Spec("cap2", 0.0, 2.0),
)

# Wear costs (EUR per MWh discharged) for the no-cap sensitivity. A cost that an existing no-cap
# spec already has reuses that spec's solves, so 0 and 20 are the no_cost and main rows.
SENSITIVITY_COSTS = (0.0, 10.0, 20.0, 30.0, 40.0)
SENSITIVITY_SPECS = tuple(
    next(
        (s for s in SPECS if s.max_cycles is None and s.cost_per_mwh == cost),
        Spec(f"cost_{cost:g}", cost, None),
    )
    for cost in SENSITIVITY_COSTS
)
SOLVED_SPECS = SPECS + tuple(s for s in SENSITIVITY_SPECS if s not in SPECS)

# Uncertainty: a circular moving-block bootstrap over days, set in advance, not tuned.
BOOT_BLOCK_DAYS = 7
BOOT_RESAMPLES = 10_000
BOOT_SEED = 20251001
BOOT_PERCENTILES = (2.5, 97.5)  # a 95% percentile interval
UNCERTAINTY_STATS = ("net_uplift_eur_per_mw_day", "net_uplift_pct")


class SolveError(Exception):
    pass


@dataclass
class Dispatch:
    """An optimal day: per-period charge, discharge and end-of-period energy at ``dt`` hours."""

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
        """Charge and discharge expanded to 15-minute periods (power constant within each)."""
        k = round(self.dt / DT_15MIN)
        return np.repeat(self.charge, k), np.repeat(self.discharge, k)


def solve_day(prices, dt: float, energy_mwh: float, spec: Spec) -> Dispatch:
    """Maximise one day's net value. Variables are x = [c, d, s], each of length n."""
    p = np.asarray(prices, dtype=float)
    n = len(p)
    s0 = SOC_START * energy_mwh
    eye = sparse.identity(n, format="csr")
    shift = sparse.eye(n, k=-1, format="csr")
    # s_t - s_{t-1} - ETA*dt*c_t + dt/ETA*d_t = 0, with s_{-1} = s0 moved to the right side.
    a_eq = sparse.hstack([-ETA * dt * eye, (dt / ETA) * eye, eye - shift], format="csr")
    b_eq = np.zeros(n)
    b_eq[0] = s0
    bounds = [(0.0, POWER_MW)] * (2 * n) + [(0.0, energy_mwh)] * (n - 1) + [(s0, s0)]
    # linprog minimises: charging costs p*dt, discharging earns (p - cost)*dt.
    obj = np.concatenate([p * dt, -(p - spec.cost_per_mwh) * dt, np.zeros(n)])
    # c_t + d_t <= POWER_MW: the converter's rated power is shared within each period.
    a_ub = sparse.hstack([eye, eye, sparse.csr_matrix((n, n))], format="csr")
    b_ub = np.full(n, POWER_MW)
    if spec.max_cycles is not None:
        cycles = sparse.csr_matrix(np.concatenate([np.zeros(n), np.full(n, dt), np.zeros(n)]))
        a_ub = sparse.vstack([a_ub, cycles], format="csr")
        b_ub = np.append(b_ub, spec.max_cycles * energy_mwh)
    res = linprog(obj, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
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
        # Periods in which the battery time-shares its converter between charging and
        # discharging (both positive, together at most POWER_MW). With a 0 EUR/MWh cost some of
        # these can be the solver's tie-breaking: in a zero-price period, charging and
        # discharging at once leaves the objective unchanged, so either solution is optimal.
        simultaneous=int(np.sum((c > SIMULTANEOUS_TOL) & (d > SIMULTANEOUS_TOL))),
    )


def daily_values(q: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    """One row per (delivery_date, duration_h, spec, resolution) from the 15-minute and hourly
    price tables. Raises BuildError if the hourly prices are not the means of the quarters."""
    q = q.sort_values(["delivery_date", "period"])
    h = h.sort_values(["delivery_date", "period"])
    hourly = {d: g["price_es"].to_numpy() for d, g in h.groupby("delivery_date")}
    if set(hourly) != set(q["delivery_date"]):
        raise BuildError("the 15-minute and hourly price files cover different days")
    rows = []
    for day, g in q.groupby("delivery_date"):
        pq = g["price_es"].to_numpy()
        ph = hourly.get(day)
        if ph is None or len(pq) != 4 * len(ph):
            raise BuildError(f"{day}: hourly prices do not match the 15-minute periods")
        if not np.allclose(pq.reshape(-1, 4).mean(axis=1), ph, rtol=0, atol=1e-9):
            raise BuildError(f"{day}: an hourly price is not the mean of its four quarter-hours")
        for hours in DURATIONS_H:
            energy = POWER_MW * hours
            for spec in SOLVED_SPECS:
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
    """Sample totals at both resolutions for one (duration_h, spec), the 15-minute uplift and
    mean cycles a day. Percent uplift is NaN when the hourly total is zero (within 1e-6 EUR/MW,
    so solver noise around zero does not give a huge percentage)."""
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
    """One row per (duration_h, spec) of SPECS, from ``_compare``."""
    return pd.DataFrame(
        [
            {"duration_h": hours, "spec": spec.name} | _compare(daily, hours, spec)
            for hours in DURATIONS_H
            for spec in SPECS
        ]
    )


def cost_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per (duration_h, cost_eur_per_mwh) of the no-cap wear-cost grid."""
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
    """Circular moving-block bootstrap of ``n`` days: shape (resamples, n), each row the day
    indices (0..n-1) of one resample. A row joins blocks of ``block`` consecutive days, each
    starting at a uniformly drawn day and wrapping from the last day to the first, cut to ``n``
    days. Draw it once and resample every statistic on the same rows, so comparisons stay
    paired."""
    if n < 1:
        raise ValueError("no days to resample")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(resamples, -(-n // block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(resamples, -1)[:, :n]


def percent(x, ref) -> np.ndarray:
    """100 * x / ref elementwise, NaN where ref is zero (within 1e-6 EUR/MW), as the summaries
    compute their percentages."""
    x, ref = np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(ref) <= 1e-6, np.nan, 100 * x / ref)


def interval(stats) -> tuple[float, float]:
    """The BOOT_PERCENTILES of a statistic over the resamples; NaN if it is undefined on any."""
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
    """The 15-minute net uplift of one (duration_h, spec) on each resample of days (the rows of
    ``idx``, indices into the sorted days), as ``_compare`` computes it: in EUR/MW a day (a mean
    over the days) and in percent of the hourly total (a ratio of sums)."""
    g = daily[(daily["duration_h"] == hours) & (daily["spec"] == spec)]
    w = g.pivot(index="delivery_date", columns="resolution", values="net_eur_per_mw").sort_index()
    v15, vh = w["15min"].to_numpy()[idx], w["hourly"].to_numpy()[idx]
    return {
        "net_uplift_eur_per_mw_day": (v15 - vh).mean(axis=1),
        "net_uplift_pct": percent((v15 - vh).sum(axis=1), vh.sum(axis=1)),
    }


def uncertainty(daily: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Per (duration_h, spec) of the summary and statistic in UNCERTAINTY_STATS: the summary's
    value and its bootstrap interval, every row resampled on the same days."""
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


def _write_all(files: dict[Path, Callable[[Path], object]]) -> None:
    """Replace a set of files all together, or leave the previous set as it was.

    Every file is written to ``<name>.part`` first. Then each existing file is moved aside to
    ``<name>.prev``, so a file another program holds open (on Windows, a CSV open in Excel)
    fails before anything new is in place. Only then are the parts renamed into place and the
    ``.prev`` files deleted. On any failure the files moved aside are moved back and the parts
    are removed.
    """
    part = {p: p.with_name(p.name + ".part") for p in files}
    prev = {p: p.with_name(p.name + ".prev") for p in files}
    # Files left by a run that was killed. While any of its parts is left it had not put all its
    # files in place, so its .prev files are the previous set and go back; otherwise they are
    # stale. shortcut: a file with no previous version that such a run had already put in place
    # stays; a journal file would fix that if it ever matters.
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
        # The parts go last: while one is left, the next run knows this one did not finish.
        for p in placed:
            if p not in moved:
                p.unlink()
        for p in moved:
            prev[p].replace(p)
        for t in part.values():
            t.unlink(missing_ok=True)
        raise
    for p in moved:
        # The new set is in place; a .prev that cannot be deleted now is deleted by the next run.
        with contextlib.suppress(OSError):
            prev[p].unlink()


def perfect_foresight(data_dir: Path) -> dict:
    """Solve every day of the dataset and write the results to ``data_dir/results``. Nothing is
    written unless every solve succeeds. Returns the meta record."""
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
        "simultaneous_tolerance_mw": SIMULTANEOUS_TOL,
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
            out / "perfect_foresight_meta.json": lambda t: t.write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            ),
        }
    )
    return meta
