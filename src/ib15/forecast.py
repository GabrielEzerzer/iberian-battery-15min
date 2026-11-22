# Day-ahead price forecasts from public data available at bid time, and the battery value they
# capture at 15-minute and hourly resolution.
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, spearmanr

from ib15 import battery
from ib15.dataset import BuildError, _git
from ib15.eda import DPI, fig_example_day, quarter_deviation
from ib15.omie import MADRID
from ib15.weather import LOCATIONS, VARIABLES

SLOTS = 96
HOURS = SLOTS // 4
QUARTER_POSITION = np.array([-1.5, -0.5, 0.5, 1.5])
WINDOW_DAYS = 56
LAGS = (1, 2, 7)


def first_eval(window: int) -> int:
    return window + max(LAGS)


FIRST_EVAL = first_eval(WINDOW_DAYS)
MODELS = ("naive", "expert", "expert_weather")
NAIVE_WEEKLY_WEEKDAYS = (0, 5, 6)
SENSITIVITY_WINDOWS = (28, 56, 112)
SENSITIVITY_MODELS = ("expert", "expert_weather")
HOLIDAYS = tuple(
    date.fromisoformat(d)
    for d in (
        "2025-01-01",
        "2025-01-06",
        "2025-04-18",
        "2025-05-01",
        "2025-08-15",
        "2025-10-12",
        "2025-11-01",
        "2025-12-06",
        "2025-12-08",
        "2025-12-25",
        "2026-01-01",
        "2026-01-06",
        "2026-04-03",
        "2026-05-01",
        "2026-08-15",
        "2026-10-12",
        "2026-11-01",
        "2026-12-06",
        "2026-12-08",
        "2026-12-25",
    )
)
HOLIDAY_YEARS = (HOLIDAYS[0].year, HOLIDAYS[-1].year)
EXPERT_FEATURES = (
    "intercept",
    "price same slot D-1",
    "price same slot D-2",
    "price same slot D-7",
    "min of D-1",
    "max of D-1",
    "last slot of D-1",
    "Monday",
    "Saturday",
    "Sunday",
    "second quarter of the hour",
    "third quarter of the hour",
    "fourth quarter of the hour",
)
RADIATION = "shortwave_radiation"
HOUR_WEATHER = tuple(v for v in VARIABLES if v != RADIATION)
WEATHER_FEATURES = tuple(f"national mean {v}, same hour on D" for v in HOUR_WEATHER)
SHARED_WEATHER_FEATURES = (
    f"national mean {RADIATION}, same hour on D",
    f"change in national mean {RADIATION} across the same hour on D, times the quarter's"
    " position in the hour (-1.5, -0.5, 0.5, 1.5)",
)
DM_LAG = 7
DM_LAGS = (DM_LAG, 14, 28)
SPEC = next(s for s in battery.SPECS if s.name == "main")
BOUND_TOL_EUR_PER_MW = 1e-6
PRICE_COLUMNS = ["delivery_start_utc", "delivery_date", "period", "price_es"]
UPLIFT_DIFFS = (("expert", "naive"), ("expert_weather", "expert"))
EXAMPLE_DURATION_H = 1
EXAMPLE_MODEL = "expert_weather"


def local_slots(starts_utc: pd.Series) -> np.ndarray:
    lt = starts_utc.dt.tz_convert(MADRID)
    return (lt.dt.hour * 4 + lt.dt.minute // 15).to_numpy()


def to_slots(values: np.ndarray, slots: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    counts = np.bincount(slots, minlength=SLOTS)
    missing = np.flatnonzero(counts == 0)
    if len(counts) != SLOTS or counts.max() > 2 or (missing < 4).any():
        raise BuildError(f"a day's periods do not map onto the {SLOTS} local quarter-hours")
    sums = np.zeros((SLOTS, *values.shape[1:]))
    np.add.at(sums, slots, values)
    shape = (SLOTS,) + (1,) * (values.ndim - 1)
    out = sums / np.maximum(counts, 1).reshape(shape)
    for s in missing:
        out[s] = out[s - 4]
    return out


def national_weather(df: pd.DataFrame, variables) -> np.ndarray:
    return np.column_stack(
        [
            df[[f"{VARIABLES[v].column_prefix}_{loc}" for loc in LOCATIONS]].mean(axis=1).to_numpy()
            for v in variables
        ]
    )


def radiation_ramp(radiation: np.ndarray) -> np.ndarray:
    hourly = np.asarray(radiation, dtype=float).reshape(-1, 4).mean(axis=1)
    return np.repeat(np.gradient(hourly), 4)


def slot_arrays(pw: pd.DataFrame):
    hour_wx = national_weather(pw, HOUR_WEATHER)
    rad = national_weather(pw, (RADIATION,))[:, 0]
    dates, prices, weather, shared, slots = [], [], [], [], []
    for day, idx in pw.groupby("delivery_date", sort=True).indices.items():
        s = local_slots(pw["delivery_start_utc"].iloc[idx])
        ramp = radiation_ramp(rad[idx]) * QUARTER_POSITION[s % 4]
        dates.append(day)
        prices.append(to_slots(pw["price_es"].to_numpy()[idx], s))
        weather.append(to_slots(hour_wx[idx], s))
        shared.append(to_slots(np.column_stack([rad[idx], ramp]), s))
        slots.append(s)
    return dates, np.array(prices), np.array(weather), np.array(shared), slots


def day_type(d: date) -> int:
    return 6 if d in HOLIDAYS else d.weekday()


def features(dates: list[date], p: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    n = len(dates)
    k = len(EXPERT_FEATURES) + (0 if w is None else w.shape[2])
    f = np.full((n, SLOTS, k), np.nan)
    for j in range(max(LAGS), n):
        prev = p[j - 1]
        wd = day_type(dates[j])
        cols = [
            np.ones(SLOTS),
            prev,
            p[j - 2],
            p[j - 7],
            np.full(SLOTS, prev.min()),
            np.full(SLOTS, prev.max()),
            np.full(SLOTS, prev[-1]),
            np.full(SLOTS, float(wd == 0)),
            np.full(SLOTS, float(wd == 5)),
            np.full(SLOTS, float(wd == 6)),
            *(np.tile(np.arange(4) == k, HOURS).astype(float) for k in (1, 2, 3)),
        ]
        if w is not None:
            cols += list(w[j].T)
        f[j] = np.column_stack(cols)
    return f


def _rows(a: np.ndarray, days: slice, hour: int) -> np.ndarray:
    return a[days, 4 * hour : 4 * hour + 4].reshape(-1, *a.shape[2:])


def _residual(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def ols_forecast(
    f: np.ndarray,
    p: np.ndarray,
    t: int,
    window: int = WINDOW_DAYS,
    shared: np.ndarray | None = None,
) -> np.ndarray:
    if t < first_eval(window):
        raise ValueError(f"day {t} has no full window; the first is {first_eval(window)}")
    past = slice(t - window, t)
    offset = np.zeros(p.shape)
    if shared is not None:
        r = np.vstack(
            [
                _residual(
                    _rows(f, past, h), np.column_stack([_rows(shared, past, h), _rows(p, past, h)])
                )
                for h in range(HOURS)
            ]
        )
        offset = shared @ np.linalg.lstsq(r[:, :-1], r[:, -1], rcond=None)[0]
    out = np.empty(SLOTS)
    for h in range(HOURS):
        beta = np.linalg.lstsq(_rows(f, past, h), _rows(p - offset, past, h), rcond=None)[0]
        today = slice(4 * h, 4 * h + 4)
        out[today] = f[t, today] @ beta + offset[t, today]
    return out


def naive_forecast(dates: list[date], p: np.ndarray, t: int) -> np.ndarray:
    if dates[t] in HOLIDAYS:
        days_since_sunday = dates[t].weekday() + 1
        return p[t - days_since_sunday]
    return p[t - 7] if dates[t].weekday() in NAIVE_WEEKLY_WEEKDAYS else p[t - 1]


def forecast_slots(
    dates: list[date],
    p: np.ndarray,
    w: np.ndarray,
    shared: np.ndarray,
    targets,
    window: int = WINDOW_DAYS,
) -> dict[str, np.ndarray]:
    f_expert, f_weather = features(dates, p), features(dates, p, w)
    return {
        "naive": np.array([naive_forecast(dates, p, t) for t in targets]),
        "expert": np.array([ols_forecast(f_expert, p, t, window) for t in targets]),
        "expert_weather": np.array(
            [ols_forecast(f_weather, p, t, window, shared) for t in targets]
        ),
    }


def forecast_periods(
    pw: pd.DataFrame, window: int = WINDOW_DAYS, first_index: int | None = None
) -> pd.DataFrame:
    dates, p, w, shared, slots = slot_arrays(pw)
    first_index = first_eval(window) if first_index is None else first_index
    targets = range(first_index, len(dates))
    fc = forecast_slots(dates, p, w, shared, targets, window)
    first = dates[first_index]
    out = pw.loc[pw["delivery_date"] >= first, PRICE_COLUMNS].reset_index(drop=True)
    for m in MODELS:
        out[m] = np.concatenate([fc[m][i][slots[t]] for i, t in enumerate(targets)])
    return out


def _hourly(fc: pd.DataFrame, col: str) -> pd.Series:
    return fc.groupby([fc["delivery_date"], (fc["period"] - 1) // 4])[col].mean()


def errors(fc: pd.DataFrame) -> pd.DataFrame:
    actual_dev = quarter_deviation(fc)
    actual_h = _hourly(fc, "price_es")
    rows = []
    for m in MODELS:
        e = fc[m] - fc["price_es"]
        eh = _hourly(fc, m) - actual_h
        dev = quarter_deviation(fc.assign(price_es=fc[m])) - actual_dev
        rows.append(
            {
                "model": m,
                "days": fc["delivery_date"].nunique(),
                "periods": len(fc),
                "hours": len(eh),
                "mae_15min": e.abs().mean(),
                "rmse_15min": math.sqrt((e**2).mean()),
                "mae_hourly": eh.abs().mean(),
                "rmse_hourly": math.sqrt((eh**2).mean()),
                "mae_within_hour_deviation": dev.abs().mean(),
                "mean_daily_kendall_tau": daily_tau(fc, m).mean(),
            }
        )
    return pd.DataFrame(rows)


def daily_mae(fc: pd.DataFrame, model: str) -> pd.Series:
    return (fc[model] - fc["price_es"]).abs().groupby(fc["delivery_date"]).mean()


def daily_tau(fc: pd.DataFrame, model: str) -> pd.Series:
    return pd.Series(
        {
            day: kendalltau(g[model], g["price_es"]).statistic
            for day, g in fc.groupby("delivery_date", sort=True)
        }
    )


def regret_drivers(fc: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    q = daily[daily["resolution"] == "15min"].pivot(
        index="delivery_date", columns=["duration_h", "model"], values="net_eur_per_mw"
    )
    mae = {m: daily_mae(fc, m) for m in MODELS}
    tau = {m: daily_tau(fc, m) for m in MODELS}
    rows = []
    for hours in battery.DURATIONS_H:
        frames = {
            m: pd.DataFrame(
                {"regret": q[(hours, "perfect")] - q[(hours, m)], "mae": mae[m], "tau": tau[m]}
            )
            for m in MODELS
        }
        for label, models in [(m, [m]) for m in MODELS] + [("pooled", list(MODELS))]:
            f = pd.concat([frames[m] for m in models])
            rows.append(
                {
                    "duration_h": hours,
                    "model": label,
                    "days": len(frames[models[0]]),
                    "observations": len(f),
                    "spearman_regret_mae": spearmanr(f["regret"], f["mae"]).statistic,
                    "spearman_regret_tau": spearmanr(f["regret"], f["tau"]).statistic,
                }
            )
    return pd.DataFrame(rows)


def newey_west_variance(d, lag: int) -> float:
    d = np.asarray(d, dtype=float) - np.mean(d)
    n = len(d)
    v = d @ d / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return float(v)


def dm_test(loss_a, loss_b, lag: int) -> dict:
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    var = newey_west_variance(d, lag)
    stat = float(np.mean(d) / math.sqrt(var / n)) if var > 0 else math.nan
    return {
        "n_days": n,
        "mean_loss_diff": float(np.mean(d)),
        "dm_stat": stat,
        "p_value": float(2 * norm.sf(abs(stat))) if var > 0 else math.nan,
        "newey_west_lag": lag,
    }


def dm_tests(fc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in (("expert", "naive"), ("expert_weather", "expert")):
        la, lb = daily_mae(fc, a), daily_mae(fc, b)
        rows += [{"model_a": a, "model_b": b} | dm_test(la, lb, lag) for lag in DM_LAGS]
    return pd.DataFrame(rows)


def settle(charge, discharge, dt: float, actual) -> dict:
    charge, discharge, actual = (np.asarray(x, dtype=float) for x in (charge, discharge, actual))
    gross = float(np.sum(actual * (discharge - charge)) * dt)
    mwh = float(np.sum(discharge) * dt)
    wear = SPEC.cost_per_mwh * mwh
    return {
        "gross_eur_per_mw": gross,
        "wear_cost_eur_per_mw": wear,
        "net_eur_per_mw": gross - wear,
        "discharged_mwh_per_mw": mwh,
    }


def day_values(day, actual_q, actual_h, plans: dict[str, np.ndarray]) -> list[dict]:
    actual_q, actual_h = np.asarray(actual_q, dtype=float), np.asarray(actual_h, dtype=float)
    rows = []
    for hours in battery.DURATIONS_H:
        energy = battery.POWER_MW * hours
        for model in ("perfect", *plans):
            fq = actual_q if model == "perfect" else np.asarray(plans[model], dtype=float)
            fh = actual_h if model == "perfect" else fq.reshape(-1, 4).mean(axis=1)
            for resolution, plan, actual, dt in (
                ("15min", fq, actual_q, battery.DT_15MIN),
                ("hourly", fh, actual_h, battery.DT_HOURLY),
            ):
                r = battery.solve_day(plan, dt, energy, SPEC)
                rows.append(
                    {"delivery_date": day, "duration_h": hours, "model": model}
                    | {"resolution": resolution}
                    | settle(r.charge, r.discharge, dt, actual)
                )
    return rows


def check_perfect_bound(daily: pd.DataFrame) -> None:
    w = daily.pivot(
        index=["delivery_date", "duration_h", "resolution"],
        columns="model",
        values="net_eur_per_mw",
    )
    excess = w[[m for m in w.columns if m != "perfect"]].max(axis=1) - w["perfect"]
    bad = excess[excess > BOUND_TOL_EUR_PER_MW]
    if len(bad):
        raise BuildError(
            f"{len(bad)} day(s) where a forecast beats perfect foresight, first "
            f"{[(str(d), h, r, round(x, 6)) for (d, h, r), x in bad.head(5).items()]}"
        )


def summarise(daily: pd.DataFrame, models=MODELS) -> pd.DataFrame:
    tot = daily.groupby(["duration_h", "model", "resolution"])[
        ["net_eur_per_mw", "gross_eur_per_mw", "wear_cost_eur_per_mw"]
    ].sum()
    days = daily["delivery_date"].nunique()

    def pct(x, ref):
        return math.nan if math.isclose(ref, 0, abs_tol=1e-6) else 100 * x / ref

    rows = []
    for hours in battery.DURATIONS_H:
        perf = tot.loc[(hours, "perfect")]
        perf_up = perf.loc["15min", "net_eur_per_mw"] - perf.loc["hourly", "net_eur_per_mw"]
        for model in ("perfect", *models):
            t = tot.loc[(hours, model)]
            row = {"duration_h": hours, "model": model, "days": days}
            for label, col in (
                ("net", "net_eur_per_mw"),
                ("gross", "gross_eur_per_mw"),
                ("wear", "wear_cost_eur_per_mw"),
            ):
                for res in ("15min", "hourly"):
                    row[f"{label}_{res}_keur_per_mw"] = t.loc[res, col] / 1000
            for res in ("15min", "hourly"):
                row[f"net_capture_{res}_pct"] = pct(
                    t.loc[res, "net_eur_per_mw"], perf.loc[res, "net_eur_per_mw"]
                )
            up = t.loc["15min", "net_eur_per_mw"] - t.loc["hourly", "net_eur_per_mw"]
            row["net_uplift_keur_per_mw"] = up / 1000
            row["net_uplift_eur_per_mw_day"] = up / days
            row["net_uplift_capture_pct"] = pct(up, perf_up)
            rows.append(row)
    return pd.DataFrame(rows)


def resampled_statistics(daily: pd.DataFrame, idx) -> dict[tuple[int, str, str], np.ndarray]:
    cols = ["duration_h", "model", "resolution"]
    w = daily.pivot(index="delivery_date", columns=cols, values="net_eur_per_mw").sort_index()
    days = idx.shape[1]
    tot = {col: w[col].to_numpy()[idx].sum(axis=1) for col in w.columns}
    out = {}
    for hours in battery.DURATIONS_H:

        def up(m, hours=hours):
            return tot[(hours, m, "15min")] - tot[(hours, m, "hourly")]

        for m in ("perfect", *MODELS):
            for res in ("15min", "hourly"):
                out[(hours, m, f"net_capture_{res}_pct")] = battery.percent(
                    tot[(hours, m, res)], tot[(hours, "perfect", res)]
                )
            out[(hours, m, "net_uplift_eur_per_mw_day")] = up(m) / days
            out[(hours, m, "net_uplift_capture_pct")] = battery.percent(up(m), up("perfect"))
        for a, b in UPLIFT_DIFFS:
            out[(hours, f"{a} minus {b}", "net_uplift_eur_per_mw_day")] = (up(a) - up(b)) / days
    return out


def uncertainty(daily: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    idx = battery.bootstrap_days(daily["delivery_date"].nunique())
    s = summary.set_index(["duration_h", "model"])
    rows = []
    for (hours, model, stat), values in resampled_statistics(daily, idx).items():
        if (hours, model) in s.index:
            estimate = s.loc[(hours, model), stat]
        else:
            a, b = model.split(" minus ")
            estimate = s.loc[(hours, a), stat] - s.loc[(hours, b), stat]
        lo, hi = battery.interval(values)
        rows.append(
            {
                "duration_h": hours,
                "model": model,
                "statistic": stat,
                "days": idx.shape[1],
                "estimate": estimate,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


def window_sensitivity(pw: pd.DataFrame, hourly: dict) -> tuple[pd.DataFrame, list[date]]:
    first = first_eval(max(SENSITIVITY_WINDOWS))
    fcs = {win: forecast_periods(pw, win, first) for win in SENSITIVITY_WINDOWS}
    keys = {(m, win): f"{m} {win}d" for win in SENSITIVITY_WINDOWS for m in SENSITIVITY_MODELS}
    base = fcs[SENSITIVITY_WINDOWS[0]]
    rows = []
    for day, g in base.groupby("delivery_date", sort=True):
        plans = {k: fcs[win].loc[g.index, m].to_numpy() for (m, win), k in keys.items()}
        rows += day_values(day, g["price_es"].to_numpy(), hourly[day], plans)
    daily = pd.DataFrame(rows)
    check_perfect_bound(daily)
    s = summarise(daily, list(keys.values())).set_index(["duration_h", "model"])
    out = []
    for hours in battery.DURATIONS_H:
        for (m, win), k in keys.items():
            r, fc = s.loc[(hours, k)], fcs[win]
            out.append(
                {
                    "duration_h": hours,
                    "model": m,
                    "window_days": win,
                    "days": int(r["days"]),
                    "mae_15min": (fc[m] - fc["price_es"]).abs().mean(),
                    "net_capture_15min_pct": r["net_capture_15min_pct"],
                    "net_capture_hourly_pct": r["net_capture_hourly_pct"],
                    "net_uplift_eur_per_mw_day": r["net_uplift_eur_per_mw_day"],
                    "net_uplift_capture_pct": r["net_uplift_capture_pct"],
                }
            )
    return pd.DataFrame(out), sorted(base["delivery_date"].unique())


def example_day(daily: pd.DataFrame) -> date:
    perfect = daily[(daily["model"] == "perfect") & (daily["duration_h"] == EXAMPLE_DURATION_H)]
    w = perfect.pivot(index="delivery_date", columns="resolution", values="net_eur_per_mw")
    uplift = (w["15min"] - w["hourly"]).rename("uplift").reset_index()
    ranked = uplift.sort_values(["uplift", "delivery_date"]).reset_index(drop=True)
    return ranked.loc[(len(ranked) - 1) // 2, "delivery_date"]


def example_day_table(day: date, fc: pd.DataFrame) -> pd.DataFrame:
    g = fc[fc["delivery_date"] == day].sort_values("period")
    actual = g["price_es"].to_numpy()
    hourly = actual.reshape(-1, 4).mean(axis=1)
    energy = battery.POWER_MW * EXAMPLE_DURATION_H
    fine = battery.solve_day(actual, battery.DT_15MIN, energy, SPEC)
    coarse = battery.solve_day(hourly, battery.DT_HOURLY, energy, SPEC)
    return pd.DataFrame(
        {
            "delivery_start_utc": g["delivery_start_utc"].to_numpy(),
            "price_15min": actual,
            "price_hourly": np.repeat(hourly, 4),
            "forecast_15min": g[EXAMPLE_MODEL].to_numpy(),
            "net_mw_15min": fine.discharge - fine.charge,
            "net_mw_hourly": np.repeat(coarse.discharge - coarse.charge, 4),
        }
    )


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = data_dir / "dataset"
    paths = {
        "q": ds / "prices_15min.parquet",
        "h": ds / "prices_hourly.parquet",
        "pw": ds / "prices_weather_15min.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise BuildError(
            "missing input(s): "
            + ", ".join(missing)
            + "; run `ib15 build` and `ib15 build weather` first"
        )
    q, h, pw = (pd.read_parquet(p) for p in paths.values())
    wx_cols = [f"{prefix}_{loc}" for prefix, _ in VARIABLES.values() for loc in LOCATIONS]
    for name, df, cols in (
        ("q", q, PRICE_COLUMNS),
        ("h", h, PRICE_COLUMNS),
        ("pw", pw, PRICE_COLUMNS + wx_cols),
    ):
        absent = [c for c in cols if c not in df.columns]
        if absent:
            raise BuildError(f"{paths[name]} has no column {', '.join(absent)}")
    key = ["delivery_date", "period"]
    q, h, pw = (df.sort_values(key).reset_index(drop=True) for df in (q, h, pw))
    if q[PRICE_COLUMNS].isna().any().any() or pw[wx_cols].isna().any().any():
        raise BuildError("the prices or the weather have missing values")
    if len(pw) != len(q) or not pw[PRICE_COLUMNS].equals(q[PRICE_COLUMNS]):
        raise BuildError(
            f"{paths['pw']} does not hold the rows of {paths['q']}; run `ib15 build weather` again"
        )
    hq = q.groupby([q["delivery_date"], (q["period"] - 1) // 4])["price_es"].agg(["mean", "size"])
    if (
        len(hq) != len(h)
        or not (hq["size"] == 4).all()
        or not np.allclose(hq["mean"].to_numpy(), h["price_es"].to_numpy(), rtol=0, atol=1e-9)
    ):
        raise BuildError(f"{paths['h']}: an hourly price is not the mean of its four quarter-hours")
    dates = sorted(q["delivery_date"].unique())
    if any(b - a != timedelta(days=1) for a, b in zip(dates, dates[1:], strict=False)):
        raise BuildError("the delivery days are not consecutive")
    need = first_eval(max(SENSITIVITY_WINDOWS)) + 1
    if len(dates) < need:
        raise BuildError(
            f"{len(dates)} day(s) of prices; one evaluation day needs at least {need}"
            f" ({max(SENSITIVITY_WINDOWS)} training days for the longest window, each with lags"
            f" back {max(LAGS)} days)"
        )
    if dates[0].year < HOLIDAY_YEARS[0] or dates[-1].year > HOLIDAY_YEARS[1]:
        raise BuildError(
            f"the days run from {dates[0]} to {dates[-1]}, but the holiday list covers"
            f" {HOLIDAY_YEARS[0]}-{HOLIDAY_YEARS[1]} only; add the BOE list for the missing years"
        )
    return pw, h


def run(data_dir: Path) -> dict:
    started = time.perf_counter()
    pw, h = load_inputs(data_dir)
    fc = forecast_periods(pw)
    forecast_seconds = time.perf_counter() - started
    err, dm = errors(fc), dm_tests(fc)

    solve_started = time.perf_counter()
    hourly = {d: g["price_es"].to_numpy() for d, g in h.groupby("delivery_date")}
    rows = []
    for day, g in fc.groupby("delivery_date", sort=True):
        plans = {m: g[m].to_numpy() for m in MODELS}
        rows += day_values(day, g["price_es"].to_numpy(), hourly[day], plans)
    daily = pd.DataFrame(rows)
    solve_seconds = time.perf_counter() - solve_started
    check_perfect_bound(daily)
    summary = summarise(daily)
    unc = uncertainty(daily, summary)
    drivers = regret_drivers(fc, daily)
    sensitivity, sens_dates = window_sensitivity(pw, hourly)
    chosen = example_day(daily)
    figure = fig_example_day(
        example_day_table(chosen, fc),
        chosen,
        f"{EXAMPLE_MODEL} forecast, 15-minute",
        f"{EXAMPLE_DURATION_H} h battery, perfect foresight, {SPEC.name} specification",
    )

    eval_dates = sorted(fc["delivery_date"].unique())
    status = _git("status", "--porcelain")
    meta = {
        "timing": "forecast for delivery day D uses prices of days up to D-1 and the weather of D"
        " from prices_weather_15min.parquet (issued before 12:00 Madrid on D-1)",
        "slots": "96 local quarter-hours; a missing DST hour takes the preceding hour's values, a"
        " repeated hour is averaged; forecasts are mapped back to the real periods",
        "window_days": WINDOW_DAYS,
        "models": {
            "naive": "on a holiday, same slot on the most recent Sunday before D; otherwise"
            " same slot on D-7 for Monday, Saturday and Sunday, on D-1 for the other weekdays",
            "expert": list(EXPERT_FEATURES),
            "expert_weather": {
                "per_hour": list(EXPERT_FEATURES + WEATHER_FEATURES),
                "shared_by_all_hours": list(SHARED_WEATHER_FEATURES),
            },
        },
        "estimation": "least squares (numpy.linalg.lstsq) per local hour, each pooling the four"
        " quarter-hours of that hour over the window, refitted for every day; in expert_weather"
        " the radiation terms have one coefficient for all 24 hours, fitted jointly with the"
        " per-hour coefficients (Frisch-Waugh-Lovell: partial each hour's regressors out of the"
        " radiation terms and the price, fit the shared coefficients on the pooled residuals,"
        " then refit each hour on the price net of the radiation terms)",
        "radiation_change": "per delivery day, from the national mean radiation of the day's UTC"
        " hours: half of the next hour's mean minus the previous hour's; at the first and last"
        " hour of the day, the one-sided difference within the day",
        "holidays": {
            "rule": "treated as a Sunday: the Sunday dummy is 1 and the Monday and Saturday"
            " dummies 0; the naive forecast copies the most recent Sunday before D",
            "selection": "national holidays held in every autonomous community and in Ceuta and"
            " Melilla",
            "dates": [str(d) for d in HOLIDAYS],
            "sources": [
                "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2024-21316",
                "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2025-21667",
            ],
        },
        "hourly_forecast": "mean of each hour's four quarter-hour forecasts",
        "evaluation": {
            "start": str(eval_dates[0]),
            "end": str(eval_dates[-1]),
            "days": len(eval_dates),
            "first_day_rule": f"day {FIRST_EVAL + 1} of the dataset, the first with a full window"
            " and lags",
        },
        "dm_test": {
            "loss": "daily mean absolute error at 15 minutes",
            "variance": "Newey-West, Bartlett weights",
            "lag_days": DM_LAG,
            "sensitivity_lags_days": list(DM_LAGS),
            "p_value": "two-sided, normal distribution",
        },
        "uncertainty": battery.bootstrap_meta()
        | {
            "days": "the evaluation days",
            "differences": [f"{a} minus {b}" for a, b in UPLIFT_DIFFS],
        },
        "kendall_tau": "per day, scipy.stats.kendalltau (tau-b, which corrects for ties) between"
        " each model's forecast and the actual price over the day's real periods; the errors"
        " file gives its mean over the evaluation days",
        "regret_drivers": "per duration and model, scipy.stats.spearmanr across days between the"
        " daily 15-minute regret (perfect-foresight net minus the model's net, EUR/MW) and the"
        " day's 15-minute MAE, and between the regret and the day's Kendall tau; the pooled row"
        " stacks the three models' days",
        "window_sensitivity": {
            "windows_days": list(SENSITIVITY_WINDOWS),
            "models": list(SENSITIVITY_MODELS),
            "start": str(sens_dates[0]),
            "end": str(sens_dates[-1]),
            "days": len(sens_dates),
            "first_day_rule": f"day {first_eval(max(SENSITIVITY_WINDOWS)) + 1} of the dataset,"
            " the first the longest window can forecast; perfect foresight on the same days",
        },
        "example_day": {
            "date": str(chosen),
            "rule": f"the evaluation day with the median perfect-foresight {EXAMPLE_DURATION_H} h"
            f" net uplift ({SPEC.name} specification, 15-minute minus hourly net value); ties are"
            " ordered by date and, for an even count, the lower of the two middle days is taken",
            "figure": f"top: actual 15-minute prices, their hourly means and the {EXAMPLE_MODEL}"
            " forecast; bottom: the perfect-foresight net power (discharge minus charge) of the"
            " 15-minute and the hourly schedule; x-axis in local Madrid time",
        },
        "battery": {
            "spec": SPEC.name,
            "cost_eur_per_mwh_discharged": SPEC.cost_per_mwh,
            "power_mw": battery.POWER_MW,
            "durations_h": list(battery.DURATIONS_H),
            "round_trip_efficiency": battery.RTE,
            "soc_start_and_end_fraction": battery.SOC_START,
            "settlement": "schedule solved on the forecast, paid the actual prices of its"
            " resolution; net = gross - wear cost of the MWh actually discharged",
        },
        "forecast_seconds": round(forecast_seconds, 1),
        "solve_seconds": round(solve_seconds, 1),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }
    out = data_dir / "results" / "forecast"
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        out / f"fig_example_day.{ext}": lambda t, e=ext: figure.savefig(t, format=e, dpi=DPI)
        for ext in ("png", "pdf")
    }
    battery._write_all(
        figures
        | {
            out / "forecasts_15min.parquet": lambda t: fc.to_parquet(t, index=False),
            out / "forecast_errors.csv": lambda t: err.to_csv(t, index=False),
            out / "forecast_dm_tests.csv": lambda t: dm.to_csv(t, index=False),
            out / "forecast_value_daily.parquet": lambda t: daily.to_parquet(t, index=False),
            out / "forecast_value_summary.csv": lambda t: summary.to_csv(t, index=False),
            out / "forecast_uncertainty.csv": lambda t: unc.to_csv(t, index=False),
            out / "forecast_regret_drivers.csv": lambda t: drivers.to_csv(t, index=False),
            out / "forecast_window_sensitivity.csv": lambda t: sensitivity.to_csv(t, index=False),
            out / "forecast_meta.json": lambda t: t.write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            ),
        }
    )
    return meta
