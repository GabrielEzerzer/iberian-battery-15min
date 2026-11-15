# Exploratory tables and figures for the paper: the prices, the weather and where the 15-minute
# uplift comes from.
import json
import math
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ib15.battery import DURATIONS_H, _write_all
from ib15.dataset import BuildError, _git, hourly_benchmark
from ib15.omie import MADRID
from ib15.weather import LOCATIONS, VARIABLES

FLAT_TOL_EUR_PER_MWH = 0.005
ZERO_UPLIFT_TOL_EUR_PER_MW = 0.01
SENSITIVITY_COLUMNS = (
    "duration_h",
    "cost_eur_per_mwh",
    "net_uplift_pct",
    "net_uplift_eur_per_mw_day",
)
SEASONS = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}
SEASON_ORDER = ("winter", "spring", "summer", "autumn")
SPREAD_WINDOWS = {
    "oct_nov_2025": (date(2025, 10, 1), date(2025, 11, 30)),
    "full": (None, None),
}
WEATHER_LABELS = {
    "temperature_2m": "temperature (°C)",
    "wind_speed_100m": "wind speed at 100 m (m/s)",
    "shortwave_radiation": "solar radiation (W/m²)",
}

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DPI = 300
FIGURE_WIDTH_INCHES = 6.5


def hour_table(q: pd.DataFrame) -> pd.DataFrame:
    q = q.sort_values(["delivery_date", "period"])
    h = q.assign(hour=(q["period"] - 1) // 4, dev=quarter_deviation(q).abs())
    g = h.groupby(["delivery_date", "hour"], sort=True)
    out = g.agg(
        start_utc=("delivery_start_utc", "min"),
        n=("price_es", "size"),
        mean=("price_es", "mean"),
        hi=("price_es", "max"),
        lo=("price_es", "min"),
        first=("price_es", "first"),
        last=("price_es", "last"),
        abs_dev=("dev", "mean"),
    ).reset_index()
    if not (out["n"] == 4).all():
        raise BuildError("an hour does not have exactly four quarter-hours")
    out["spread"] = out["hi"] - out["lo"]
    out["slope"] = out["last"] - out["first"]
    return out[["delivery_date", "hour", "start_utc", "mean", "spread", "slope", "abs_dev"]]


def quarter_deviation(q: pd.DataFrame) -> pd.Series:
    hour = (q["period"] - 1) // 4
    return q["price_es"] - q.groupby([q["delivery_date"], hour])["price_es"].transform("mean")


def spread_summary(hours: pd.DataFrame) -> dict:
    s = hours["spread"]
    return {
        "hours": len(s),
        "mean_spread": s.mean(),
        "median_spread": s.median(),
        "p90_spread": s.quantile(0.9),
        "share_flat_hours": (s < FLAT_TOL_EUR_PER_MWH).mean(),
        "mean_abs_quarter_deviation": hours["abs_dev"].mean(),
    }


def price_stats(prices: pd.Series, days: int) -> dict:
    p = prices
    return {
        "days": days,
        "periods": len(p),
        "mean": p.mean(),
        "median": p.median(),
        "sd": p.std(),
        "min": p.min(),
        "max": p.max(),
        "p5": p.quantile(0.05),
        "p95": p.quantile(0.95),
        "share_le_0": (p <= 0).mean(),
        "share_lt_0": (p < 0).mean(),
    }


def concentration(uplift) -> dict:
    u = np.sort(np.asarray(uplift, dtype=float))[::-1]
    total = u.sum()
    out = {}
    for frac in (0.10, 0.25):
        k = math.ceil(frac * len(u) - 1e-9)
        out[f"share_from_top_{round(frac * 100)}pct_days"] = u[:k].sum() / total
    out["share_days_zero_uplift"] = float(np.mean(np.abs(u) < ZERO_UPLIFT_TOL_EUR_PER_MW))
    out["median_daily_uplift_eur_per_mw"] = float(np.median(u))
    out["mean_daily_uplift_eur_per_mw"] = float(np.mean(u))
    return out


def autocorr(s: pd.Series, lag: pd.Timedelta) -> tuple[float, int]:
    lagged = s.reindex(s.index - lag).to_numpy()
    x = pd.Series(s.to_numpy())
    y = pd.Series(lagged)
    return float(x.corr(y)), int((x.notna() & y.notna()).sum())


def national_mean(weather: pd.DataFrame) -> pd.DataFrame:
    counts = weather.groupby("valid_time_utc")["location"].agg(["nunique", "size"])
    if (
        set(weather["location"]) != set(LOCATIONS)
        or not ((counts["nunique"] == len(LOCATIONS)) & (counts["size"] == len(LOCATIONS))).all()
    ):
        raise BuildError(f"weather does not have each of the {len(LOCATIONS)} locations every hour")
    return weather.groupby("valid_time_utc")[list(VARIABLES)].mean().sort_index()


def daily_uplift(pf: pd.DataFrame) -> pd.DataFrame:
    m = pf[pf["spec"] == "main"]
    w = m.pivot_table(
        index=["delivery_date", "duration_h"], columns="resolution", values="net_eur_per_mw"
    ).reset_index()
    if w[["15min", "hourly"]].isna().any().any():
        raise BuildError("perfect_foresight_daily: a main day lacks one of the resolutions")
    w = w.rename(columns={"15min": "net_15min", "hourly": "net_hourly"})
    w["uplift"] = w["net_15min"] - w["net_hourly"]
    return w[["delivery_date", "duration_h", "net_15min", "net_hourly", "uplift"]]


def _month(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates.astype(str)).dt.strftime("%Y-%m")


def price_summary(q: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    days = q["delivery_date"].nunique()
    rows = [
        {"series": "15min", **price_stats(q["price_es"], days)},
        {"series": "hourly", **price_stats(h["price_es"], h["delivery_date"].nunique())},
    ]
    return pd.DataFrame(rows)


def within_hour_table(hours: pd.DataFrame) -> pd.DataFrame:
    rows = [{"month": "all", **spread_summary(hours)}]
    for month, g in hours.groupby(_month(hours["delivery_date"]), sort=True):
        rows.append({"month": month, **spread_summary(g)})
    return pd.DataFrame(rows)


def daily_table(q: pd.DataFrame, hours: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    d = q.groupby("delivery_date")["price_es"].agg(["mean", "max", "min"])
    out = pd.DataFrame(
        {
            "mean_price": d["mean"],
            "price_range": d["max"] - d["min"],
            "mean_spread": hours.groupby("delivery_date")["spread"].mean(),
        }
    )
    w = wx.reindex(hours["start_utc"]).set_axis(hours.index)
    w["delivery_date"] = hours["delivery_date"]
    out = out.join(w.groupby("delivery_date")[list(VARIABLES)].mean())
    return out.reset_index()


def uplift_by_month(up: pd.DataFrame, q: pd.DataFrame, hours: pd.DataFrame) -> pd.DataFrame:
    up = up.assign(month=_month(up["delivery_date"]))
    hm = pd.DataFrame(
        {
            "mean_spread": hours.groupby(_month(hours["delivery_date"]))["spread"].mean(),
            "mean_price": q.groupby(_month(q["delivery_date"]))["price_es"].mean(),
        }
    )
    rows = []
    for (month, hours_), g in up.groupby(["month", "duration_h"], sort=True):
        n15, nh = g["net_15min"].sum(), g["net_hourly"].sum()
        rows.append(
            {
                "month": month,
                "duration_h": hours_,
                "days": len(g),
                "net_15min_eur_per_mw": n15,
                "net_hourly_eur_per_mw": nh,
                "uplift_eur_per_mw": n15 - nh,
                "uplift_eur_per_mw_day": (n15 - nh) / len(g),
                "uplift_pct": 100 * (n15 - nh) / nh,
                "mean_within_hour_spread": hm.loc[month, "mean_spread"],
                "mean_price": hm.loc[month, "mean_price"],
            }
        )
    return pd.DataFrame(rows)


def uplift_concentration(up: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.set_index("delivery_date")
    rows = []
    for hours_, g in up.groupby("duration_h", sort=True):
        g = g.set_index("delivery_date")
        rows.append(
            {
                "duration_h": hours_,
                "days": len(g),
                **concentration(g["uplift"]),
                "corr_uplift_mean_spread": g["uplift"].corr(d["mean_spread"].reindex(g.index)),
                "corr_uplift_price_range": g["uplift"].corr(d["price_range"].reindex(g.index)),
            }
        )
    return pd.DataFrame(rows)


def one_cycle_spread(q: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly_benchmark(q)
    rows = []
    for market in ("es", "pt"):
        col = f"price_{market}"
        days = pd.DataFrame(
            {
                "s15": q.groupby("delivery_date")[col].agg(
                    lambda p: np.sort(p)[-4:].mean() - np.sort(p)[:4].mean()
                ),
                "sh": hourly.groupby("delivery_date")[col].agg(lambda p: p.max() - p.min()),
            }
        )
        for window, (start, end) in SPREAD_WINDOWS.items():
            start = days.index.min() if start is None else start
            end = days.index.max() if end is None else end
            w = days[(days.index >= start) & (days.index <= end)]
            rows.append(
                {
                    "market": market,
                    "window": window,
                    "start": w.index.min(),
                    "end": w.index.max(),
                    "days": len(w),
                    "spread_15min_eur_per_mwh": w["s15"].mean(),
                    "spread_hourly_eur_per_mwh": w["sh"].mean(),
                    "uplift_pct": 100 * (w["s15"].sum() / w["sh"].sum() - 1),
                }
            )
    return pd.DataFrame(rows)


def weather_summary(weather: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for loc in LOCATIONS:
        g = weather[weather["location"] == loc]
        for var in VARIABLES:
            v = g[var]
            rows.append(
                {
                    "location": loc,
                    "variable": var,
                    "unit": VARIABLES[var][1],
                    "hours": len(v),
                    "mean": v.mean(),
                    "sd": v.std(),
                    "min": v.min(),
                    "max": v.max(),
                }
            )
    return pd.DataFrame(rows)


def forecast_motivation(
    q: pd.DataFrame, hours: pd.DataFrame, daily: pd.DataFrame, wx: pd.DataFrame
) -> pd.DataFrame:
    rows = []

    def add(quantity, against, r, n, description):
        rows.append(
            {
                "quantity": quantity,
                "against": against,
                "pearson_r": r,
                "n": n,
                "description": description,
            }
        )

    s = q.set_index("delivery_start_utc").sort_index()
    price = s["price_es"]
    dev = quarter_deviation(q).set_axis(q["delivery_start_utc"]).sort_index()
    for name, series in (("price_15min", price), ("within_hour_deviation", dev)):
        for days in (1, 7):
            r, n = autocorr(series, pd.Timedelta(days=days))
            add(
                name, f"own value {days} day(s) earlier", r, n, "autocorrelation, same quarter-hour"
            )

    def corr(a: pd.Series, b: pd.Series):
        ok = a.notna() & b.notna()
        return float(a[ok].corr(b[ok])), int(ok.sum())

    qwx = wx.reindex(price.index.floor("h"))
    hwx = wx.reindex(hours["start_utc"])
    rad = "shortwave_radiation"
    nxt, prv = (wx.shift(n, freq="h").reindex(wx.index) for n in (-1, 1))
    change = nxt - wx
    change[rad] = (nxt[rad] - prv[rad]) / 2
    change = change.reindex(hours["start_utc"])
    for var in VARIABLES:
        add(
            "daily_mean_price",
            f"national mean {var}, daily mean",
            *corr(daily["mean_price"], daily[var]),
            "per delivery day",
        )
        add(
            "price_15min",
            f"national mean {var}, containing UTC hour",
            *corr(price.reset_index(drop=True), qwx[var].reset_index(drop=True)),
            "per quarter-hour",
        )
        add(
            "within_hour_spread",
            f"national mean {var}, same UTC hour",
            *corr(hours["spread"], hwx[var].reset_index(drop=True)),
            "per hour; spread = max - min of the four quarters",
        )
        if var == rad:
            against = f"change in national mean {var} per hour, centred on the UTC hour"
            how = (
                "per hour; slope = Q4 - Q1; radiation is the mean over its hour, so its change is"
                " (next hour - previous hour) / 2, centred on the middle of the hour like the slope"
            )
        else:
            against = f"change in national mean {var} to the next UTC hour"
            how = "per hour; slope = Q4 - Q1"
        add(
            "within_hour_slope",
            against,
            *corr(hours["slope"], change[var].reset_index(drop=True)),
            how,
        )
    return pd.DataFrame(rows)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "lines.linewidth": 1.4,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def fig_daily(daily: pd.DataFrame) -> plt.Figure:
    x = pd.to_datetime(daily["delivery_date"].astype(str))
    fig, (a, b) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH_INCHES, 4), sharex=True)
    a.plot(x, daily["mean_price"], color=SERIES[0])
    a.set_ylabel("EUR/MWh")
    a.set_title("Daily mean Spanish day-ahead price (15-minute)")
    b.plot(x, daily["mean_spread"], color=SERIES[0])
    b.set_ylabel("EUR/MWh")
    b.set_title("Daily mean within-hour spread (max − min of the four quarter-hours)")
    b.set_ylim(bottom=0)
    fig.autofmt_xdate()
    return fig


def fig_profile(q: pd.DataFrame) -> plt.Figure:
    lt = q["delivery_start_utc"].dt.tz_convert(MADRID)
    dates = pd.to_datetime(q["delivery_date"].astype(str))
    df = pd.DataFrame(
        {
            "season": dates.dt.month.map(SEASONS),
            "slot": lt.dt.hour * 4 + lt.dt.minute // 15,
            "price": q["price_es"],
        }
    )
    prof = df.groupby(["season", "slot"])["price"].mean().unstack("season")
    seasons = [s for s in SEASON_ORDER if s in prof.columns]
    hourly = prof.groupby(prof.index // 4).transform("mean")
    shape = (prof - hourly).abs().mean()
    season_with_most_shape = shape[seasons].idxmax()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, 3.4))
    for s in seasons:
        color = SERIES[SEASON_ORDER.index(s)]
        ax.plot(prof.index / 4 + 0.125, prof[s], color=color, label=f"{s}, 15-minute")
    steps = prof[season_with_most_shape].groupby(prof.index // 4).mean()
    ax.stairs(
        steps.to_numpy(),
        np.arange(len(steps) + 1),
        color=INK_2,
        linewidth=1.0,
        label=f"{season_with_most_shape}, hourly benchmark",
    )
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hour of day, local time (Madrid)")
    ax.set_ylabel("mean price (EUR/MWh)")
    ax.set_title(
        "Average price by quarter-hour of day and season, all years pooled"
        f" ({dates.min():%b %Y} – {dates.max():%b %Y})"
    )
    ax.legend(ncol=2, fontsize=7)
    return fig


def fig_quarter_deviation(q: pd.DataFrame) -> plt.Figure:
    lt = q["delivery_start_utc"].dt.tz_convert(MADRID)
    df = pd.DataFrame(
        {
            "hour": lt.dt.hour,
            "quarter": (q["period"] - 1) % 4 + 1,
            "dev": quarter_deviation(q),
        }
    )
    m = df.groupby(["hour", "quarter"])["dev"].mean().unstack("quarter")
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, 3.2))
    ax.axhline(0, color=AXIS, linewidth=0.8)
    for k in m.columns:
        ax.plot(m.index, m[k], color=SERIES[k - 1], marker="o", markersize=3, label=f"Q{k}")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_xlabel("hour of day, local time (Madrid)")
    ax.set_ylabel("quarter price − hourly mean (EUR/MWh)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return fig


def fig_spread_hist(hours: pd.DataFrame) -> plt.Figure:
    s = hours["spread"]
    cut = s.quantile(0.99)
    beyond = (s > cut).mean()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, 3))
    ax.hist(s[s <= cut], bins=60, color=SERIES[0], edgecolor="white", linewidth=0.4)
    ax.set_xlim(0, cut)
    ax.set_xlabel("within-hour spread, max − min of the four quarter-hours (EUR/MWh)")
    ax.set_ylabel("hours")
    ax.set_title("Distribution of the within-hour spread, all hours")
    ax.text(
        0.99,
        0.95,
        f"x axis cut at the 99th percentile ({cut:.1f} EUR/MWh);\n"
        f"{100 * beyond:.1f}% of hours lie beyond it",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK_2,
        fontsize=7,
    )
    return fig


def fig_uplift_month(by_month: pd.DataFrame) -> plt.Figure:
    months = sorted(by_month["month"].unique())
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, 3.2))
    for i, d in enumerate(DURATIONS_H):
        g = by_month[by_month["duration_h"] == d].set_index("month").reindex(months)
        ax.plot(
            range(len(months)),
            g["uplift_eur_per_mw_day"],
            color=SERIES[i],
            marker="o",
            markersize=3,
            label=f"{d} h",
        )
    days = by_month.groupby("month")["days"].max()
    labels = [m if days[m] == pd.Period(m).days_in_month else f"{m} ({days[m]} d)" for m in months]
    ax.set_xticks(range(len(months)), labels, rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("uplift (EUR/MW per day)")
    ax.legend(title="duration")
    return fig


def fig_uplift_scatter(up: pd.DataFrame, daily: pd.DataFrame) -> plt.Figure:
    d = daily.set_index("delivery_date")["mean_spread"]
    fig, axes = plt.subplots(
        1, len(DURATIONS_H), figsize=(FIGURE_WIDTH_INCHES, 2.8), sharex=True, sharey=True
    )
    for i, (ax, dur) in enumerate(zip(axes, DURATIONS_H, strict=True)):
        g = up[up["duration_h"] == dur]
        ax.scatter(
            d.reindex(g["delivery_date"]).to_numpy(),
            g["uplift"],
            s=6,
            color=SERIES[i],
            alpha=0.5,
            linewidths=0,
        )
        ax.set_title(f"{dur} h battery")
    axes[0].set_ylabel("daily uplift (EUR/MW)")
    fig.supxlabel("day's mean within-hour spread (EUR/MWh)", fontsize=8)
    return fig


def fig_sensitivity(s: pd.DataFrame) -> plt.Figure:
    fig, (a, b) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_INCHES, 2.8), sharex=True)
    for i, d in enumerate(DURATIONS_H):
        g = s[s["duration_h"] == d].sort_values("cost_eur_per_mwh")
        kw = {"color": SERIES[i], "marker": "o", "markersize": 3, "label": f"{d} h"}
        a.plot(g["cost_eur_per_mwh"], g["net_uplift_pct"], **kw)
        b.plot(g["cost_eur_per_mwh"], g["net_uplift_eur_per_mw_day"], **kw)
    a.set_ylabel("uplift (%)")
    a.set_title("15-minute uplift, percent of hourly")
    b.set_ylabel("uplift (EUR/MW per day)")
    b.set_title("15-minute uplift per day")
    for ax in (a, b):
        ax.set_xlabel("wear cost (EUR per MWh discharged)")
        ax.set_ylim(bottom=0)
    b.legend(title="duration", loc="lower left")
    return fig


def fig_price_weather(q: pd.DataFrame, wx: pd.DataFrame) -> plt.Figure:
    w = wx.reindex(q["delivery_start_utc"].dt.floor("h"))
    fig, axes = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH_INCHES, 2.9), sharey=True, layout="constrained"
    )
    bins = []
    for ax, var in zip(axes, ("shortwave_radiation", "wind_speed_100m"), strict=True):
        hb = ax.hexbin(
            w[var].to_numpy(),
            q["price_es"].to_numpy(),
            gridsize=40,
            bins="log",
            cmap="Blues",
            mincnt=1,
            linewidths=0,
        )
        ax.set_xlabel(f"national mean {WEATHER_LABELS[var]}")
        ax.grid(False)
        bins.append(hb)
    shared_count_scale = matplotlib.colors.LogNorm(1, max(b.get_array().max() for b in bins))
    for b in bins:
        b.set_norm(shared_count_scale)
    fig.colorbar(bins[0], ax=axes, label="quarter-hours (log scale)")
    axes[0].set_ylabel("15-minute price (EUR/MWh)")
    fig.suptitle("Price against the forecast weather of its hour (descriptive)")
    return fig


def _input(path: Path, time_col: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(path)
    t = df[time_col]
    rec = {"path": str(path), "rows": len(df), "start": str(t.min()), "end": str(t.max())}
    return df, rec


def run(data_dir: Path) -> dict:
    started = time.perf_counter()
    ds, res = data_dir / "dataset", data_dir / "results"
    required = {
        "prices_15min": (ds / "prices_15min.parquet", "delivery_date"),
        "prices_hourly": (ds / "prices_hourly.parquet", "delivery_date"),
        "weather_hourly": (ds / "weather_hourly.parquet", "valid_time_utc"),
        "perfect_foresight_daily": (res / "perfect_foresight_daily.parquet", "delivery_date"),
    }
    missing = [str(p) for p, _ in required.values() if not p.exists()]
    if missing:
        raise BuildError(
            "missing input(s): " + ", ".join(missing) + "; run `ib15 build`, `ib15 build weather`"
            " and `ib15 perfect-foresight` first"
        )
    frames, inputs = {}, {}
    for key, (path, col) in required.items():
        frames[key], inputs[key] = _input(path, col)
    q, h = frames["prices_15min"], frames["prices_hourly"]
    weather, pf = frames["weather_hourly"], frames["perfect_foresight_daily"]
    sens_path = res / "perfect_foresight_cost_sensitivity.csv"
    sens = pd.read_csv(sens_path) if sens_path.exists() else None
    if sens is not None:
        absent = [c for c in SENSITIVITY_COLUMNS if c not in sens.columns]
        if absent:
            raise BuildError(
                f"{sens_path} has no column {', '.join(absent)}; run `ib15 perfect-foresight`"
                " again to rewrite it"
            )
        inputs["perfect_foresight_cost_sensitivity"] = {"path": str(sens_path), "rows": len(sens)}

    hours = hour_table(q)
    wx = national_mean(weather)
    missing_wx = ~hours["start_utc"].isin(wx.index)
    if missing_wx.any():
        raise BuildError(f"weather is missing {int(missing_wx.sum())} price hour(s)")
    daily = daily_table(q, hours, wx)
    up = daily_uplift(pf)

    tables = {
        "price_summary.csv": price_summary(q, h),
        "within_hour_spread.csv": within_hour_table(hours),
        "uplift_by_month.csv": uplift_by_month(up, q, hours),
        "uplift_concentration.csv": uplift_concentration(up, daily),
        "one_cycle_spread.csv": one_cycle_spread(q),
        "weather_summary.csv": weather_summary(weather),
        "forecast_motivation.csv": forecast_motivation(q, hours, daily, wx),
    }
    _style()
    figures = {
        "fig_a_daily_price_and_spread": fig_daily(daily),
        "fig_b_profile_by_season": fig_profile(q),
        "fig_c_quarter_deviation_by_hour": fig_quarter_deviation(q),
        "fig_d_spread_distribution": fig_spread_hist(hours),
        "fig_e_uplift_by_month": fig_uplift_month(tables["uplift_by_month.csv"]),
        "fig_f_uplift_vs_spread": fig_uplift_scatter(up, daily),
    }
    skipped = []
    if sens is not None:
        figures["fig_g_cost_sensitivity"] = fig_sensitivity(sens)
    else:
        skipped.append(f"fig_g_cost_sensitivity: {sens_path} does not exist")
    figures["fig_h_price_vs_weather"] = fig_price_weather(q, wx)

    out = res / "eda"
    files = {
        out / name: (lambda t, df=df: df.round(4).to_csv(t, index=False))
        for name, df in tables.items()
    }
    for name, fig in figures.items():
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        for ext in ("png", "pdf"):
            files[out / f"{name}.{ext}"] = lambda t, f=fig, e=ext: f.savefig(t, format=e, dpi=DPI)
    status = _git("status", "--porcelain")
    meta = {
        "inputs": inputs,
        "outputs": [p.name for p in files] + ["eda_meta.json"],
        "skipped": skipped,
        "seconds": None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }

    def write_meta(t: Path) -> None:
        meta["seconds"] = round(time.perf_counter() - started, 1)
        t.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    files[out / "eda_meta.json"] = write_meta
    out.mkdir(parents=True, exist_ok=True)
    try:
        _write_all(files)
    finally:
        for fig in figures.values():
            plt.close(fig)
    if sens is None:
        for p in out.glob("fig_g_*"):
            p.unlink()
    return meta
