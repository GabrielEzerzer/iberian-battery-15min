"""Build the price dataset from the raw OMIE files."""

import json
import subprocess
from datetime import timedelta
from pathlib import Path

import pandas as pd

from ib15.omie import (
    FILE_RE,
    GO_LIVE,
    DayFile,
    OmieFormatError,
    expected_periods,
    local_midnight_utc,
    read_day,
)

UNITS = "EUR/MWh"


class BuildError(Exception):
    pass


def load_days(raw_dir: Path) -> list[DayFile]:
    """Read every raw file, validate each one and the day sequence. Raises BuildError listing
    every problem found."""
    if not raw_dir.is_dir():
        raise BuildError(f"raw directory {raw_dir} does not exist")
    paths = sorted(p for p in raw_dir.iterdir() if FILE_RE.match(p.name))
    if not paths:
        raise BuildError(f"no raw OMIE files in {raw_dir}")
    errors, days = [], []
    by_day: dict[str, list[str]] = {}
    for path in paths:
        by_day.setdefault(FILE_RE.match(path.name).group(1), []).append(path.name)
    for names in by_day.values():
        if len(names) > 1:
            errors.append(f"more than one version of one day: {', '.join(names)}")
    first = min(by_day)
    if first > f"{GO_LIVE:%Y%m%d}":
        errors.append(
            f"series must start on {GO_LIVE}, but the first raw file is {by_day[first][0]};"
            " download the missing days"
        )
    for path in paths:
        try:
            days.append(read_day(path))
        except OmieFormatError as e:
            errors.append(str(e))
    days.sort(key=lambda d: d.day)
    for prev, cur in zip(days, days[1:], strict=False):
        if cur.day != prev.day and cur.day - prev.day != timedelta(days=1):
            errors.append(f"gap in day sequence: {prev.day} is followed by {cur.day}")
    if errors:
        raise BuildError("\n".join(errors))
    return days


def quarter_hours(days: list[DayFile]) -> pd.DataFrame:
    """One row per 15-minute delivery period. The UTC start is local midnight (Europe/Madrid)
    in UTC plus (period - 1) * 15 minutes, which stays correct on DST change days."""
    records = []
    for d in days:
        midnight = local_midnight_utc(d.day)
        for period, price_pt, price_es in d.rows:
            start = midnight + timedelta(minutes=15 * (period - 1))
            records.append((start, d.day, period, price_es, price_pt, d.version))
    df = pd.DataFrame(
        records,
        columns=[
            "delivery_start_utc",
            "delivery_date",
            "period",
            "price_es",
            "price_pt",
            "source_version",
        ],
    )
    df["delivery_start_utc"] = pd.to_datetime(df["delivery_start_utc"], utc=True)
    return df


def hourly_benchmark(q: pd.DataFrame) -> pd.DataFrame:
    """Constructed hourly benchmark: each hour is the arithmetic mean of exactly its four
    quarter-hours (consecutive groups of four periods from local midnight); ``period`` is the
    1-based hour index. This is built from the 15-minute prices, not OMIE's own hourly product.
    """
    h = q.assign(period=(q["period"] - 1) // 4 + 1)
    g = h.groupby(["delivery_date", "period"], sort=True)
    if not (g.size() == 4).all():
        raise BuildError("an hour does not have exactly four quarter-hours")
    out = g.agg(
        delivery_start_utc=("delivery_start_utc", "min"),
        price_es=("price_es", "mean"),
        price_pt=("price_pt", "mean"),
        source_version=("source_version", "first"),
    ).reset_index()
    return out[q.columns.tolist()]


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def coverage(days: list[DayFile], q: pd.DataFrame) -> dict:
    # Validation already rejects missing or duplicate periods, so these counts are 0 in any
    # dataset that was written; they are recorded so the file states it.
    status = _git("status", "--porcelain")
    return {
        "date_range": {"start": days[0].day.isoformat(), "end": days[-1].day.isoformat()},
        "days": len(days),
        "periods_15min": len(q),
        "hours": len(q) // 4,
        "days_not_96_periods": {
            d.day.isoformat(): {"expected": n, "actual": len(d.rows)}
            for d in days
            if (n := expected_periods(d.day)) != 96
        },
        "missing_periods": 0,
        "duplicate_periods": 0,
        "days_not_version_1": {d.day.isoformat(): d.version for d in days if d.version != 1},
        "price_es": {
            "min": float(q["price_es"].min()),
            "max": float(q["price_es"].max()),
            "mean": float(q["price_es"].mean()),
        },
        "units": UNITS,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def build(data_dir: Path) -> dict:
    """Validate the raw files and write the dataset. Nothing is written if validation fails."""
    days = load_days(data_dir / "raw" / "omie")
    q = quarter_hours(days)
    h = hourly_benchmark(q)
    cov = coverage(days, q)
    out = data_dir / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    q.to_parquet(out / "prices_15min.parquet", index=False)
    h.to_parquet(out / "prices_hourly.parquet", index=False)
    (out / "coverage.json").write_text(json.dumps(cov, indent=2) + "\n", encoding="utf-8")
    return cov
