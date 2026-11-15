# Archived weather forecasts from Open-Meteo's Previous Runs API, as issued before the bid.
# The evidence for the bid-time rule and the radiation label is in the README, "Method notes".
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlencode

import pandas as pd

from ib15.dataset import BuildError, _git
from ib15.omie import MADRID, http_get

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs025"
LEAD = "previous_day2"
LEAD_DAYS = 2
PUBLISHED_AFTER = timedelta(hours=8)


class Variable(NamedTuple):
    column_prefix: str
    unit: str


VARIABLES = {
    "temperature_2m": Variable("temp", "°C"),
    "wind_speed_100m": Variable("wind100", "m/s"),
    "shortwave_radiation": Variable("ghi", "W/m²"),
}
LIMITS = {
    "temperature_2m": (-40.0, 60.0),
    "wind_speed_100m": (0.0, 100.0),
    "shortwave_radiation": (0.0, 1500.0),
}
LOCATIONS = {
    "madrid": (40.42, -3.70),
    "barcelona": (41.39, 2.17),
    "valencia": (39.47, -0.38),
    "sevilla": (37.39, -5.98),
    "zaragoza": (41.65, -0.88),
    "burgos": (42.34, -3.70),
    "lugo": (43.01, -7.56),
    "badajoz": (38.88, -6.97),
    "ciudad_real": (38.98, -3.93),
    "albacete": (38.99, -1.86),
}
MAX_GRID_OFFSET_DEG = 0.25
SETTLED_AFTER = timedelta(days=2)
FILE_RE = re.compile(
    r"^(?P<loc>[a-z_]+)\.(?P<model>[a-z0-9_]+)\.(?P<start>\d{4}-\d{2}-\d{2})"
    r"\.(?P<end>\d{4}-\d{2}-\d{2})\.json$"
)


def file_name(location: str, start: date, end: date) -> str:
    return f"{location}.{MODEL}.{start}.{end}.json"


def request_url(lat: float, lon: float, start: date, end: date) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(f"{v}_{LEAD}" for v in VARIABLES),
        "models": MODEL,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "GMT",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
    }
    return f"{URL}?{urlencode(params, safe=',')}"


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks, cur = [], start
    while cur <= end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        chunks.append((cur, min(end, nxt - timedelta(days=1))))
        cur = nxt
    return chunks


def _floor(t: datetime, hours: int) -> datetime:
    return t - timedelta(seconds=t.timestamp() % (hours * 3600))


def latest_run(label: datetime, lead_days: int = LEAD_DAYS) -> datetime:
    grid = _floor(label, 3)
    point = label if label == grid else grid + timedelta(hours=6)
    return _floor(point - timedelta(days=lead_days), 6)


def bid_deadline(day: date) -> datetime:
    d = day - timedelta(days=1)
    return datetime(d.year, d.month, d.day, 12, tzinfo=MADRID).astimezone(UTC)


def hour_run(hour: datetime, lead_days: int = LEAD_DAYS) -> datetime:
    return max(latest_run(hour, lead_days), latest_run(hour + timedelta(hours=1), lead_days))


def issued_before_bid(hour: datetime, day: date, lead_days: int = LEAD_DAYS) -> bool:
    return hour_run(hour, lead_days) + PUBLISHED_AFTER < bid_deadline(day)


class WeatherFormatError(ValueError):
    pass


def _number(x) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def parse_response(
    name: str, body: bytes, lat: float, lon: float, start: date, end: date
) -> tuple[pd.DatetimeIndex, dict[str, list[float | None]]]:

    def fail(msg: str):
        raise WeatherFormatError(f"{name}: {msg}")

    try:
        d = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        fail(f"not JSON ({e})")
    if not isinstance(d, dict):
        fail(f"expected one JSON object, got {type(d).__name__}")
    if d.get("error"):
        fail(f"API error: {d.get('reason')!r}")
    for key in ("latitude", "longitude", "utc_offset_seconds", "hourly_units", "hourly"):
        if key not in d:
            fail(f"missing key {key!r}")
    if d["utc_offset_seconds"] != 0:
        fail(f"utc_offset_seconds is {d['utc_offset_seconds']}, expected 0 (GMT)")
    if not (_number(d["latitude"]) and _number(d["longitude"])) or (
        abs(d["latitude"] - lat) > MAX_GRID_OFFSET_DEG
        or abs(d["longitude"] - lon) > MAX_GRID_OFFSET_DEG
    ):
        fail(f"grid cell {d['latitude']},{d['longitude']} is not near {lat},{lon}")
    hourly, units = d["hourly"], d["hourly_units"]
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        fail("hourly or hourly_units is not an object")
    want = {f"{v}_{LEAD}" for v in VARIABLES}
    if "time" not in hourly:
        fail("missing hourly time array")
    if missing := sorted(want - set(hourly)):
        fail(f"missing variable(s) {missing}")
    if extra := sorted(set(hourly) - want - {"time"}):
        fail(f"unexpected variable(s) {extra}")
    times = hourly["time"]
    first = datetime(start.year, start.month, start.day)
    n = ((end - start).days + 1) * 24
    expected = [f"{first + timedelta(hours=i):%Y-%m-%dT%H:%M}" for i in range(n)]
    if times != expected:
        got = f"{len(times)} times" if isinstance(times, list) else type(times).__name__
        fail(f"hourly time is not every hour of {start}..{end} in GMT ({got}, expected {n})")
    values = {}
    for var, (_, unit) in VARIABLES.items():
        key = f"{var}_{LEAD}"
        if units.get(key) != unit:
            fail(f"{key} unit is {units.get(key)!r}, expected {unit!r}")
        vals = hourly[key]
        if not isinstance(vals, list) or len(vals) != n:
            got = len(vals) if isinstance(vals, list) else type(vals).__name__
            fail(f"{key} has {got} values, expected {n}")
        if bad := [x for x in vals if x is not None and not _number(x)]:
            fail(f"{key} has non-numeric values {bad[:3]}")
        lo, hi = LIMITS[var]
        if bad := [x for x in vals if x is not None and not lo <= x <= hi]:
            fail(f"{key} has {len(bad)} value(s) outside {lo}..{hi} {unit}, first {bad[:3]}")
        values[var] = vals
    return pd.to_datetime(times).tz_localize(UTC), values


class WeatherDownloadError(Exception):
    pass


@dataclass
class DownloadResult:
    saved: list[str]
    skipped: list[str]
    errors: list[tuple[str, str]]


def download(
    raw_dir: Path,
    start: date,
    end: date,
    *,
    fetch=None,
    today: date | None = None,
    pause: float = 1.0,
) -> DownloadResult:
    fetch = fetch or http_get
    today = today or datetime.now(UTC).date()
    raw_dir.mkdir(parents=True, exist_ok=True)
    result = DownloadResult([], [], [])
    requests = 0
    for loc, (lat, lon) in LOCATIONS.items():
        for cs, ce in month_chunks(start, end):
            name = file_name(loc, cs, ce)
            if ce <= today - SETTLED_AFTER and _covered(raw_dir, loc, lat, lon, cs, ce):
                result.skipped.append(name)
                continue
            if requests:
                time.sleep(pause)
            requests += 1
            try:
                try:
                    body = fetch(request_url(lat, lon, cs, ce))
                except Exception as e:
                    raise WeatherDownloadError(f"{type(e).__name__}: {e}") from e
                if body is None:
                    raise WeatherDownloadError("HTTP 404")
                parse_response(name, body, lat, lon, cs, ce)
            except (WeatherDownloadError, WeatherFormatError) as e:
                msg = str(e).removeprefix(f"{name}: ")
                result.errors.append((name, msg))
                print(f"{name} ERROR {msg}")
                continue
            tmp = raw_dir / (name + ".part")
            tmp.write_bytes(body)
            tmp.replace(raw_dir / name)
            for p in raw_dir.iterdir():
                m = FILE_RE.match(p.name)
                if (
                    m
                    and p.name != name
                    and m["loc"] == loc
                    and m["model"] == MODEL
                    and m["start"] >= f"{cs}"
                    and m["end"] <= f"{ce}"
                ):
                    p.unlink()
            result.saved.append(name)
            print(f"{name} saved")
    return result


def _covered(raw_dir: Path, loc: str, lat: float, lon: float, start: date, end: date) -> bool:
    for p in raw_dir.iterdir():
        m = FILE_RE.match(p.name)
        if not (
            m
            and m["loc"] == loc
            and m["model"] == MODEL
            and m["start"] <= f"{start}"
            and m["end"] >= f"{end}"
        ):
            continue
        try:
            own = date.fromisoformat(m["start"]), date.fromisoformat(m["end"])
        except ValueError:
            continue
        if _complete(p, lat, lon, *own):
            return True
    return False


def _complete(path: Path, lat: float, lon: float, start: date, end: date) -> bool:
    if not path.exists():
        return False
    try:
        _, values = parse_response(path.name, path.read_bytes(), lat, lon, start, end)
    except WeatherFormatError:
        return False
    return all(None not in vals for vals in values.values())


def load_hourly(raw_dir: Path) -> pd.DataFrame:
    if not raw_dir.is_dir():
        raise BuildError(f"raw directory {raw_dir} does not exist; run `ib15 download weather`")
    errors, frames = [], {loc: [] for loc in LOCATIONS}
    for path in sorted(raw_dir.iterdir()):
        if path.suffix != ".json":
            continue
        m = FILE_RE.match(path.name)
        if not m:
            errors.append(f"{path.name}: unexpected file name")
            continue
        if m["model"] != MODEL:
            errors.append(f"{path.name}: unexpected model {m['model']!r}, expected {MODEL!r}")
            continue
        if m["loc"] not in LOCATIONS:
            errors.append(f"{path.name}: unexpected location {m['loc']!r}")
            continue
        try:
            labels, values = parse_response(
                path.name,
                path.read_bytes(),
                *LOCATIONS[m["loc"]],
                date.fromisoformat(m["start"]),
                date.fromisoformat(m["end"]),
            )
        except (WeatherFormatError, ValueError) as e:
            errors.append(str(e))
            continue
        frames[m["loc"]].append((path.name, pd.DataFrame(values, index=labels, dtype="float64")))
    rows = []
    for loc, parts in frames.items():
        if not parts:
            errors.append(f"no raw files for location {loc}")
            continue
        overlaps = [
            f"{loc}: duplicate hours, {a} and {b} overlap in {len(both)} hour(s) from {both[0]};"
            " delete one of the two files"
            for i, (a, x) in enumerate(parts)
            for b, y in parts[i + 1 :]
            if len(both := x.index.intersection(y.index))
        ]
        if overlaps:
            errors.extend(overlaps)
            continue
        df = pd.concat([x for _, x in parts]).sort_index()
        ghi = df["shortwave_radiation"].reindex(df.index + pd.Timedelta(hours=1))
        df["shortwave_radiation"] = ghi.to_numpy()
        rows.append(df.rename_axis("valid_time_utc").reset_index().assign(location=loc))
    if errors:
        raise BuildError("\n".join(errors))
    out = pd.concat(rows, ignore_index=True).assign(model=MODEL)
    return out[["valid_time_utc", "location", *VARIABLES, "model"]]


def validate(hourly: pd.DataFrame, hours: pd.DatetimeIndex) -> None:
    want = ["valid_time_utc", "location", *VARIABLES, "model"]
    if list(hourly.columns) != want:
        raise BuildError(f"unexpected columns {list(hourly.columns)}, expected {want}")
    errors = []
    if models := sorted(set(hourly["model"]) - {MODEL}):
        errors.append(f"unexpected model(s) {models}, expected {MODEL!r}")
    dup = hourly[hourly.duplicated(["valid_time_utc", "location"], keep=False)]
    for loc, g in dup.groupby("location"):
        errors.append(f"{loc}: duplicate hours {[str(t) for t in g['valid_time_utc'][:5]]}")
    span = f"{hours[0]}..{hours[-1]}"
    for loc in LOCATIONS:
        g = hourly[hourly["location"] == loc].drop_duplicates("valid_time_utc")
        if g.empty:
            errors.append(f"{loc}: no weather rows")
            continue
        lo, hi = g["valid_time_utc"].min(), g["valid_time_utc"].max()
        if lo > hours[0] or hi < hours[-1]:
            errors.append(f"{loc}: weather range {lo}..{hi} does not cover price range {span}")
        g = g.set_index("valid_time_utc").reindex(hours)
        if missing := g.index[g["location"].isna()].tolist():
            errors.append(
                f"{loc}: {len(missing)} missing hour(s) in the price range, first"
                f" {[str(t) for t in missing[:5]]}"
            )
        present = g[g["location"].notna()]
        for var in VARIABLES:
            if nulls := present.index[present[var].isna()].tolist():
                errors.append(
                    f"{loc}: {len(nulls)} null {var} value(s) in the price range at"
                    f" {[str(t) for t in nulls[:10]]}"
                )
    if errors:
        raise BuildError("\n".join(errors))


def check_bid_time(prices: pd.DataFrame) -> float:
    pairs = (
        prices.assign(hour=prices["delivery_start_utc"].dt.floor("h"))[["hour", "delivery_date"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    margins, late = [], []
    for hour, day in pairs:
        h = hour.to_pydatetime()
        margins.append(bid_deadline(day) - (hour_run(h) + PUBLISHED_AFTER))
        if not issued_before_bid(h, day):
            late.append(f"{hour} (delivery day {day})")
    if late:
        raise BuildError(
            f"{len(late)} hour(s) use a run published after the bid deadline, first {late[:5]}"
        )
    return min(margins) / timedelta(hours=1)


def join(prices: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    wide = hourly.pivot(index="valid_time_utc", columns="location", values=list(VARIABLES))
    present = set(hourly["location"])
    cols = [(var, loc) for var in VARIABLES for loc in LOCATIONS if loc in present]
    wide = wide[cols]
    wide.columns = [f"{VARIABLES[var][0]}_{loc}" for var, loc in cols]
    hour = prices["delivery_start_utc"].dt.floor("h")
    out = pd.concat([prices, wide.reindex(hour).reset_index(drop=True)], axis=1)
    if len(out) != len(prices) or out[wide.columns].isna().any().any():
        raise BuildError("join left a quarter-hour without weather")
    return out


def build(data_dir: Path) -> dict:
    out = data_dir / "dataset"
    prices_path = out / "prices_15min.parquet"
    if not prices_path.exists():
        raise BuildError(f"{prices_path} does not exist; run `ib15 build` first")
    prices = pd.read_parquet(prices_path)
    hours = pd.DatetimeIndex(prices["delivery_start_utc"].dt.floor("h").unique()).sort_values()
    hourly = load_hourly(data_dir / "raw" / "openmeteo")
    validate(hourly, hours)
    margin = check_bid_time(prices)
    weather = (
        hourly[hourly["valid_time_utc"].isin(hours)]
        .sort_values(["valid_time_utc", "location"])
        .reset_index(drop=True)
    )
    joined = join(prices, weather)
    status = _git("status", "--porcelain")
    cov = {
        "source": URL,
        "model": MODEL,
        "lead": LEAD,
        "price_range_hours_utc": {"start": str(hours[0]), "end": str(hours[-1])},
        "locations": {loc: {"lat": lat, "lon": lon} for loc, (lat, lon) in LOCATIONS.items()},
        "hours_expected": len(hours),
        "hours_present": weather.groupby("location").size().to_dict(),
        "nulls": {},
        "units": {var: unit for var, (_, unit) in VARIABLES.items()},
        "radiation": "shortwave_radiation is the mean over the hour starting at valid_time_utc",
        "bid_time_min_margin_hours": margin,
        "rows_weather_hourly": len(weather),
        "rows_prices_weather_15min": len(joined),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }
    weather.to_parquet(out / "weather_hourly.parquet", index=False)
    joined.to_parquet(out / "prices_weather_15min.parquet", index=False)
    (out / "weather_coverage.json").write_text(json.dumps(cov, indent=2) + "\n", encoding="utf-8")
    return cov
