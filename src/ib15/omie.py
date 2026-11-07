"""OMIE day-ahead marginal price files (marginalpdbc): download and parse one file per day.

A file covers one delivery day. Layout (latin-1 text): a ``MARGINALPDBC;`` header line, then
rows ``year;month;day;period;price_PT;price_ES;`` in EUR/MWh, then a ``*`` trailer line.
Periods count from local midnight Europe/Madrid. From GO_LIVE a period is 15 minutes, so a
day has 96 periods (92 or 100 on DST change days); before that a period was one hour.
"""

import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

GO_LIVE = date(2025, 10, 1)
MADRID = ZoneInfo("Europe/Madrid")
URL = "https://www.omie.es/es/file-download?parents=marginalpdbc&filename={name}"
USER_AGENT = "research-client/1.0"
HEADER = "MARGINALPDBC;"
# OMIE serves only the latest version of a day's file (.1, .2, ...); seen up to .3 so far.
MAX_VERSION = 9
FILE_RE = re.compile(r"^marginalpdbc_(\d{8})\.(\d+)$")


def file_name(day: date, version: int) -> str:
    return f"marginalpdbc_{day:%Y%m%d}.{version}"


def madrid_today() -> date:
    return datetime.now(MADRID).date()


def local_midnight_utc(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=MADRID).astimezone(UTC)


def expected_periods(day: date) -> int:
    """Number of 15-minute periods in a local day: 96, or 92/100 on DST change days."""
    length = local_midnight_utc(day + timedelta(days=1)) - local_midnight_utc(day)
    return int(length / timedelta(minutes=15))


# --- download -------------------------------------------------------------------------------


def http_get(url: str) -> bytes | None:
    """GET a URL. Returns the body, or None on 404. Other failures raise."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


@dataclass
class DownloadResult:
    saved: list[tuple[date, int]]
    skipped: list[date]
    not_published: list[date]
    errors: list[tuple[date, str]]


def download(
    raw_dir: Path,
    start: date,
    end: date,
    *,
    fetch=None,
    today: date | None = None,
    pause: float = 0.5,
) -> DownloadResult:
    """Download each day's file from start to end (inclusive) into raw_dir, unchanged.

    Sequential, with ``pause`` seconds between requests. A day that already has a file in
    raw_dir is skipped. For each day, versions 1..MAX_VERSION are tried in order and the first
    that exists is saved under its own name, so the file name records the version. A day after
    today (Madrid) that has no file yet is reported as not published, not as an error; days
    more than one day ahead are not requested at all.
    """
    fetch = fetch or http_get
    today = today or madrid_today()
    raw_dir.mkdir(parents=True, exist_ok=True)
    have = {m.group(1) for p in raw_dir.iterdir() if (m := FILE_RE.match(p.name))}
    result = DownloadResult([], [], [], [])
    first_request = True

    day = start
    while day <= end:
        if f"{day:%Y%m%d}" in have:
            result.skipped.append(day)
        elif day > today + timedelta(days=1):
            result.not_published.append(day)
        else:
            outcome = None
            for version in range(1, MAX_VERSION + 1):
                if not first_request:
                    time.sleep(pause)
                first_request = False
                name = file_name(day, version)
                try:
                    body = fetch(URL.format(name=name))
                except Exception as e:  # noqa: BLE001 - reported per day at the end
                    outcome = ("error", f"{name}: {type(e).__name__}: {e}")
                    break
                if body is None:
                    continue
                if not body.startswith(HEADER.encode("latin-1")):
                    outcome = ("error", f"{name}: response is not an OMIE marginalpdbc file")
                    break
                tmp = raw_dir / (name + ".part")
                tmp.write_bytes(body)
                tmp.replace(raw_dir / name)
                outcome = ("saved", version)
                break
            if outcome is None:
                if day > today:
                    result.not_published.append(day)
                else:
                    result.errors.append((day, f"no version 1..{MAX_VERSION} found (all 404)"))
            elif outcome[0] == "saved":
                result.saved.append((day, outcome[1]))
                print(f"{day} saved version {outcome[1]}")
            else:
                result.errors.append((day, outcome[1]))
                print(f"{day} ERROR {outcome[1]}")
        day += timedelta(days=1)
    return result


# --- parse ----------------------------------------------------------------------------------


class OmieFormatError(ValueError):
    pass


@dataclass
class DayFile:
    day: date
    version: int
    rows: list[tuple[int, float, float]]  # (period, price_pt, price_es), sorted by period


def _price(text: str, name: str, lineno: int) -> float:
    try:
        value = float(text)
    except ValueError:
        raise OmieFormatError(f"{name} line {lineno}: non-numeric price {text!r}") from None
    if not math.isfinite(value):
        raise OmieFormatError(f"{name} line {lineno}: non-numeric price {text!r}")
    return value


def read_day(path: Path) -> DayFile:
    """Parse and validate one raw file. Raises OmieFormatError with the file name on any fault."""
    name = path.name
    m = FILE_RE.match(name)
    if not m:
        raise OmieFormatError(f"{name}: unexpected file name")
    day = datetime.strptime(m.group(1), "%Y%m%d").date()
    version = int(m.group(2))
    if day < GO_LIVE:
        raise OmieFormatError(f"{name}: day {day} is before {GO_LIVE}, out of scope")

    lines = path.read_bytes().decode("latin-1").splitlines()
    if not lines or lines[0].strip() != HEADER:
        got = lines[0].strip() if lines else ""
        raise OmieFormatError(f"{name}: unexpected header {got!r}, expected {HEADER!r}")

    rows: dict[int, tuple[float, float]] = {}
    trailer = False
    for lineno, raw in enumerate(lines[1:], start=2):
        line = raw.strip()
        if not line:
            continue
        if trailer:
            raise OmieFormatError(f"{name} line {lineno}: content after the '*' trailer")
        if line == "*":
            trailer = True
            continue
        fields = line.removesuffix(";").split(";")
        if len(fields) != 6:
            raise OmieFormatError(f"{name} line {lineno}: expected 6 fields, got {len(fields)}")
        try:
            row_day = date(int(fields[0]), int(fields[1]), int(fields[2]))
            period = int(fields[3])
        except ValueError:
            raise OmieFormatError(f"{name} line {lineno}: bad date or period {line!r}") from None
        if row_day != day:
            raise OmieFormatError(
                f"{name} line {lineno}: row date {row_day} differs from file date"
            )
        if period in rows:
            raise OmieFormatError(f"{name} line {lineno}: duplicate period {period}")
        rows[period] = (_price(fields[4], name, lineno), _price(fields[5], name, lineno))
    if not trailer:
        raise OmieFormatError(f"{name}: missing '*' trailer")

    n = expected_periods(day)
    if len(rows) == 24:
        raise OmieFormatError(f"{name}: hourly file (24 periods) on or after {GO_LIVE}")
    want = set(range(1, n + 1))
    if set(rows) != want:
        missing = sorted(want - set(rows))
        extra = sorted(set(rows) - want)
        raise OmieFormatError(
            f"{name}: wrong period count, expected {n} for this day, got {len(rows)}"
            f" (missing {missing[:10]}, unexpected {extra[:10]})"
        )
    return DayFile(day, version, [(p, *rows[p]) for p in sorted(rows)])
