"""Command line: ``ib15 download omie|weather``, ``ib15 build [prices|weather]`` and
``ib15 perfect-foresight``."""

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from ib15 import battery, dataset, omie, weather

ENV_DATA_DIR = "IB15_DATA_DIR"


def _data_dir(arg: str | None) -> Path:
    value = arg or os.environ.get(ENV_DATA_DIR)
    if not value:
        raise SystemExit(f"error: no data directory; pass --data-dir or set {ENV_DATA_DIR}")
    return Path(value)


def _download(args) -> int:
    data_dir = _data_dir(args.data_dir)
    start = args.start or omie.GO_LIVE
    # D+1 is published around 13:00 Madrid time; before that, tomorrow reports as not published.
    end = args.end or omie.madrid_today() + timedelta(days=1)
    if start < omie.GO_LIVE:
        raise SystemExit(f"error: --start {start} is before {omie.GO_LIVE}, out of scope")
    if end < start:
        raise SystemExit(f"error: --end {end} is before --start {start}")
    r = omie.download(data_dir / "raw" / "omie", start, end)
    print(
        f"saved {len(r.saved)}, updated {len(r.updated)}, up to date {len(r.up_to_date)}, "
        f"not published yet {len(r.not_published)}, errors {len(r.errors)}"
    )
    for d, v in r.saved:
        if v != 1:
            print(f"  {d} used version {v}")
    if r.not_published:
        print(f"  not published yet: {r.not_published[0]} .. {r.not_published[-1]}")
    for d, msg in r.errors:
        print(f"  ERROR {d}: {msg}", file=sys.stderr)
    return 1 if r.errors else 0


def _latest_price_day(data_dir: Path):
    """The latest delivery day among the raw OMIE files, or None if there are none."""
    raw = data_dir / "raw" / "omie"
    days = [m.group(1) for p in raw.glob("marginalpdbc_*") if (m := omie.FILE_RE.match(p.name))]
    return datetime.strptime(max(days), "%Y%m%d").date() if days else None


def _download_weather(args) -> int:
    data_dir = _data_dir(args.data_dir)
    # The first price day starts at 22:00 UTC the day before (local midnight in Madrid).
    start = args.start or omie.GO_LIVE - timedelta(days=1)
    end = args.end or _latest_price_day(data_dir) or omie.madrid_today() + timedelta(days=1)
    if end < start:
        raise SystemExit(f"error: --end {end} is before --start {start}")
    print(f"weather {start} .. {end} (UTC dates), {weather.MODEL} {weather.LEAD}")
    r = weather.download(data_dir / "raw" / "openmeteo", start, end)
    print(f"saved {len(r.saved)}, already saved {len(r.skipped)}, errors {len(r.errors)}")
    for name, msg in r.errors:
        print(f"  ERROR {name}: {msg}", file=sys.stderr)
    return 1 if r.errors else 0


def _build(args) -> int:
    data_dir = _data_dir(args.data_dir)
    if args.what == "weather":
        try:
            cov = weather.build(data_dir)
        except dataset.BuildError as e:
            print(f"weather build failed:\n{e}", file=sys.stderr)
            return 1
        print(
            f"built weather {cov['model']} {cov['lead']}: {cov['rows_weather_hourly']} hourly rows,"
            f" {cov['rows_prices_weather_15min']} quarter-hours -> {data_dir / 'dataset'}"
        )
        return 0
    try:
        cov = dataset.build(data_dir)
    except dataset.BuildError as e:
        print(f"build failed:\n{e}", file=sys.stderr)
        return 1
    print(
        f"built {cov['date_range']['start']} .. {cov['date_range']['end']}: "
        f"{cov['days']} days, {cov['periods_15min']} periods -> {data_dir / 'dataset'}"
    )
    return 0


def _perfect_foresight(args) -> int:
    data_dir = _data_dir(args.data_dir)
    try:
        meta = battery.perfect_foresight(data_dir)
    except (dataset.BuildError, battery.SolveError) as e:
        print(f"perfect-foresight failed:\n{e}", file=sys.stderr)
        return 1
    print(
        f"solved {meta['date_range']['start']} .. {meta['date_range']['end']}: {meta['days']} days"
        f" in {meta['solve_seconds']} s -> {data_dir / 'results'}"
    )
    for key, n in meta["simultaneous_periods"].items():
        print(f"  simultaneous charge and discharge periods, {key}: {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", help=f"data directory (default: ${ENV_DATA_DIR})")

    parser = argparse.ArgumentParser(prog="ib15")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="download raw files")
    dl_sub = dl.add_subparsers(dest="source", required=True)
    om = dl_sub.add_parser("omie", parents=[common], help="OMIE day-ahead marginal prices")
    om.add_argument("--start", type=date.fromisoformat, help=f"first day (default {omie.GO_LIVE})")
    om.add_argument(
        "--end",
        type=date.fromisoformat,
        help="last day (default: tomorrow in Madrid, the latest that can be published)",
    )
    om.set_defaults(func=_download)
    we = dl_sub.add_parser(
        "weather", parents=[common], help="Open-Meteo archived forecasts (ECMWF IFS, 48 h lead)"
    )
    we.add_argument(
        "--start",
        type=date.fromisoformat,
        help=f"first UTC date (default {omie.GO_LIVE - timedelta(days=1)}, where the first"
        " price day starts in UTC)",
    )
    we.add_argument(
        "--end",
        type=date.fromisoformat,
        help="last UTC date (default: the latest day in raw/omie, else tomorrow in Madrid)",
    )
    we.set_defaults(func=_download_weather)
    b = sub.add_parser("build", parents=[common], help="validate raw files and write the dataset")
    b.add_argument(
        "what",
        nargs="?",
        choices=["prices", "weather"],
        default="prices",
        help="prices (default), or weather: join the forecasts to prices_15min.parquet",
    )
    b.set_defaults(func=_build)
    pf = sub.add_parser(
        "perfect-foresight",
        parents=[common],
        help="perfect-foresight battery value at 15-minute and hourly prices",
    )
    pf.set_defaults(func=_perfect_foresight)

    args = parser.parse_args(argv)
    return args.func(args)
