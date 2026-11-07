"""Command line: ``ib15 download omie`` and ``ib15 build``."""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from ib15 import dataset, omie

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


def _build(args) -> int:
    data_dir = _data_dir(args.data_dir)
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
    b = sub.add_parser("build", parents=[common], help="validate raw files and write the dataset")
    b.set_defaults(func=_build)

    args = parser.parse_args(argv)
    return args.func(args)
