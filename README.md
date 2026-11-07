# Iberian Battery 15-Minute Paper

A research paper on how Spain's move to 15-minute electricity prices in 2025 changed what a battery can earn. It compares battery arbitrage value under hourly and 15-minute prices, both with perfect foresight and with a forecast built only from public data (OMIE and ESIOS).

## Price dataset

The `ib15` package downloads OMIE's day-ahead marginal price files and builds the price dataset the paper uses. It covers delivery days from 1 October 2025, the first day of 15-minute day-ahead prices.

Data lives outside the repo. Set `IB15_DATA_DIR` to the data folder, or pass `--data-dir` to each command. The commands refuse to run if neither is set.

```
uv sync
export IB15_DATA_DIR=D:/code/iberian-battery-15min-data
uv run ib15 download omie    # optional: --start YYYY-MM-DD --end YYYY-MM-DD
uv run ib15 build
```

`download omie` saves each day's file unchanged to `$IB15_DATA_DIR/raw/omie/`. It requests one file at a time with a short pause, and uses whichever file version OMIE currently serves (`.1`, `.2`, ...); the version is kept in the file name. OMIE serves only the latest version of a day, so on every run it re-checks each day already downloaded: if the saved version is no longer served, it fetches the newer one, replaces the old file and reports the day as updated. Each response is validated the same way `build` validates files before it is saved; a truncated or malformed response is reported as an error and nothing is written. The default range runs from 2025-10-01 to tomorrow; tomorrow's prices appear around 13:00 Madrid time, so before then it is reported as not published yet. Any other failure is listed at the end and the command exits non-zero.

`build` validates every raw file (period count for the day's length, duplicate periods, dates, prices, gaps between days, more than one version of a day, a series that does not start on 2025-10-01) and stops with an error if anything is wrong. Otherwise it writes to `$IB15_DATA_DIR/dataset/`:

- `prices_15min.parquet`: one row per 15-minute delivery period, with its UTC start, local delivery date, period number, Spanish and Portuguese prices in EUR/MWh, and the OMIE file version.
- `prices_hourly.parquet`: an hourly series constructed as the mean of each hour's four quarter-hours. It is not OMIE's own hourly product.
- `coverage.json`: date range, counts, DST days, days that used a later file version, price summary and the git commit of the build.

Run the tests with `uv run pytest`. They run offline.
