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

## Weather forecasts

`ib15` also downloads archived weather forecasts as they were issued before each day's bid, and joins them to the prices.

```
uv run ib15 download weather    # optional: --start YYYY-MM-DD --end YYYY-MM-DD (UTC dates)
uv run ib15 build weather       # after `ib15 build`
```

The source is Open-Meteo's Previous Runs API (free for non-commercial use, no key). It serves one fixed model, ECMWF IFS 0.25° (`ecmwf_ifs025`), at the `_previous_day2` lead: the forecast made at least 48 hours before each hour. The variables are hourly `temperature_2m` (°C), `wind_speed_100m` (m/s) and `shortwave_radiation` (W/m²). There are ten locations in mainland Spain: Madrid, Barcelona, Valencia, Sevilla, Zaragoza, Burgos, Lugo, Badajoz, Ciudad Real and Albacete. The coordinates are in `src/ib15/weather.py`.

Bids for delivery day D close at 12:00 Madrid time on D-1. Every forecast value used for D comes from a model run that ECMWF had published by about 08:00 UTC on D-1. The newest is the 00 UTC run of D-1, which the late hours of D use through Open-Meteo's interpolation from 3-hourly to hourly values. That is at least two hours before the deadline. The one-day lead (`_previous_day1`) would break this rule. The evidence is in the docstring of `src/ib15/weather.py`, and `build weather` checks the rule for every hour.

`download weather` makes one request per location and calendar month and saves each response unchanged to `$IB15_DATA_DIR/raw/openmeteo/` as `<location>.<model>.<start>.<end>.json`. It pauses between requests and validates each response before saving it. A month already saved is skipped, unless it ends less than two days before today or its file has null values: then it is fetched again, because its last hours may not have been filled in when it was fetched. A new file replaces older files of the same location whose dates lie inside its own; a wider file is kept, and if it overlaps the new one, `build weather` names both files so you can delete one. The default range starts on 2025-09-30, because the first price day begins at 22:00 UTC that day, and ends on the latest day in `raw/omie`. Any failed request is listed at the end and the command exits non-zero.

`build weather` reads `prices_15min.parquet` and the raw weather files. It stops with an error if any location has a missing hour, a null value, a duplicate hour or a range that does not cover the prices, if a file has an unexpected model or variable, or if any hour would break the bid-time rule. Otherwise it writes to `$IB15_DATA_DIR/dataset/`:

- `weather_hourly.parquet`: one row per UTC hour and location, for every hour of the price range. Each row describes the hour that starts at `valid_time_utc`. Temperature and wind are the values at that time. Radiation is the mean over that hour; Open-Meteo labels that mean with the hour's end.
- `prices_weather_15min.parquet`: the rows of `prices_15min.parquet`, each with the weather of the UTC hour containing its start, one column per variable and location (`temp_madrid`, `wind100_zaragoza`, `ghi_badajoz`, ...).
- `weather_coverage.json`: model, lead, locations, hours expected and present per location, nulls, units, the smallest margin before the bid deadline and the git commit.

## Tests

Run the tests with `uv run pytest`. They run offline.
