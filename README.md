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

## Perfect-foresight battery value

```
uv run ib15 perfect-foresight    # after `ib15 build`
```

This computes the most a battery could have earned on each delivery day if it had known the day's Spanish day-ahead prices (`price_es`) in advance. It does this once with the 15-minute prices and once with the hourly benchmark, where charging and discharging power stay constant for each whole hour.

The battery has 1 MW of power and 1, 2 or 4 hours of storage (1, 2 or 4 MWh). Its round-trip efficiency is 88%, split equally between charging and discharging. Every delivery day, including the DST change days, starts and ends half full and is optimised on its own as a linear programme solved with HiGHS. There are three specifications. `main` charges 20 EUR for every MWh discharged as a degradation cost. `no_cost` has no degradation cost. `cap2` has no degradation cost and allows at most two full cycles a day.

The command writes to `$IB15_DATA_DIR/results/`:

- `perfect_foresight_daily.parquet`: one row per delivery day, duration, specification and resolution, with gross revenue, degradation cost and net value in EUR per MW, MWh discharged, full cycles, and the number of periods in which the solution both charges and discharges. Charging and discharging share the 1 MW converter, so together they never exceed 1 MW in a period; a period with both stands for the battery alternating between them within it, which burns energy in the losses and is worth doing only at negative prices. The file also holds the `cost_10`, `cost_30` and `cost_40` specifications that the cost sensitivity adds: no cycle cap and a degradation cost of 10, 30 or 40 EUR per MWh discharged.
- `perfect_foresight_summary.csv`: for each duration and specification, the net and gross value over the whole sample at both resolutions, the 15-minute uplift in kEUR/MW, EUR/MW a day and percent, the degradation cost at both resolutions, and the mean cycles a day. A percent uplift is empty (NaN) when the hourly total is zero.
- `perfect_foresight_cost_sensitivity.csv`: the same columns for the setting without a cycle cap at degradation costs of 0, 10, 20, 30 and 40 EUR per MWh discharged, one row per duration and cost. The 20 and 0 rows are the `main` and `no_cost` rows of the summary, taken from the same solves.
- `perfect_foresight_meta.json`: the battery parameters, the degradation cost and cycle cap of every specification in the daily file, the cost grid of the sensitivity, the dates covered, the solve time and the git commit.

Every file is first written to a temporary `.part` name. Once all of them are complete, each previous file is moved aside to a `.prev` name, the new files are renamed into place, and the `.prev` files are deleted. A file that another program holds open, such as a CSV open in Excel on Windows, cannot be moved aside, so the run stops with an error naming that file before any new file is in place. On any error the files already moved aside are moved back and the `.part` files are removed, so a failed run leaves the previous results as they were. If a run is killed part way through, the next run, before it writes its own files, moves the `.prev` files it finds back into place, or deletes them if the killed run had already put all its new files in place. Nothing is written if any day fails to solve.

## Exploratory analysis

```
uv run ib15 eda    # after `ib15 build`, `ib15 build weather` and `ib15 perfect-foresight`
```

This writes the tables and figures that describe the data and where the 15-minute value comes from to `$IB15_DATA_DIR/results/eda/`. It stops with an error if `prices_15min.parquet`, `prices_hourly.parquet`, `weather_hourly.parquet` or `perfect_foresight_daily.parquet` is missing. An hour is the group of four consecutive periods of a local delivery day, as in the hourly benchmark, and its within-hour spread is the highest minus the lowest of its four quarter-hour prices. National mean weather is the mean of the ten locations at each UTC hour. Uplift is the 15-minute net value minus the hourly net value of the `main` specification. All correlations are plain Pearson coefficients and are descriptive only.

- `price_summary.csv`: distribution of the Spanish price at 15 minutes and in the hourly benchmark.
- `within_hour_spread.csv`: within-hour spread overall and by month (mean, median, 90th percentile, share of flat hours) and the mean absolute deviation of a quarter-hour from its hourly mean.
- `uplift_by_month.csv`: net value at both resolutions and the uplift per month and duration, with that month's mean spread and mean price.
- `uplift_concentration.csv`: per duration, the share of the total uplift from the top 10% and 25% of days, the share of days with no uplift, the median and mean daily uplift, and the correlation of daily uplift with the day's mean spread and its price range.
- `weather_summary.csv`: mean, standard deviation, minimum and maximum of each weather variable at each location.
- `forecast_motivation.csv`: autocorrelation of the price and of the within-hour deviation at lags of one and seven days, and correlations of the price, the within-hour spread and the within-hour slope (last quarter minus first) with the national mean weather. The signed within-hour deviation is uncorrelated with the weather of its hour by construction, since the deviations sum to zero within each hour, so the shape is described by the spread and the slope instead.
- Figures, each as PNG and PDF: `fig_a` daily mean price and daily mean spread; `fig_b` average price by quarter-hour of day for each season, with the hourly benchmark of one season as steps; `fig_c` mean deviation of each quarter from its hourly mean by hour of day; `fig_d` distribution of the within-hour spread; `fig_e` uplift by month; `fig_f` daily uplift against the day's mean spread; `fig_g` uplift against the wear cost, drawn only when `perfect_foresight_cost_sensitivity.csv` exists; `fig_h` price against national mean radiation and wind.
- `eda_meta.json`: the inputs with their row counts and date ranges, the outputs, anything skipped, and the git commit.

## Tests

Run the tests with `uv run pytest`. They run offline.
