from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ib15 import battery, eda, forecast, omie, weather
from ib15.dataset import hourly_benchmark, quarter_hours

settings.register_profile("ib15", derandomize=True, deadline=None, max_examples=60)
settings.load_profile("ib15")

PRICE = st.floats(-500, 3000, allow_nan=False).map(lambda x: round(x, 2))
DAYS = st.dates(date(2025, 10, 1), date(2030, 12, 31))


def price_day(day: date, data) -> omie.DayFile:
    n = omie.expected_periods(day)
    pairs = data.draw(st.lists(st.tuples(PRICE, PRICE), min_size=n, max_size=n))
    return omie.DayFile(day, 1, [omie.PriceRow(i + 1, pt, es) for i, (pt, es) in enumerate(pairs)])


def render(d: omie.DayFile) -> bytes:
    rows = [
        f"{d.day.year};{d.day.month:02d};{d.day.day:02d};{r.period};{r.price_pt:.2f};{r.price_es:.2f};"
        for r in d.rows
    ]
    return "\r\n".join([omie.HEADER, *rows, "*", ""]).encode("latin-1")


@given(day=DAYS, data=st.data())
def test_an_omie_file_round_trips_through_the_parser(day, data):
    d = price_day(day, data)
    assert omie.parse_day(omie.file_name(day, 1), render(d)) == d


@given(year=st.integers(2025, 2035))
def test_a_year_has_one_short_and_one_long_day_and_keeps_its_quarter_hours(year):
    days = [date(year, 1, 1) + timedelta(days=i) for i in range(366)]
    counts = [omie.expected_periods(d) for d in days if d.year == year]
    assert sorted(set(counts)) == [92, 96, 100]
    assert counts.count(92) == counts.count(100) == 1
    assert sum(counts) == 96 * len(counts)


@given(day=DAYS, data=st.data())
def test_the_hourly_benchmark_keeps_each_hours_quarter_hours(day, data):
    q = quarter_hours([price_day(day, data)])
    h = hourly_benchmark(q)
    quarters = q["price_es"].to_numpy().reshape(-1, 4)
    np.testing.assert_allclose(h["price_es"].to_numpy() * 4, quarters.sum(axis=1), atol=1e-6)
    assert (h["price_es"].to_numpy() >= quarters.min(axis=1) - 1e-9).all()
    assert (h["price_es"].to_numpy() <= quarters.max(axis=1) + 1e-9).all()
    assert (q["delivery_start_utc"].diff().dropna() == pd.Timedelta(minutes=15)).all()


QUARTER_PRICES = st.integers(2, 6).flatmap(
    lambda hours: st.lists(
        st.floats(-100, 300, allow_nan=False), min_size=4 * hours, max_size=4 * hours
    )
)


@settings(max_examples=40)
@given(prices=QUARTER_PRICES, hours=st.sampled_from(battery.DURATIONS_H))
def test_finer_dispatch_never_earns_less_and_doing_nothing_is_a_floor(prices, hours):
    p = np.array(prices)
    fine = battery.solve_day(p, battery.DT_15MIN, hours, battery.MAIN_SPEC)
    coarse = battery.solve_day(
        p.reshape(-1, 4).mean(axis=1), battery.DT_HOURLY, hours, battery.MAIN_SPEC
    )
    assert coarse.net >= -1e-6
    assert fine.net >= coarse.net - 1e-6


@settings(max_examples=40)
@given(prices=QUARTER_PRICES, hours=st.sampled_from(battery.DURATIONS_H))
def test_a_solved_day_keeps_its_energy_within_capacity_and_ends_where_it_started(prices, hours):
    r = battery.solve_day(np.array(prices), battery.DT_15MIN, hours, battery.MAIN_SPEC)
    assert (r.energy >= -1e-7).all() and (r.energy <= hours + 1e-7).all()
    assert r.energy[-1] == pytest.approx(battery.SOC_START * hours, abs=1e-7)
    assert (r.charge + r.discharge <= battery.POWER_MW + 1e-7).all()


@settings(max_examples=30)
@given(prices=QUARTER_PRICES, low=st.floats(0, 40), extra=st.floats(0.5, 40))
def test_a_higher_wear_cost_never_raises_the_net_value(prices, low, extra):
    p = np.array(prices)
    cheap = battery.solve_day(p, battery.DT_15MIN, 2, battery.Spec("a", low, None))
    dear = battery.solve_day(p, battery.DT_15MIN, 2, battery.Spec("b", low + extra, None))
    assert dear.net <= cheap.net + 1e-6
    assert dear.discharged_mwh <= cheap.discharged_mwh + 1e-6


@given(n=st.integers(1, 60), block=st.integers(1, 10), seed=st.integers(0, 2**32 - 1))
def test_bootstrap_rows_are_circular_runs_of_valid_days(n, block, seed):
    idx = battery.bootstrap_days(n, block=block, resamples=5, seed=seed)
    assert idx.shape == (5, n)
    assert ((idx >= 0) & (idx < n)).all()
    steps = (np.diff(idx, axis=1) % n)[:, : n - 1]
    within_block = (np.arange(1, n) % block) != 0
    assert (steps[:, within_block] == 1 % n).all()


@given(
    x=st.floats(-1e6, 1e6, allow_nan=False),
    ref=st.floats(-1e6, 1e6, allow_nan=False).filter(lambda r: abs(r) > 1e-3),
)
def test_a_percentage_times_its_reference_gives_back_the_value(x, ref):
    assert float(battery.percent(x, ref)) * ref / 100 == pytest.approx(x, rel=1e-9, abs=1e-9)


@given(values=st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=92, max_size=92))
def test_the_missing_dst_hour_takes_the_hour_before(values):
    slots = np.concatenate([np.arange(8), np.arange(12, 96)])
    out = forecast.to_slots(np.array(values), slots)
    np.testing.assert_array_equal(out[8:12], out[4:8])
    np.testing.assert_array_equal(out[slots], values)


@given(values=st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=100, max_size=100))
def test_the_repeated_dst_hour_is_the_mean_of_its_occurrences(values):
    slots = np.concatenate([np.arange(12), np.arange(8, 96)])
    out = forecast.to_slots(np.array(values), slots)
    v = np.array(values)
    np.testing.assert_allclose(out[8:12], (v[8:12] + v[12:16]) / 2)
    assert out.mean() * 96 == pytest.approx(v.sum() - (v[8:12] + v[12:16]).sum() / 2, abs=1e-6)


LOSSES = st.lists(st.floats(-100, 100, allow_nan=False), min_size=10, max_size=60)


@given(d=LOSSES, shift=st.floats(-50, 50), scale=st.floats(0.1, 10), lag=st.integers(0, 8))
def test_the_newey_west_variance_ignores_a_shift_and_scales_with_the_square(d, shift, scale, lag):
    d = np.array(d)
    v = forecast.newey_west_variance(d, lag)
    assert forecast.newey_west_variance(d + shift, lag) == pytest.approx(v, rel=1e-6, abs=1e-6)
    assert forecast.newey_west_variance(d * scale, lag) == pytest.approx(
        v * scale**2, rel=1e-6, abs=1e-6
    )
    if lag == 0:
        assert v == pytest.approx(np.var(d), rel=1e-9, abs=1e-9)


@given(a=LOSSES, data=st.data(), lag=st.integers(0, 8))
def test_swapping_the_models_flips_the_dm_statistic(a, data, lag):
    b = data.draw(st.lists(st.floats(-100, 100, allow_nan=False), min_size=len(a), max_size=len(a)))
    ab, ba = forecast.dm_test(a, b, lag), forecast.dm_test(b, a, lag)
    assert ab["mean_loss_diff"] == pytest.approx(-ba["mean_loss_diff"], abs=1e-9)
    if not np.isnan(ab["dm_stat"]):
        assert ab["dm_stat"] == pytest.approx(-ba["dm_stat"], rel=1e-9, abs=1e-9)
        assert ab["p_value"] == pytest.approx(ba["p_value"], rel=1e-9, abs=1e-12)


@given(start=DAYS, length=st.integers(0, 800))
def test_month_chunks_tile_the_range_one_calendar_month_each(start, length):
    end = start + timedelta(days=length)
    chunks = weather.month_chunks(start, end)
    assert chunks[0][0] == start and chunks[-1][1] == end
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
        assert next_start - prev_end == timedelta(days=1)
    for cs, ce in chunks:
        assert cs <= ce and (cs.year, cs.month) == (ce.year, ce.month)


@given(
    label=st.datetimes(datetime(2025, 9, 1), datetime(2030, 12, 31)).map(
        lambda t: t.replace(minute=0, second=0, microsecond=0, tzinfo=UTC)
    )
)
def test_the_run_behind_a_day2_value_is_between_42_and_54_hours_old(label):
    run = weather.latest_run(label)
    assert run.hour % 6 == 0 and run.minute == 0
    assert timedelta(hours=42) <= label - run < timedelta(hours=54)


@given(day=DAYS, hour=st.integers(0, 24))
def test_every_hour_of_a_delivery_day_uses_weather_public_before_the_bid(day, hour):
    start = omie.local_midnight_utc(day)
    end = omie.local_midnight_utc(day + timedelta(days=1))
    h = min(start + timedelta(hours=hour), end - timedelta(hours=1))
    assert weather.issued_before_bid(h, day)


UPLIFTS = st.lists(st.floats(0, 1e4, allow_nan=False), min_size=1, max_size=200).filter(
    lambda u: sum(u) > 1e-3
)


@given(uplift=UPLIFTS, data=st.data())
def test_the_top_days_hold_at_least_their_share_and_order_does_not_matter(uplift, data):
    c = eda.concentration(uplift)
    n = len(uplift)
    top10, top25 = c["share_from_top_10pct_days"], c["share_from_top_25pct_days"]
    assert top10 <= top25 + 1e-12 <= 1 + 1e-9
    assert top10 >= np.ceil(0.10 * n - 1e-9) / n - 1e-9
    assert top25 >= np.ceil(0.25 * n - 1e-9) / n - 1e-9
    assert eda.concentration(data.draw(st.permutations(uplift))) == pytest.approx(c)


@given(day=DAYS, data=st.data())
def test_quarter_deviations_sum_to_zero_within_each_hour(day, data):
    q = quarter_hours([price_day(day, data)])
    dev = eda.quarter_deviation(q).to_numpy().reshape(-1, 4)
    np.testing.assert_allclose(dev.sum(axis=1), 0, atol=1e-6)
