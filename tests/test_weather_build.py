import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from conftest import FIXTURES

from ib15 import weather
from ib15.cli import main
from ib15.dataset import BuildError, quarter_hours
from ib15.omie import read_day

OCT = (date(2025, 10, 25), date(2025, 10, 27))
MAR = (date(2026, 3, 28), date(2026, 3, 30))
OCT_PRICES = ["marginalpdbc_20251026.1", "marginalpdbc_20251027.1"]
MAR_PRICES = ["marginalpdbc_20260329.1", "marginalpdbc_20260330.1"]
TWO = {"madrid": weather.LOCATIONS["madrid"], "sevilla": weather.LOCATIONS["sevilla"]}


def raw_json(loc, rng):
    return json.loads((FIXTURES / weather.file_name(loc, *rng)).read_bytes())


def synthetic(loc, rng):
    d = raw_json("madrid", rng)
    d["latitude"], d["longitude"] = weather.LOCATIONS[loc]
    k = list(weather.LOCATIONS).index(loc)
    for key, vals in d["hourly"].items():
        if key != "time":
            d["hourly"][key] = [None if v is None else v + k for v in vals]
    return json.dumps(d).encode()


def prices(names):
    return quarter_hours([read_day(FIXTURES / n) for n in names])


def hours_of(p):
    return pd.DatetimeIndex(p["delivery_start_utc"].dt.floor("h").unique()).sort_values()


def raw_value(loc, rng, var, label):
    d = raw_json(loc, rng)["hourly"]
    return d[f"{var}_{weather.LEAD}"][d["time"].index(f"{label:%Y-%m-%dT%H:%M}")]


@pytest.fixture
def two_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "LOCATIONS", TWO)
    raw = tmp_path / "raw" / "openmeteo"
    raw.mkdir(parents=True)
    for loc in TWO:
        for rng in (OCT, MAR):
            name = weather.file_name(loc, *rng)
            (raw / name).write_bytes((FIXTURES / name).read_bytes())
    return raw


def test_parse_real_response():
    name = weather.file_name("madrid", *OCT)
    labels, values = weather.parse_response(
        name, (FIXTURES / name).read_bytes(), *weather.LOCATIONS["madrid"], *OCT
    )
    assert len(labels) == 72 and str(labels.tz) == "UTC"
    assert labels[0] == datetime(2025, 10, 25, tzinfo=UTC)
    assert labels[-1] == datetime(2025, 10, 27, 23, tzinfo=UTC)
    assert list(values) == list(weather.VARIABLES)
    assert values["temperature_2m"][0] == raw_value("madrid", OCT, "temperature_2m", labels[0])
    assert all(len(v) == 72 and None not in v for v in values.values())


def bad(mutate):
    d = raw_json("madrid", OCT)
    mutate(d)
    return json.dumps(d).encode()


def _drop_var(d):
    del d["hourly"]["wind_speed_100m_previous_day2"]


def _extra_var(d):
    d["hourly"]["temperature_2m_previous_day1"] = d["hourly"]["temperature_2m_previous_day2"]


def _short_values(d):
    d["hourly"]["shortwave_radiation_previous_day2"].pop()


def _missing_hour(d):
    d["hourly"]["time"].pop(5)


def _wrong_unit(d):
    d["hourly_units"]["wind_speed_100m_previous_day2"] = "km/h"


def _not_gmt(d):
    d["utc_offset_seconds"] = 3600


def _far_cell(d):
    d["latitude"] = 43.0


def _text_value(d):
    d["hourly"]["temperature_2m_previous_day2"][3] = "12.1"


def _set(var, value):
    def mutate(d):
        d["hourly"][f"{var}_previous_day2"][7] = value

    return mutate


@pytest.mark.parametrize(
    "body, message",
    [
        (b"<!DOCTYPE html>", "not JSON"),
        (b'{"error":true,"reason":"Parameter x is invalid"}', "API error"),
        (b"[]", "expected one JSON object"),
        (bad(lambda d: d.pop("hourly")), "missing key 'hourly'"),
        (bad(lambda d: d["hourly"].pop("time")), "missing hourly time array"),
        (bad(_drop_var), "missing variable(s) ['wind_speed_100m_previous_day2']"),
        (bad(_extra_var), "unexpected variable(s) ['temperature_2m_previous_day1']"),
        (bad(_short_values), "shortwave_radiation_previous_day2 has 71 values, expected 72"),
        (bad(_missing_hour), "hourly time is not every hour"),
        (bad(_wrong_unit), "unit is 'km/h', expected 'm/s'"),
        (bad(_not_gmt), "expected 0 (GMT)"),
        (bad(_far_cell), "is not near"),
        (bad(_text_value), "non-numeric"),
        (bad(_set("temperature_2m", 60.5)), "outside -40.0..60.0 °C"),
        (bad(_set("wind_speed_100m", -0.1)), "outside 0.0..100.0 m/s"),
        (bad(_set("shortwave_radiation", 1600)), "outside 0.0..1500.0 W/m²"),
    ],
)
def test_parse_rejects(body, message):
    with pytest.raises(weather.WeatherFormatError, match="madrid") as e:
        weather.parse_response("madrid.x.json", body, *weather.LOCATIONS["madrid"], *OCT)
    assert message in str(e.value)


def test_radiation_is_the_mean_of_the_hour_starting_at_valid_time(two_locations):
    h = weather.load_hourly(two_locations)
    t = datetime(2025, 10, 26, 11, tzinfo=UTC)
    row = h[(h["location"] == "madrid") & (h["valid_time_utc"] == t)].iloc[0]
    assert row["temperature_2m"] == raw_value("madrid", OCT, "temperature_2m", t)
    assert row["wind_speed_100m"] == raw_value("madrid", OCT, "wind_speed_100m", t)
    nxt = t + timedelta(hours=1)
    assert row["shortwave_radiation"] == raw_value("madrid", OCT, "shortwave_radiation", nxt)


def test_radiation_of_a_files_last_hour_comes_from_the_next_file(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "LOCATIONS", {"madrid": TWO["madrid"]})
    d = raw_json("madrid", OCT)
    d["hourly"]["shortwave_radiation_previous_day2"][48] = 7.0
    for (start, end), sl in [
        ((OCT[0], date(2025, 10, 26)), slice(0, 48)),
        ((OCT[1],) * 2, slice(48, None)),
    ]:
        part = {**d, "hourly": {k: v[sl] for k, v in d["hourly"].items()}}
        (tmp_path / weather.file_name("madrid", start, end)).write_text(json.dumps(part))
    h = weather.load_hourly(tmp_path).set_index("valid_time_utc")
    assert h.loc[pd.Timestamp("2025-10-26 23:00", tz=UTC), "shortwave_radiation"] == 7.0


@pytest.mark.parametrize("names, rng", [(OCT_PRICES, OCT), (MAR_PRICES, MAR)], ids=["oct", "mar"])
def test_join_gives_each_quarter_hour_its_utc_hour(two_locations, names, rng):
    p = prices(names)
    h = weather.load_hourly(two_locations)
    j = weather.join(p, h[h["valid_time_utc"].isin(hours_of(p))])
    assert len(j) == len(p)
    pd.testing.assert_frame_equal(j[p.columns], p)
    assert [c for c in j.columns if c not in p.columns] == [
        "temp_madrid",
        "temp_sevilla",
        "wind100_madrid",
        "wind100_sevilla",
        "ghi_madrid",
        "ghi_sevilla",
    ]
    lookup = h.set_index(["location", "valid_time_utc"])
    for _, row in j.iterrows():
        hour = row["delivery_start_utc"].floor("h")
        for loc in TWO:
            want = lookup.loc[(loc, hour)]
            assert row[f"temp_{loc}"] == want["temperature_2m"]
            assert row[f"wind100_{loc}"] == want["wind_speed_100m"]
            assert row[f"ghi_{loc}"] == want["shortwave_radiation"]
    assert (j["temp_madrid"] != j["temp_sevilla"]).any()


def test_join_on_the_repeated_local_hour_of_2025_10_26(two_locations):
    p = prices(OCT_PRICES)
    h = weather.load_hourly(two_locations)
    j = weather.join(p, h[h["valid_time_utc"].isin(hours_of(p))])
    day = j[j["delivery_date"] == date(2025, 10, 26)]
    assert len(day) == 100
    for periods, hour in [((9, 12), 0), ((13, 16), 1)]:
        rows = day[day["period"].between(*periods)]
        t = datetime(2025, 10, 26, hour, tzinfo=UTC)
        assert (rows["temp_madrid"] == raw_value("madrid", OCT, "temperature_2m", t)).all()
        assert (rows["temp_sevilla"] == raw_value("sevilla", OCT, "temperature_2m", t)).all()


def test_join_on_the_short_day_2026_03_29(two_locations):
    p = prices(MAR_PRICES)
    h = weather.load_hourly(two_locations)
    j = weather.join(p, h[h["valid_time_utc"].isin(hours_of(p))])
    day = j[j["delivery_date"] == date(2026, 3, 29)]
    assert len(day) == 92
    row = day[day["period"] == 9].iloc[0]
    t = datetime(2026, 3, 29, 1, tzinfo=UTC)
    assert row["delivery_start_utc"] == t
    assert row["wind100_sevilla"] == raw_value("sevilla", MAR, "wind_speed_100m", t)


@pytest.fixture
def good():
    hours = hours_of(prices(OCT_PRICES))
    frame = pd.DataFrame(
        [(t, loc, 10.0, 5.0, 100.0, weather.MODEL) for t in hours for loc in weather.LOCATIONS],
        columns=["valid_time_utc", "location", *weather.VARIABLES, "model"],
    )
    weather.validate(frame, hours)
    return frame, hours


def test_validate_missing_hour(good):
    frame, hours = good
    drop = (frame["location"] == "lugo") & (frame["valid_time_utc"] == hours[10])
    with pytest.raises(BuildError, match=r"lugo: 1 missing hour\(s\) in the price range"):
        weather.validate(frame[~drop], hours)


def test_validate_null_in_range(good):
    frame, hours = good
    at = (frame["location"] == "burgos") & (frame["valid_time_utc"] == hours[3])
    frame.loc[at, "temperature_2m"] = None
    with pytest.raises(BuildError, match=r"burgos: 1 null temperature_2m value\(s\)") as e:
        weather.validate(frame, hours)
    assert str(hours[3]) in str(e.value)


def test_validate_duplicate_hour(good):
    frame, hours = good
    dup = frame[(frame["location"] == "valencia") & (frame["valid_time_utc"] == hours[0])]
    with pytest.raises(BuildError, match="valencia: duplicate hours"):
        weather.validate(pd.concat([frame, dup]), hours)


def test_validate_short_range(good):
    frame, hours = good
    short = frame[~((frame["location"] == "albacete") & (frame["valid_time_utc"] == hours[-1]))]
    with pytest.raises(BuildError, match="albacete: weather range .* does not cover price range"):
        weather.validate(short, hours)


def test_validate_unexpected_model(good):
    frame, hours = good
    frame.loc[frame.index[0], "model"] = "best_match"
    with pytest.raises(BuildError, match=r"unexpected model\(s\) \['best_match'\]"):
        weather.validate(frame, hours)


def utc(*a):
    return datetime(*a, tzinfo=UTC)


def test_latest_run_matches_the_previous_day2_construction():
    assert weather.latest_run(utc(2026, 11, 4, 0)) == utc(2026, 11, 2, 0)
    assert weather.latest_run(utc(2026, 11, 4, 3)) == utc(2026, 11, 2, 0)
    assert weather.latest_run(utc(2026, 11, 4, 6)) == utc(2026, 11, 2, 6)
    assert weather.latest_run(utc(2026, 11, 4, 22)) == utc(2026, 11, 3, 0)
    assert weather.latest_run(utc(2026, 11, 4, 23)) == utc(2026, 11, 3, 0)


@pytest.mark.parametrize(
    "day, last_hour, deadline",
    [
        pytest.param(
            date(2025, 10, 26),
            utc(2025, 10, 26, 22),
            utc(2025, 10, 25, 10),
            id="25h-day-23h-cet-is-22-utc-d-1-still-cest-deadline-10-utc",
        ),
        pytest.param(
            date(2026, 3, 29),
            utc(2026, 3, 29, 21),
            utc(2026, 3, 28, 11),
            id="23h-day-23h-cest-is-21-utc-d-1-cet-deadline-11-utc",
        ),
        pytest.param(
            date(2026, 7, 15), utc(2026, 7, 15, 21), utc(2026, 7, 14, 10), id="summer-day"
        ),
    ],
)
def test_bid_time_rule_for_the_last_hour_of_the_day(day, last_hour, deadline):
    assert weather.bid_deadline(day) == deadline
    label = last_hour + timedelta(hours=1)
    assert weather.latest_run(label) == datetime.combine(
        day - timedelta(days=1), datetime.min.time(), UTC
    )
    assert weather.hour_run(last_hour) == weather.latest_run(label)
    assert weather.issued_before_bid(last_hour, day)
    assert not weather.issued_before_bid(last_hour, day, lead_days=1)


def test_previous_day1_fails_the_rule_in_the_afternoon():
    assert not weather.issued_before_bid(utc(2026, 7, 15, 13), date(2026, 7, 15), lead_days=1)


def test_build_check_fails_when_a_run_is_published_too_late(monkeypatch):
    p = prices(OCT_PRICES)
    assert weather.check_bid_time(p) == pytest.approx(2.0)
    monkeypatch.setattr(weather, "PUBLISHED_AFTER", timedelta(hours=10, minutes=1))
    with pytest.raises(BuildError, match="published after the bid deadline"):
        weather.check_bid_time(p)


@pytest.mark.parametrize(
    "start, newest_label",
    [
        pytest.param(
            "2026-03-30 21:45",
            "21 UTC, the last hour of 2026-03-30 (CEST): radiation's label 22:00 interpolates "
            "towards 03:00 next day (run 00 UTC D-1); the own label 21:00 is a grid point "
            "(run 18 UTC D-2)",
            id="radiation-label-newer",
        ),
        pytest.param(
            "2026-03-30 20:15",
            "20 UTC: the own label 20:00 interpolates towards 00:00 next day (run 00 UTC D-1); "
            "radiation's label 21:00 is a grid point (run 18 UTC D-2)",
            id="own-label-newer",
        ),
    ],
)
def test_build_check_takes_the_newest_run_of_either_label(start, newest_label):
    one = pd.DataFrame(
        {
            "delivery_start_utc": [pd.Timestamp(start, tz=UTC)],
            "delivery_date": [date(2026, 3, 30)],
        }
    )
    assert weather.check_bid_time(one) == pytest.approx(2.0), (
        f"{newest_label}; deadline 10:00 UTC on 2026-03-29 (CEST), and the 00 UTC run of that "
        "day is public at 08:00"
    )


@pytest.fixture
def full(tmp_path):
    (tmp_path / "dataset").mkdir()
    prices(OCT_PRICES).to_parquet(tmp_path / "dataset" / "prices_15min.parquet", index=False)
    raw = tmp_path / "raw" / "openmeteo"
    raw.mkdir(parents=True)
    for loc in weather.LOCATIONS:
        (raw / weather.file_name(loc, *OCT)).write_bytes(synthetic(loc, OCT))
    return tmp_path


def test_build_writes_weather_files(full):
    cov = weather.build(full)
    out = full / "dataset"
    p = pd.read_parquet(out / "prices_15min.parquet")
    h = pd.read_parquet(out / "weather_hourly.parquet")
    j = pd.read_parquet(out / "prices_weather_15min.parquet")
    n_hours = len(hours_of(p))
    assert n_hours == 49
    assert list(h.columns) == ["valid_time_utc", "location", *weather.VARIABLES, "model"]
    assert str(h["valid_time_utc"].dt.tz) == "UTC"
    assert len(h) == n_hours * 10 and set(h["model"]) == {weather.MODEL}
    assert len(j) == len(p) == 196
    pd.testing.assert_frame_equal(j[p.columns], p)
    assert len(j.columns) == len(p.columns) + 30
    assert (j["temp_zaragoza"] - j["temp_madrid"]).round(6).eq(4).all()
    on_disk = json.loads((out / "weather_coverage.json").read_text())
    assert on_disk == json.loads(json.dumps(cov))
    assert cov["model"] == "ecmwf_ifs025" and cov["lead"] == "previous_day2"
    assert cov["hours_expected"] == n_hours
    assert cov["hours_present"] == {loc: n_hours for loc in sorted(weather.LOCATIONS)}
    assert cov["units"] == {
        "temperature_2m": "°C",
        "wind_speed_100m": "m/s",
        "shortwave_radiation": "W/m²",
    }
    assert cov["nulls"] == {}
    assert cov["locations"]["ciudad_real"] == {"lat": 38.98, "lon": -3.93}


def test_build_fails_on_null_and_writes_nothing(full, capsys):
    name = weather.file_name("lugo", *OCT)
    d = json.loads((full / "raw" / "openmeteo" / name).read_bytes())
    d["hourly"]["wind_speed_100m_previous_day2"][30] = None
    (full / "raw" / "openmeteo" / name).write_text(json.dumps(d))
    assert main(["build", "weather", "--data-dir", str(full)]) == 1
    err = capsys.readouterr().err
    assert "lugo: 1 null wind_speed_100m value(s) in the price range" in err
    assert "2025-10-26 06:00:00+00:00" in err
    assert not (full / "dataset" / "weather_hourly.parquet").exists()
    assert not (full / "dataset" / "prices_weather_15min.parquet").exists()


def test_build_fails_on_overlapping_files(full):
    raw = full / "raw" / "openmeteo"
    src = raw / weather.file_name("badajoz", *OCT)
    d = json.loads(src.read_bytes())
    for key in d["hourly"]:
        d["hourly"][key] = d["hourly"][key][24:]
    narrow = weather.file_name("badajoz", date(2025, 10, 26), OCT[1])
    (raw / narrow).write_text(json.dumps(d))
    with pytest.raises(BuildError, match="badajoz: duplicate hours") as e:
        weather.build(full)
    assert f"{src.name} and {narrow} overlap in 48 hour(s)" in str(e.value)
    assert "delete one of the two files" in str(e.value)


def test_build_fails_on_unexpected_model_or_location_file(full):
    raw = full / "raw" / "openmeteo"
    (raw / "madrid.best_match.2025-10-25.2025-10-27.json").write_bytes(synthetic("madrid", OCT))
    (raw / f"toledo.{weather.MODEL}.2025-10-25.2025-10-27.json").write_bytes(b"{}")
    with pytest.raises(BuildError) as e:
        weather.build(full)
    assert "unexpected model 'best_match'" in str(e.value)
    assert "unexpected location 'toledo'" in str(e.value)


def test_build_fails_on_unexpected_variable_in_raw_file(full):
    name = weather.file_name("lugo", *OCT)
    d = json.loads((full / "raw" / "openmeteo" / name).read_bytes())
    d["hourly"]["temperature_2m"] = d["hourly"]["temperature_2m_previous_day2"]
    (full / "raw" / "openmeteo" / name).write_text(json.dumps(d))
    with pytest.raises(BuildError, match=r"unexpected variable\(s\) \['temperature_2m'\]"):
        weather.build(full)


def test_build_fails_without_prices(tmp_path):
    with pytest.raises(BuildError, match="run `ib15 build` first"):
        weather.build(tmp_path)
