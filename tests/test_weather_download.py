import json
import urllib.request
from datetime import date, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import pytest
from conftest import FIXTURES

from ib15 import weather
from ib15.cli import main

OCT = (date(2025, 10, 25), date(2025, 10, 27))
TWO = {"madrid": weather.LOCATIONS["madrid"], "sevilla": weather.LOCATIONS["sevilla"]}
TODAY = date(2026, 11, 7)


def body(loc, start, end):
    """A valid response for loc over start..end: the real Madrid October fixture with its grid
    cell, times and values replaced."""
    d = json.loads((FIXTURES / weather.file_name("madrid", *OCT)).read_bytes())
    d["latitude"], d["longitude"] = weather.LOCATIONS[loc]
    n = ((end - start).days + 1) * 24
    t0 = datetime(start.year, start.month, start.day)
    d["hourly"] = {
        "time": [f"{t0 + timedelta(hours=i):%Y-%m-%dT%H:%M}" for i in range(n)],
        **{f"{v}_{weather.LEAD}": [1.0] * n for v in weather.VARIABLES},
    }
    return json.dumps(d).encode()


def query(url):
    q = parse_qs(urlsplit(url).query)
    return {k: v[0] for k, v in q.items()}


class FakeOpenMeteo:
    """Stands in for http_get: answers each request with a valid body for the requested
    location and dates, unless ``override`` maps (lat, start) to a body or an exception."""

    def __init__(self, override=None, log=None):
        self.override = override or {}
        self.requested = []
        self.log = log

    def __call__(self, url):
        q = query(url)
        self.requested.append(url)
        if self.log is not None:
            self.log.append(("get", q["latitude"], q["start_date"]))
        key = (float(q["latitude"]), q["start_date"])
        if key in self.override:
            value = self.override[key]
            if isinstance(value, Exception):
                raise value
            return value
        loc = next(n for n, (lat, _) in weather.LOCATIONS.items() if lat == float(q["latitude"]))
        start, end = date.fromisoformat(q["start_date"]), date.fromisoformat(q["end_date"])
        return body(loc, start, end)


@pytest.fixture(autouse=True)
def two_locations(monkeypatch):
    monkeypatch.setattr(weather, "LOCATIONS", TWO)


def run(raw, fake, start, end, today=TODAY, pause=0):
    return weather.download(raw, start, end, fetch=fake, today=today, pause=pause)


def names(raw):
    return sorted(p.name for p in raw.iterdir())


def test_request_url_asks_for_previous_day2_model_and_gmt():
    url = weather.request_url(40.42, -3.70, date(2025, 10, 1), date(2025, 10, 31))
    assert url.startswith("https://previous-runs-api.open-meteo.com/v1/forecast?")
    assert query(url) == {
        "latitude": "40.42",
        "longitude": "-3.7",
        "hourly": "temperature_2m_previous_day2,wind_speed_100m_previous_day2,"
        "shortwave_radiation_previous_day2",
        "models": "ecmwf_ifs025",
        "start_date": "2025-10-01",
        "end_date": "2025-10-31",
        "timezone": "GMT",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
    }


def test_one_request_per_location_and_month_saved_unchanged(tmp_path):
    fake = FakeOpenMeteo()
    r = run(tmp_path, fake, date(2025, 9, 30), date(2025, 11, 2))
    assert not r.errors and not r.skipped
    want = [
        weather.file_name(loc, s, e)
        for loc in TWO
        for s, e in [
            (date(2025, 9, 30), date(2025, 9, 30)),
            (date(2025, 10, 1), date(2025, 10, 31)),
            (date(2025, 11, 1), date(2025, 11, 2)),
        ]
    ]
    assert r.saved == want and names(tmp_path) == sorted(want)
    assert len(fake.requested) == 6
    oct_name = weather.file_name("sevilla", date(2025, 10, 1), date(2025, 10, 31))
    assert (tmp_path / oct_name).read_bytes() == body(
        "sevilla", date(2025, 10, 1), date(2025, 10, 31)
    )


def test_malformed_response_is_not_saved_and_others_continue(tmp_path):
    lat = TWO["madrid"][0]
    fake = FakeOpenMeteo({(lat, "2025-10-25"): b'{"latitude": 40.5}'})
    r = run(tmp_path, fake, *OCT)
    assert [n for n, _ in r.errors] == [weather.file_name("madrid", *OCT)]
    assert "missing key" in r.errors[0][1]
    assert names(tmp_path) == [weather.file_name("sevilla", *OCT)]


def test_settled_chunk_on_disk_is_skipped(tmp_path):
    for loc in TWO:
        (tmp_path / weather.file_name(loc, *OCT)).write_bytes(body(loc, *OCT))
    fake = FakeOpenMeteo()
    r = run(tmp_path, fake, *OCT, today=date(2025, 10, 29))  # end is two days back
    assert fake.requested == [] and len(r.skipped) == 2 and not r.saved


def test_settled_chunk_with_nulls_is_fetched_again_and_replaced(tmp_path):
    # Saved while its end was still in the future: the last hours came back null.
    d = json.loads(body("madrid", *OCT))
    d["hourly"]["temperature_2m_previous_day2"][-5:] = [None] * 5
    with_nulls = weather.file_name("madrid", *OCT)
    (tmp_path / with_nulls).write_text(json.dumps(d))
    complete = weather.file_name("sevilla", *OCT)
    (tmp_path / complete).write_bytes(body("sevilla", *OCT))
    fake = FakeOpenMeteo()
    r = run(tmp_path, fake, *OCT, today=date(2025, 11, 7))
    assert r.saved == [with_nulls] and r.skipped == [complete] and len(fake.requested) == 1
    assert (tmp_path / with_nulls).read_bytes() == body("madrid", *OCT)


def test_rerun_keeps_a_partly_overlapping_file_and_deletes_a_covered_one(tmp_path):
    # Oct 1-20 overlaps Oct 15-31 without covering it, so the chunk is fetched.
    part = weather.file_name("madrid", date(2025, 10, 1), date(2025, 10, 20))
    (tmp_path / part).write_bytes(body("madrid", date(2025, 10, 1), date(2025, 10, 20)))
    covered = weather.file_name("sevilla", date(2025, 10, 20), date(2025, 10, 25))
    (tmp_path / covered).write_bytes(body("sevilla", date(2025, 10, 20), date(2025, 10, 25)))
    r = run(tmp_path, FakeOpenMeteo(), date(2025, 10, 15), date(2025, 10, 31))
    assert not r.errors and len(r.saved) == 2
    narrow = [weather.file_name(loc, date(2025, 10, 15), date(2025, 10, 31)) for loc in TWO]
    # Oct 1-14 survive in the older madrid file; the sevilla file inside the new range is gone.
    assert names(tmp_path) == sorted([part, *narrow])


def test_narrower_rerun_inside_a_complete_saved_month_is_skipped(tmp_path):
    month = (date(2025, 10, 1), date(2025, 10, 31))
    full = [weather.file_name(loc, *month) for loc in TWO]
    run(tmp_path, FakeOpenMeteo(), *month)
    fake = FakeOpenMeteo()
    r = run(tmp_path, fake, date(2025, 10, 15), date(2025, 10, 31))
    assert fake.requested == [] and not r.saved and not r.errors
    assert r.skipped == [
        weather.file_name(loc, date(2025, 10, 15), date(2025, 10, 31)) for loc in TWO
    ]
    assert names(tmp_path) == sorted(full)
    weather.load_hourly(tmp_path)  # no overlap error


def test_incomplete_covering_file_does_not_skip_a_narrower_chunk(tmp_path):
    month = (date(2025, 10, 1), date(2025, 10, 31))
    d = json.loads(body("madrid", *month))
    d["hourly"]["temperature_2m_previous_day2"][-5:] = [None] * 5
    (tmp_path / weather.file_name("madrid", *month)).write_text(json.dumps(d))
    (tmp_path / weather.file_name("sevilla", *month)).write_bytes(b"stale")
    fake = FakeOpenMeteo()
    r = run(tmp_path, fake, date(2025, 10, 15), date(2025, 10, 31))
    assert len(fake.requested) == 2 and not r.skipped and len(r.saved) == 2


def test_recent_chunk_is_fetched_again_and_replaces_the_shorter_file(tmp_path):
    old = weather.file_name("madrid", date(2025, 10, 25), date(2025, 10, 26))
    (tmp_path / old).write_bytes(body("madrid", date(2025, 10, 25), date(2025, 10, 26)))
    current = weather.file_name("sevilla", *OCT)
    (tmp_path / current).write_bytes(b"stale")
    fake = FakeOpenMeteo()
    r = run(tmp_path, fake, *OCT, today=date(2025, 10, 28))  # end is only one day back
    assert len(fake.requested) == 2 and not r.skipped and not r.errors
    assert names(tmp_path) == sorted([weather.file_name("madrid", *OCT), current])
    assert (tmp_path / current).read_bytes() == body("sevilla", *OCT)


def test_sleeps_between_requests(tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr(weather.time, "sleep", lambda s: log.append(("sleep", s)))
    run(tmp_path, FakeOpenMeteo(log=log), *OCT, pause=1.0)
    assert log == [
        ("get", "40.42", "2025-10-25"),
        ("sleep", 1.0),
        ("get", "37.39", "2025-10-25"),
    ]


def test_cli_exits_nonzero_on_bad_response_and_saves_nothing(tmp_path, monkeypatch, capsys):
    lat = TWO["sevilla"][0]
    fake = FakeOpenMeteo({(lat, "2025-10-25"): b"<!DOCTYPE html>", (40.42, "2025-10-25"): None})
    monkeypatch.setattr(weather, "http_get", fake)
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    argv = ["download", "weather", "--data-dir", str(tmp_path)]
    assert main([*argv, "--start", "2025-10-25", "--end", "2025-10-27"]) == 1
    err = capsys.readouterr().err
    assert "not JSON" in err and "HTTP 404" in err
    assert names(tmp_path / "raw" / "openmeteo") == []


def test_cli_default_range_is_the_price_days(tmp_path, monkeypatch, capsys):
    omie_raw = tmp_path / "raw" / "omie"
    omie_raw.mkdir(parents=True)
    (omie_raw / "marginalpdbc_20251001.1").write_bytes(b"")
    (omie_raw / "marginalpdbc_20251003.2").write_bytes(b"")
    fake = FakeOpenMeteo()
    monkeypatch.setattr(weather, "http_get", fake)
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    assert main(["download", "weather", "--data-dir", str(tmp_path)]) == 0
    assert {(query(u)["start_date"], query(u)["end_date"]) for u in fake.requested} == {
        ("2025-09-30", "2025-09-30"),
        ("2025-10-01", "2025-10-03"),
    }


def test_user_agent_is_generic_and_request_has_no_identifying_data(tmp_path, monkeypatch):
    seen = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.data

    def urlopen(req, timeout):
        seen.append(req)
        q = query(req.full_url)
        loc = next(n for n, (lat, _) in TWO.items() if lat == float(q["latitude"]))
        return Resp(body(loc, *OCT))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    r = weather.download(tmp_path, *OCT, today=TODAY, pause=0)
    assert not r.errors and len(seen) == 2
    for req in seen:
        assert req.header_items() == [("User-agent", "research-client/1.0")]
        assert req.data is None
        assert "@" not in req.full_url
        assert set(query(req.full_url)) == {
            "latitude",
            "longitude",
            "hourly",
            "models",
            "start_date",
            "end_date",
            "timezone",
            "temperature_unit",
            "wind_speed_unit",
        }
