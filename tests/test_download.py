import urllib.error
import urllib.request
from datetime import date

import pytest
from conftest import FIXTURES, day_text

from ib15 import omie
from ib15.cli import main

BODY = (FIXTURES / "marginalpdbc_20251001.1").read_bytes()
TODAY = date(2026, 11, 7)


def body(day):
    return day_text(day).encode("latin-1")


class FakeOmie:
    """Stands in for http_get: maps file name -> body, None (404) or an exception."""

    def __init__(self, files, log=None):
        self.files = files
        self.requested = []
        self.log = log

    def __call__(self, url):
        name = url.rsplit("filename=", 1)[1]
        self.requested.append(name)
        if self.log is not None:
            self.log.append(("get", name))
        value = self.files.get(name)
        if isinstance(value, Exception):
            raise value
        return value


def run(tmp_path, fake, start, end, pause=0):
    return omie.download(tmp_path, start, end, fetch=fake, today=TODAY, pause=pause)


def names(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir())


def test_falls_back_to_version_2_when_1_is_404(tmp_path):
    nov27 = body(date(2025, 11, 27))
    fake = FakeOmie({"marginalpdbc_20251127.2": nov27})
    r = run(tmp_path, fake, date(2025, 11, 27), date(2025, 11, 27))
    assert fake.requested == ["marginalpdbc_20251127.1", "marginalpdbc_20251127.2"]
    assert r.saved == [(date(2025, 11, 27), 2)] and not r.errors
    assert (tmp_path / "marginalpdbc_20251127.2").read_bytes() == nov27  # saved unchanged
    assert names(tmp_path) == ["marginalpdbc_20251127.2"]


def test_day_on_disk_still_served_is_up_to_date(tmp_path):
    (tmp_path / "marginalpdbc_20251001.3").write_bytes(BODY)
    fake = FakeOmie(
        {"marginalpdbc_20251001.3": BODY, "marginalpdbc_20251002.1": body(date(2025, 10, 2))}
    )
    r = run(tmp_path, fake, date(2025, 10, 1), date(2025, 10, 2))
    # The version on disk is requested again; still served means no newer one exists.
    assert fake.requested == ["marginalpdbc_20251001.3", "marginalpdbc_20251002.1"]
    assert r.up_to_date == [date(2025, 10, 1)] and not r.updated and not r.errors
    assert r.saved == [(date(2025, 10, 2), 1)]
    assert (tmp_path / "marginalpdbc_20251001.3").read_bytes() == BODY


def test_newer_version_replaces_the_old_file(tmp_path):
    (tmp_path / "marginalpdbc_20251001.1").write_bytes(BODY)
    v2 = BODY.replace(b";105.1;105.1;", b";99.5;99.5;")
    fake = FakeOmie({"marginalpdbc_20251001.2": v2})
    r = run(tmp_path, fake, date(2025, 10, 1), date(2025, 10, 1))
    assert fake.requested == ["marginalpdbc_20251001.1", "marginalpdbc_20251001.2"]
    assert r.updated == [(date(2025, 10, 1), 1, 2)] and not r.up_to_date and not r.errors
    assert names(tmp_path) == ["marginalpdbc_20251001.2"]
    assert (tmp_path / "marginalpdbc_20251001.2").read_bytes() == v2


def test_newer_version_found_past_a_404_version(tmp_path):
    # OMIE serves only the latest version: going from .1 to .3, .2 is a 404 as well.
    (tmp_path / "marginalpdbc_20251030.1").write_bytes(body(date(2025, 10, 30)))
    fake = FakeOmie({"marginalpdbc_20251030.3": body(date(2025, 10, 30))})
    r = run(tmp_path, fake, date(2025, 10, 30), date(2025, 10, 30))
    assert fake.requested == [f"marginalpdbc_20251030.{v}" for v in (1, 2, 3)]
    assert r.updated == [(date(2025, 10, 30), 1, 3)]
    assert names(tmp_path) == ["marginalpdbc_20251030.3"]


def test_day_on_disk_no_longer_served_and_nothing_newer_is_an_error(tmp_path):
    (tmp_path / "marginalpdbc_20251001.1").write_bytes(BODY)
    r = run(tmp_path, FakeOmie({}), date(2025, 10, 1), date(2025, 10, 1))
    assert [d for d, _ in r.errors] == [date(2025, 10, 1)]
    assert "no longer served" in r.errors[0][1]
    assert names(tmp_path) == ["marginalpdbc_20251001.1"]


def test_bad_newer_version_keeps_the_old_file(tmp_path):
    (tmp_path / "marginalpdbc_20251001.1").write_bytes(BODY)
    fake = FakeOmie({"marginalpdbc_20251001.2": BODY.replace(b"*", b"")})
    r = run(tmp_path, fake, date(2025, 10, 1), date(2025, 10, 1))
    assert r.errors and "missing '*' trailer" in r.errors[0][1] and not r.updated
    assert names(tmp_path) == ["marginalpdbc_20251001.1"]


def test_sleeps_between_consecutive_requests(tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr(omie.time, "sleep", lambda s: log.append(("sleep", s)))
    (tmp_path / "marginalpdbc_20251001.1").write_bytes(BODY)
    fake = FakeOmie(
        {"marginalpdbc_20251001.1": BODY, "marginalpdbc_20251002.2": body(date(2025, 10, 2))},
        log,
    )
    run(tmp_path, fake, date(2025, 10, 1), date(2025, 10, 2), pause=0.5)
    assert log == [
        ("get", "marginalpdbc_20251001.1"),
        ("sleep", 0.5),
        ("get", "marginalpdbc_20251002.1"),
        ("sleep", 0.5),
        ("get", "marginalpdbc_20251002.2"),
    ]


def test_future_day_is_reported_not_published(tmp_path):
    fake = FakeOmie({})  # every version 404
    r = run(tmp_path, fake, date(2026, 11, 8), date(2026, 11, 10))
    assert r.not_published == [date(2026, 11, 8), date(2026, 11, 9), date(2026, 11, 10)]
    assert not r.errors and not r.saved
    # Tomorrow is tried; days after it cannot be published yet and are not requested.
    assert fake.requested
    assert all(n.startswith("marginalpdbc_20261108.") for n in fake.requested)


def test_past_day_with_no_file_is_an_error(tmp_path):
    r = run(tmp_path, FakeOmie({}), date(2026, 11, 7), date(2026, 11, 7))
    assert r.errors and "all 404" in r.errors[0][1]


def test_http_failure_is_an_error_and_other_days_continue(tmp_path):
    err = urllib.error.HTTPError("u", 500, "Server Error", {}, None)
    fake = FakeOmie(
        {"marginalpdbc_20251001.1": err, "marginalpdbc_20251002.1": body(date(2025, 10, 2))}
    )
    r = run(tmp_path, fake, date(2025, 10, 1), date(2025, 10, 2))
    assert [d for d, _ in r.errors] == [date(2025, 10, 1)]
    assert "500" in r.errors[0][1]
    assert r.saved == [(date(2025, 10, 2), 1)]
    assert not (tmp_path / "marginalpdbc_20251001.1").exists()


def test_html_page_with_200_is_an_error_and_not_saved(tmp_path):
    fake = FakeOmie({"marginalpdbc_20251001.1": b"<!DOCTYPE html>"})
    r = run(tmp_path, fake, date(2025, 10, 1), date(2025, 10, 1))
    assert r.errors and "unexpected header" in r.errors[0][1]
    assert list(tmp_path.iterdir()) == []


def _drop_period_40(b):
    lines = b.splitlines(keepends=True)
    return b"".join(ln for ln in lines if not ln.startswith(b"2025;10;01;40;"))


@pytest.mark.parametrize(
    "bad, message",
    [
        (BODY[: BODY.index(b"*")], "missing '*' trailer"),
        (_drop_period_40(BODY), "wrong period count"),
    ],
    ids=["no-trailer", "missing-period"],
)
def test_cli_download_rejects_truncated_body(tmp_path, monkeypatch, capsys, bad, message):
    assert bad != BODY
    monkeypatch.setattr(omie, "http_get", FakeOmie({"marginalpdbc_20251001.1": bad}))
    monkeypatch.setattr(omie.time, "sleep", lambda s: None)
    argv = ["download", "omie", "--data-dir", str(tmp_path)]
    assert main([*argv, "--start", "2025-10-01", "--end", "2025-10-01"]) == 1
    assert message in capsys.readouterr().err
    assert list((tmp_path / "raw" / "omie").iterdir()) == []


def test_cli_download_exits_nonzero_on_http_failure(tmp_path, monkeypatch, capsys):
    def urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 503, "Unavailable", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(omie.time, "sleep", lambda s: None)
    argv = [
        "download",
        "omie",
        "--data-dir",
        str(tmp_path),
        "--start",
        "2025-10-01",
        "--end",
        "2025-10-02",
    ]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "ERROR 2025-10-01" in err and "ERROR 2025-10-02" in err and "503" in err


def test_cli_rejects_start_before_go_live(tmp_path):
    with pytest.raises(SystemExit, match="out of scope"):
        main(["download", "omie", "--data-dir", str(tmp_path), "--start", "2025-09-30"])


def test_user_agent_is_generic_and_url_has_no_identifying_data(monkeypatch):
    seen = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return BODY

    def urlopen(req, timeout):
        seen.append(req)
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert omie.http_get(omie.URL.format(name="marginalpdbc_20251001.1")) == BODY
    (req,) = seen
    assert req.header_items() == [("User-agent", "research-client/1.0")]
    assert req.full_url == (
        "https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_20251001.1"
    )
    assert req.data is None


def test_http_get_maps_404_to_none(monkeypatch):
    def urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert omie.http_get("https://example.invalid/x") is None
