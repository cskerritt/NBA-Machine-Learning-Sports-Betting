import threading
from datetime import datetime, timezone

import pytest

import sports_edge.models.train as train_mod
from sports_edge.scheduler import (
    needs_bootstrap,
    seconds_until_utc_hour,
    start_background_daily,
)


def _utc(h, m=0):
    return datetime(2026, 6, 12, h, m, tzinfo=timezone.utc)


def test_seconds_until_utc_hour():
    assert seconds_until_utc_hour(13, now=_utc(12, 30)) == 1800
    assert seconds_until_utc_hour(13, now=_utc(13)) == 86400   # next day
    assert seconds_until_utc_hour(3, now=_utc(23)) == 4 * 3600


def test_bootstrap_runs_immediately():
    """With an empty database the runner fires right away on startup."""
    ran = threading.Event()
    start_background_daily(run_at_utc_hour=0, sports=["mlb"],
                           runner=lambda sports: ran.set(),
                           bootstrap_check=lambda: True)
    assert ran.wait(timeout=5)


def test_no_bootstrap_waits_for_schedule():
    """With data present the runner must NOT fire before the scheduled hour."""
    ran = threading.Event()
    start_background_daily(run_at_utc_hour=0, sports=["mlb"],
                           runner=lambda sports: ran.set(),
                           bootstrap_check=lambda: False)
    assert not ran.wait(timeout=0.5)


def test_needs_bootstrap_empty_db(tmp_path, monkeypatch):
    import sports_edge.scheduler as sched
    from sports_edge.data.store import GameStore

    monkeypatch.setattr("sports_edge.data.store.DB_PATH", tmp_path / "g.sqlite")
    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path / "models")
    assert sched.needs_bootstrap() is True
