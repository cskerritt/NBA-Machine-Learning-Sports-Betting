"""In-process daily scheduler.

Lets a single hosted service (Railway, Docker, any PaaS) keep its data and
models fresh without a separate worker or cron service: the web process
bootstraps itself on first boot (empty database / missing models) and then
re-runs the daily pipeline every day at a fixed UTC hour.
"""

import os
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

DEFAULT_UTC_HOUR = 13  # ~9am ET: previous night's games are final


def seconds_until_utc_hour(hour: int, now: datetime | None = None) -> float:
    """Seconds from `now` until the next occurrence of hour:00 UTC."""
    now = now or datetime.now(timezone.utc)
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def needs_bootstrap() -> bool:
    """True when any sport has no games or no trained model yet."""
    from sports_edge.config import SPORTS
    from sports_edge.data.store import GameStore
    from sports_edge.models.train import model_path

    store = GameStore()
    try:
        return any(store.count(key) == 0 or not model_path(cfg).exists()
                   for key, cfg in SPORTS.items())
    finally:
        store.close()


def _run_once(sports: list[str] | None):
    from sports_edge.daily import run_daily

    print("[scheduler] running daily pipeline...", flush=True)
    try:
        for r in run_daily(sports=sports):
            print(f"[scheduler] {r.sport}: {r.status}"
                  + (f" ({r.error})" if r.error else ""), flush=True)
    except Exception:
        traceback.print_exc()


def _loop(run_at_utc_hour: int, sports: list[str] | None,
          runner=_run_once, bootstrap_check=None):
    if (bootstrap_check or needs_bootstrap)():
        runner(sports)
    while True:
        time.sleep(seconds_until_utc_hour(run_at_utc_hour))
        runner(sports)


def start_background_daily(run_at_utc_hour: int | None = None,
                           sports: list[str] | None = None,
                           runner=_run_once,
                           bootstrap_check=None) -> threading.Thread:
    """Start the daemon thread. Hour comes from SPORTS_EDGE_DAILY_UTC_HOUR
    when not given explicitly."""
    if run_at_utc_hour is None:
        run_at_utc_hour = int(os.environ.get("SPORTS_EDGE_DAILY_UTC_HOUR",
                                             DEFAULT_UTC_HOUR))
    thread = threading.Thread(
        target=_loop, args=(run_at_utc_hour, sports, runner, bootstrap_check),
        daemon=True, name="sports-edge-daily")
    thread.start()
    return thread
