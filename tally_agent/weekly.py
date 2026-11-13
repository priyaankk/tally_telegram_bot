"""The weekly export: every confirmed slip is priced from the rates sheet as it stands at the
scheduled time (Saturday night by default) and sent as one Tally XML file.

If the bot was off at that time, the run happens as soon as it's back."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from .config import Masters
from .models import Voucher, price_slip
from .rates import Rate
from .store import Pending


def last_scheduled(now: datetime, weekday: int, at: time) -> datetime:
    """The most recent scheduled moment at or before `now` (weekday: Monday=0 … Sunday=6)."""
    days_back = (now.weekday() - weekday) % 7
    candidate = datetime.combine(now.date() - timedelta(days=days_back), at, tzinfo=now.tzinfo)
    if candidate > now:
        candidate -= timedelta(days=7)
    return candidate


def is_due(now: datetime, last_run: datetime | None, weekday: int, at: time) -> bool:
    due = last_scheduled(now, weekday, at)
    return last_run is None or last_run < due


@dataclass(frozen=True)
class Batch:
    ready: list[tuple[Pending, Voucher]]
    held: list[tuple[Pending, list[str]]]


def build_batch(pending: list[Pending], masters: Masters, rates: dict[tuple[str, str], Rate]) -> Batch:
    ready, held = [], []
    for p in pending:
        voucher, problems = price_slip(p.slip, masters, rates)
        if voucher is None:
            held.append((p, problems))
        else:
            ready.append((p, voucher))
    ready.sort(key=lambda pv: (pv[1].date, *_challan_order(pv[1].challan_number), pv[1].party))
    return Batch(ready, held)


def _challan_order(number: str | None) -> tuple[float, str]:
    """Challan numbers in number order (9 before 10); slips without one go last in their day."""
    digits = re.match(r"\s*(\d+)", number or "")
    return (int(digits.group(1)) if digits else float("inf"), number or "")


def batch_name(now: datetime) -> str:
    return f"Tally_Sales_{now:%Y-%m-%d_%H%M}"
