"""Customer rates, maintained by a person in a sheet: Customer | Item | Rate | Last updated | Notes.

The sheet is the only source of rates. Names in the sheet are matched to Tally's exact
names ignoring case and spacing ("ACME TRADERS" == Tally's "ACME TRADERS ").
Backends: a local CSV file (for testing / before Google access is set up) and Google Sheets.
"""

from __future__ import annotations

import csv
import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from .config import Masters, Settings

log = logging.getLogger(__name__)

HEADER = ["Customer", "Item", "Rate", "Last updated", "Notes"]


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().casefold()


@dataclass(frozen=True)
class Rate:
    rate: Decimal
    updated: str      # as written in the sheet, e.g. "09-10-2026"


class RateSource(Protocol):
    def table(self) -> dict[tuple[str, str], Rate]:
        """(Tally party ledger, Tally item name) -> rate. Re-read on every call."""
        ...

    def set_rate(self, party: str, item: str, rate: Decimal) -> None:
        """Add or update one row."""
        ...


def rows_to_table(rows: list[list[str]], masters: Masters) -> dict[tuple[str, str], Rate]:
    parties = {_norm(p.ledger): p.ledger for p in masters.parties}
    items = {_norm(i.name): i.name for i in masters.items}
    table: dict[tuple[str, str], Rate] = {}
    for n, row in enumerate(rows[1:], 2):  # row 1 is the header
        row = (row + [""] * 5)[:5]
        customer, item, rate_text, updated = row[0], row[1], row[2].strip(), row[3]
        if not customer.strip() and not item.strip():
            continue
        party = parties.get(_norm(customer))
        stock = items.get(_norm(item))
        try:
            rate = Decimal(rate_text.replace(",", "").replace("₹", "")) if rate_text else None
        except InvalidOperation:
            rate = None
        if party is None or stock is None or rate is None:
            log.warning("Rates sheet row %d ignored (unknown customer/item or bad rate): %s", n, row[:3])
            continue
        table[(party, stock)] = Rate(rate, updated)
    return table


def _today() -> str:
    return date.today().strftime("%d-%m-%Y")


class CsvRates:
    def __init__(self, path: Path, masters: Masters) -> None:
        self.path, self.masters = path, masters

    def _rows(self) -> list[list[str]]:
        if not self.path.exists():
            return [HEADER]
        with self.path.open(newline="", encoding="utf-8") as f:
            return list(csv.reader(f))

    def table(self) -> dict[tuple[str, str], Rate]:
        return rows_to_table(self._rows(), self.masters)

    def set_rate(self, party: str, item: str, rate: Decimal) -> None:
        rows = self._rows()
        for row in rows[1:]:
            if _norm(row[0]) == _norm(party) and _norm(row[1]) == _norm(item):
                row[2:4] = [f"{rate:f}", _today()]
                break
        else:
            rows.append([party.strip(), item.strip(), f"{rate:f}", _today(), "added from Telegram"])
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        tmp.replace(self.path)


class GoogleSheetRates:
    """The 'Rates' tab of a Google Sheet, shared with a service account (Editor).
    Reads are cached briefly so one slip doesn't hit the API several times."""

    def __init__(self, url: str, credentials_file: Path, masters: Masters,
                 tab: str = "Rates", cache_seconds: float = 20) -> None:
        import gspread  # only needed when a sheet is configured

        self.ws = gspread.service_account(filename=str(credentials_file)).open_by_url(url).worksheet(tab)
        self.masters = masters
        self.cache_seconds = cache_seconds
        self._cached: tuple[float, list[list[str]]] | None = None

    def _rows(self) -> list[list[str]]:
        now = time.monotonic()
        if self._cached is None or now - self._cached[0] > self.cache_seconds:
            self._cached = (now, self.ws.get_all_values())
        return self._cached[1]

    def table(self) -> dict[tuple[str, str], Rate]:
        return rows_to_table(self._rows(), self.masters)

    def set_rate(self, party: str, item: str, rate: Decimal) -> None:
        self._cached = None
        rows = self._rows()
        for n, row in enumerate(rows[1:], 2):
            row = row + ["", ""]
            if _norm(row[0]) == _norm(party) and _norm(row[1]) == _norm(item):
                self.ws.update(range_name=f"C{n}:D{n}", values=[[f"{rate:f}", _today()]],
                               value_input_option="USER_ENTERED")
                break
        else:
            self.ws.append_row([party.strip(), item.strip(), f"{rate:f}", _today(), "added from Telegram"],
                               value_input_option="USER_ENTERED")
        self._cached = None


def make_rate_source(settings: Settings, masters: Masters) -> RateSource:
    if settings.rates_sheet_url:
        return GoogleSheetRates(settings.rates_sheet_url, settings.google_credentials_file, masters)
    return CsvRates(settings.rates_csv, masters)


def customer_items(table: dict[tuple[str, str], Rate]) -> dict[str, list[str]]:
    """customer -> items they currently buy (= have a rate for)."""
    out: dict[str, list[str]] = {}
    for party, item in table:
        out.setdefault(party, []).append(item)
    return out
