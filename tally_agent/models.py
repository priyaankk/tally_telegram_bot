"""What Claude reads off a slip, and the priced voucher we build from it."""

from __future__ import annotations

import calendar
import difflib
import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pydantic import BaseModel

from .config import Masters
from .rates import Rate

UNKNOWN = "UNKNOWN"
PAISE = Decimal("0.01")


class SlipLine(BaseModel):
    item_text: str                # item as written, without the quantity, e.g. "Hinge H4"
    quantity: float
    item: str                     # a stock item name from the masters, or UNKNOWN
    confidence: Literal["high", "medium", "low"]
    rate_override: float | None   # only when a rate is written on the slip or the user gives one


class Slip(BaseModel):
    is_slip: bool                 # False if the photo isn't an order/delivery slip at all
    party: str                    # a party ledger from the masters, or UNKNOWN
    party_raw: str
    date: str | None              # ISO YYYY-MM-DD
    challan_number: str | None    # the number PRINTED on the slip pad page
    bags: int | None
    lines: list[SlipLine]
    notes: str                    # anything unclear, for the human to check


def normalize_text(text: str) -> str:
    """Spelling key for learned aliases: "Hinge H-4" and "hinge h4" are the same."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def learned_key(party: str, item_text: str) -> str:
    return f"{party}\u241f{normalize_text(item_text)}"   # separator can't appear in Tally names


def apply_learned(slip: Slip, learned: dict[str, str], masters: Masters) -> Slip:
    """Override the model's match with what a person confirmed for this exact spelling
    from this same customer before ("Pan Screw" can mean different items for different customers)."""
    lines = []
    for line in slip.lines:
        item = learned.get(learned_key(slip.party, line.item_text))
        if item and masters.item(item) and item != line.item:
            line = line.model_copy(update={"item": item, "confidence": "high"})
        elif item == line.item:
            line = line.model_copy(update={"confidence": "high"})
        lines.append(line)
    return slip.model_copy(update={"lines": lines})


def slip_json_schema(masters: Masters) -> dict:
    """JSON schema for structured output, with party/item constrained to the masters."""
    nullable = lambda t: {"anyOf": [{"type": t}, {"type": "null"}]}  # noqa: E731
    line = {
        "type": "object",
        "properties": {
            "item_text": {"type": "string"},
            "quantity": {"type": "number"},
            "item": {"type": "string", "enum": [i.name for i in masters.items] + [UNKNOWN]},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "rate_override": nullable("number"),
        },
        "required": ["item_text", "quantity", "item", "confidence", "rate_override"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "is_slip": {"type": "boolean"},
            "party": {"type": "string", "enum": [p.ledger for p in masters.parties] + [UNKNOWN]},
            "party_raw": {"type": "string"},
            "date": nullable("string"),
            "challan_number": nullable("string"),
            "bags": nullable("integer"),
            "lines": {"type": "array", "items": line},
            "notes": {"type": "string"},
        },
        "required": ["is_slip", "party", "party_raw", "date", "challan_number", "bags", "lines", "notes"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class PricedLine:
    item: str
    quantity: Decimal
    unit: str
    rate: Decimal
    amount: Decimal
    rate_note: str    # where the rate came from, shown in the preview


@dataclass(frozen=True)
class TaxLine:
    ledger: str
    percent: Decimal
    amount: Decimal


@dataclass(frozen=True)
class Voucher:
    party: str
    date: date
    challan_number: str | None
    narration: str
    lines: tuple[PricedLine, ...]
    taxes: tuple[TaxLine, ...]

    @property
    def subtotal(self) -> Decimal:
        return sum((l.amount for l in self.lines), Decimal("0"))

    @property
    def total(self) -> Decimal:
        return self.subtotal + sum((t.amount for t in self.taxes), Decimal("0"))


def _money(x: Decimal) -> Decimal:
    return x.quantize(PAISE, rounding=ROUND_HALF_UP)


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().casefold()


class Names:
    """Exact Tally names from what the model wrote (case, spacing and known aliases ignored)."""

    def __init__(self, masters: Masters) -> None:
        self.parties = {_norm_name(n): p.ledger for p in masters.parties for n in (p.ledger, *p.aliases)}
        self.items = {_norm_name(n): i.name for i in masters.items for n in (i.name, *i.aliases)}

    @staticmethod
    def _find(name: str, names: dict[str, str], kind: str) -> str:
        if (exact := names.get(_norm_name(name))) is not None:
            return exact
        close = difflib.get_close_matches(_norm_name(name), list(names), n=3, cutoff=0.5)
        hint = f" Closest: {', '.join(names[c].strip() for c in close)}." if close else ""
        raise ValueError(f"There is no {kind} called “{name}”.{hint}")

    def party(self, name: str) -> str:
        return self._find(name, self.parties, "customer")

    def item(self, name: str) -> str:
        return self._find(name, self.items, "stock item")


def financial_year(today: date) -> tuple[date, date]:
    """The Indian financial year (1 April to 31 March) that today falls in."""
    start = date(today.year if today.month >= 4 else today.year - 1, 4, 1)
    return start, date(start.year + 1, 3, 31)


def a_month_before(today: date) -> date:
    year, month = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    return date(year, month, min(today.day, calendar.monthrange(year, month)[1]))


def check_slip(slip: Slip, masters: Masters, today: date | None = None,
               date_sanity: bool = True) -> tuple[list[str], list[str]]:
    """Problems that don't depend on rates. Returns (blockers, warnings): blockers stop the
    slip from being saved/exported; warnings are shown for the user to double-check.
    date_sanity compares the slip date with today (off when re-checking saved slips)."""
    if not slip.is_slip:
        return ["This doesn't look like an order/delivery slip."], []
    blockers: list[str] = []
    warnings: list[str] = []
    if slip.party == UNKNOWN or masters.party(slip.party) is None:
        blockers.append(f"Customer “{slip.party_raw or '?'}” is not in Tally's customer list.")

    if not slip.date:
        blockers.append("Date not found on the slip. Reply with the date, e.g. “date is 18/8”.")
    else:
        try:
            d = date.fromisoformat(slip.date)
        except ValueError:
            blockers.append(f"Date “{slip.date}” is not valid.")
        else:
            # Only slips from the last month, in this financial year, are saved: an older date is
            # almost always a misread, and Tally refuses a voucher outside the company's year
            # (which also stops the rest of the file importing).
            today = today or date.today()
            fy_start, fy_end = financial_year(today)
            since = max(a_month_before(today), fy_start)
            fix = "Reply with the right date if I read it wrong."
            if not date_sanity:
                pass
            elif d > today:
                blockers.append(f"Date {d:%d-%m-%Y} is in the future, so it can't be saved. {fix}")
            elif d < fy_start:
                blockers.append(f"Date {d:%d-%m-%Y} is not in this financial year ({fy_start:%d-%m-%Y} to "
                                f"{fy_end:%d-%m-%Y}), so it can't be saved. {fix}")
            elif d < since:
                blockers.append(f"Date {d:%d-%m-%Y} is more than a month ago, so it can't be saved: only "
                                f"slips dated {since:%d-%m-%Y} or later. {fix}")

    if not slip.lines:
        blockers.append("No item lines found.")
    for n, line in enumerate(slip.lines, 1):
        if masters.item(line.item) is None:
            blockers.append(f"Line {n}: “{line.item_text}” doesn't match any stock item. "
                            f"Reply with the item, e.g. “line {n} is S3 SCREW”.")
        if line.quantity <= 0:
            blockers.append(f"Line {n}: quantity {line.quantity:g} is not valid.")
    return blockers, warnings


def missing_rates(slip: Slip, rates: dict[tuple[str, str], Rate]) -> list[str]:
    """Stock items on the slip that have neither a sheet rate for this customer nor a rate given.
    Unmatched items and an unknown customer are reported by check_slip, not here."""
    if slip.party == UNKNOWN:
        return []
    return list(dict.fromkeys(
        line.item for line in slip.lines
        if line.item != UNKNOWN and line.rate_override is None and (slip.party, line.item) not in rates
    ))


def price_slip(slip: Slip, masters: Masters,
               rates: dict[tuple[str, str], Rate]) -> tuple[Voucher | None, list[str]]:
    """Price each line from the customer's rates sheet (or a rate the user gave).
    Returns (voucher, problems); voucher is None if anything blocks export."""
    problems, _ = check_slip(slip, masters, date_sanity=False)   # checked when it was saved
    problems += [f"No rate for {item} for {slip.party.strip()} in the rates sheet."
                 for item in missing_rates(slip, rates)]
    if problems:
        return None, problems

    priced: list[PricedLine] = []
    for line in slip.lines:
        item = masters.item(line.item)
        qty = Decimal(str(line.quantity))
        if line.rate_override is not None:
            rate, note = Decimal(str(line.rate_override)), "rate you gave"
        else:
            sheet = rates[(slip.party, item.name)]
            rate, note = sheet.rate, f"sheet, {sheet.updated}" if sheet.updated else "sheet"
        priced.append(PricedLine(item.name, qty, item.unit, rate, _money(qty * rate), note))

    subtotal = sum((l.amount for l in priced), Decimal("0"))
    taxes = tuple(
        TaxLine(t.ledger, t.percent, _money(subtotal * t.percent / 100)) for t in masters.taxes
    )
    narration_bits = [f"Challan {slip.challan_number}" if slip.challan_number else None,
                      f"{slip.bags} bag(s)" if slip.bags else None,
                      "via Telegram bot"]
    return Voucher(
        party=slip.party,
        date=date.fromisoformat(slip.date),
        challan_number=slip.challan_number,
        narration=", ".join(b for b in narration_bits if b),
        lines=tuple(priced),
        taxes=taxes,
    ), []


def _similar_items(name: str, candidates: list[str], n: int = 3) -> list[str]:
    """Candidates sharing most of `name`'s words ("6MM HOOK" -> "6MM BIG HOOK",
    "B 1 BOLT" -> "B 1 BOLT (STEEL)" before "B 9 BOLT")."""
    words = lambda s: set(re.findall(r"[A-Z0-9/]+", s.upper()))  # noqa: E731
    target = words(name)
    scored = []
    for c in candidates:
        coverage = len(target & words(c)) / max(len(target), 1)
        if coverage >= 0.5:
            scored.append((coverage, difflib.SequenceMatcher(None, name, c).ratio(), c))
    return [c for *_, c in sorted(scored, reverse=True)[:n]]


def unusual_items(slip: Slip, customer_items: dict[str, list[str]]) -> list[str]:
    """Warnings for lines whose item this customer doesn't currently buy (per the rates sheet).
    A misread variant ("6MM HOOK" instead of "6MM BIG HOOK") usually shows up this way."""
    usual = customer_items.get(slip.party)
    if not usual:
        return []   # new customer: nothing to compare with
    warnings = []
    for n, line in enumerate(slip.lines, 1):
        if line.item == UNKNOWN or line.item in usual:
            continue
        similar = _similar_items(line.item, usual)
        hint = f" (they usually get {', '.join(similar)})" if similar else ""
        warnings.append(f"Line {n}: {slip.party.strip()} hasn't bought {line.item} before{hint}. Please check.")
    return warnings
