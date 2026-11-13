"""Read Tally's own XML exports: masters (items, parties, ledgers) and sales vouchers."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

# Tally emits character references that XML 1.0 forbids (e.g. &#4;); drop them.
_BAD_CHAR_REFS = re.compile(r"&#(?:[0-8]|1[124-9]|2[0-9]|3[01]);")


def parse_tally_xml(data: bytes) -> ET.Element:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = data.decode("utf-16")
    else:
        text = data.decode("utf-8-sig")
    return ET.fromstring(_BAD_CHAR_REFS.sub("", text))


def _objects(root: ET.Element, tag: str) -> list[ET.Element]:
    return [child for msg in root.iter("TALLYMESSAGE") for child in msg if child.tag == tag]


def export_kind(root: ET.Element) -> str:
    """'masters', 'vouchers', or 'unknown' (e.g. an on-screen report export)."""
    if _objects(root, "STOCKITEM") or _objects(root, "LEDGER"):
        return "masters"
    if _objects(root, "VOUCHER"):
        return "vouchers"
    return "unknown"


# -- masters -------------------------------------------------------------------

@dataclass(frozen=True)
class TallyMasters:
    company: str
    items: list[tuple[str, str]]          # (name, unit)
    parties: list[str]                    # Sundry Debtors ledgers
    sales_ledgers: list[str]
    godowns: list[str]


def read_masters(root: ET.Element) -> TallyMasters:
    company = root.findtext(".//SVCURRENTCOMPANY") or ""
    items = [(e.get("NAME"), (e.findtext("BASEUNITS") or "").strip())
             for e in _objects(root, "STOCKITEM")]
    ledgers = _objects(root, "LEDGER")
    by_group = lambda g: [e.get("NAME") for e in ledgers if e.findtext("PARENT") == g]  # noqa: E731
    return TallyMasters(
        company=company,
        items=[(n, u) for n, u in items if n and u],   # items without a unit can't be invoiced
        parties=by_group("Sundry Debtors"),
        sales_ledgers=by_group("Sales Accounts"),
        godowns=[e.get("NAME") for e in _objects(root, "GODOWN")],
    )


def _toml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)  # JSON string escapes are valid TOML basic strings


def masters_toml(m: TallyMasters, sales_ledger: str) -> str:
    """config/masters.toml generated from Tally. Don't edit by hand: re-import instead."""
    lines = [
        "# GENERATED from a Tally masters export by `python -m tally_agent.tally_import`.",
        "# Do not edit: names must match Tally exactly (including trailing spaces).",
        "# Hand-written hints (spellings, glossary) go in config/hints.toml.",
        f"# Generated {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        f"company = {_toml_str(m.company)}",
        'voucher_type = "Sales"',
        f"sales_ledger = {_toml_str(sales_ledger)}",
        f"godown = {_toml_str(m.godowns[0] if len(m.godowns) == 1 else '')}",
        "",
    ]
    for p in m.parties:
        lines += ["[[parties]]", f"ledger = {_toml_str(p)}", ""]
    for name, unit in m.items:
        lines += ["[[items]]", f"name = {_toml_str(name)}", f"unit = {_toml_str(unit)}", ""]
    return "\n".join(lines)


# -- sales vouchers → rate history -------------------------------------------------

@dataclass(frozen=True)
class SaleLine:
    date: date
    party: str
    item: str
    rate: Decimal
    voucher_number: str
    sales_ledger: str


def _rate(text: str | None) -> Decimal | None:
    # "2.50/PCS" -> 2.50
    if not text:
        return None
    try:
        return Decimal(text.split("/")[0].strip())
    except InvalidOperation:
        return None


def read_sales(root: ET.Element) -> list[SaleLine]:
    out = []
    for v in _objects(root, "VOUCHER"):
        if v.findtext("ISCANCELLED") == "Yes" or v.findtext("ISOPTIONAL") == "Yes":
            continue
        if (v.get("VCHTYPE") or v.findtext("VOUCHERTYPENAME")) != "Sales":
            continue
        d = datetime.strptime(v.findtext("DATE"), "%Y%m%d").date()
        party = v.findtext("PARTYLEDGERNAME") or ""
        for inv in v.findall("ALLINVENTORYENTRIES.LIST"):
            rate = _rate(inv.findtext("RATE"))
            if rate is None or rate <= 0:
                continue
            out.append(SaleLine(
                date=d, party=party, item=inv.findtext("STOCKITEMNAME") or "", rate=rate,
                voucher_number=(v.findtext("VOUCHERNUMBER") or "").strip(),
                sales_ledger=inv.findtext("ACCOUNTINGALLOCATIONS.LIST/LEDGERNAME") or "",
            ))
    return out


def most_common_sales_ledger(sales: list[SaleLine]) -> str | None:
    counts = Counter(s.sales_ledger for s in sales if s.sales_ledger)
    return counts.most_common(1)[0][0] if counts else None
