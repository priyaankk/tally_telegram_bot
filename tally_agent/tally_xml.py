"""Build a Tally import XML for Sales vouchers (item invoice), in the shape Tally 9 itself
exports: UTF-16 LE with BOM, CRLF line ends, no <?xml?> declaration.

Sign convention in Tally XML: debit amounts are negative and ISDEEMEDPOSITIVE=Yes,
credit amounts are positive and ISDEEMEDPOSITIVE=No. On a sales invoice the party is
debited with the total; the sales ledger (via the inventory allocations) and any tax
ledgers are credited.

Not set (left to Tally): the voucher number (Tally numbers Sales automatically) and
bill-wise allocations for the party.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from decimal import Decimal

from .config import Masters
from .models import Voucher


def _fmt_amount(x: Decimal) -> str:
    return f"{x:.2f}"


def _fmt_rate(rate: Decimal, unit: str) -> str:
    # At least 2 decimals like Tally ("2.50/PCS"), more only if the rate has them.
    text = f"{rate:.2f}" if rate == rate.quantize(Decimal("0.01")) else f"{rate.normalize():f}"
    return f"{text}/{unit}"


def _fmt_qty(qty: Decimal, unit: str) -> str:
    # Tally writes " 250 PCS"; normalize() turns 250.0 into 250.
    return f" {qty.normalize():f} {unit}"


def _sub(parent: ET.Element, tag: str, text: str | None = None) -> ET.Element:
    el = ET.SubElement(parent, tag)
    if text is not None:
        el.text = text
    return el


def _ledger_entry(parent: ET.Element, ledger: str, amount: Decimal, *, debit: bool,
                  is_party: bool = False) -> None:
    entry = _sub(parent, "LEDGERENTRIES.LIST")
    _sub(entry, "LEDGERNAME", ledger)
    _sub(entry, "ISDEEMEDPOSITIVE", "Yes" if debit else "No")
    _sub(entry, "ISPARTYLEDGER", "Yes" if is_party else "No")
    _sub(entry, "AMOUNT", _fmt_amount(-amount if debit else amount))


def build_voucher_element(voucher: Voucher, masters: Masters) -> ET.Element:
    v = ET.Element("VOUCHER", VCHTYPE=masters.voucher_type, ACTION="Create")
    tally_date = voucher.date.strftime("%Y%m%d")
    _sub(v, "DATE", tally_date)
    _sub(v, "VOUCHERTYPENAME", masters.voucher_type)
    if voucher.challan_number:
        _sub(v, "REFERENCE", voucher.challan_number)
    _sub(v, "PARTYLEDGERNAME", voucher.party)
    _sub(v, "BASICBUYERNAME", voucher.party.strip())
    _sub(v, "EFFECTIVEDATE", tally_date)
    _sub(v, "ISINVOICE", "Yes")
    _sub(v, "NARRATION", voucher.narration)

    _ledger_entry(v, voucher.party, voucher.total, debit=True, is_party=True)
    for tax in voucher.taxes:
        _ledger_entry(v, tax.ledger, tax.amount, debit=False)

    for line in voucher.lines:
        qty = _fmt_qty(line.quantity, line.unit)
        inv = _sub(v, "ALLINVENTORYENTRIES.LIST")
        _sub(inv, "STOCKITEMNAME", line.item)
        _sub(inv, "ISDEEMEDPOSITIVE", "No")
        _sub(inv, "RATE", _fmt_rate(line.rate, line.unit))
        _sub(inv, "AMOUNT", _fmt_amount(line.amount))
        _sub(inv, "ACTUALQTY", qty)
        _sub(inv, "BILLEDQTY", qty)
        alloc = _sub(inv, "ACCOUNTINGALLOCATIONS.LIST")
        _sub(alloc, "LEDGERNAME", masters.sales_ledger)
        _sub(alloc, "ISDEEMEDPOSITIVE", "No")
        _sub(alloc, "ISPARTYLEDGER", "No")
        _sub(alloc, "AMOUNT", _fmt_amount(line.amount))
        if masters.godown:
            batch = _sub(inv, "BATCHALLOCATIONS.LIST")
            _sub(batch, "GODOWNNAME", masters.godown)
            _sub(batch, "BATCHNAME", "Primary Batch")
            _sub(batch, "DESTINATIONGODOWNNAME", masters.godown)
            _sub(batch, "AMOUNT", _fmt_amount(line.amount))
            _sub(batch, "ACTUALQTY", qty)
            _sub(batch, "BILLEDQTY", qty)

    return v


def build_import_xml(vouchers: list[Voucher], masters: Masters) -> bytes:
    envelope = ET.Element("ENVELOPE")
    header = _sub(envelope, "HEADER")
    _sub(header, "TALLYREQUEST", "Import Data")
    import_data = _sub(_sub(envelope, "BODY"), "IMPORTDATA")
    desc = _sub(import_data, "REQUESTDESC")
    _sub(desc, "REPORTNAME", "Vouchers")
    if masters.company:
        _sub(_sub(desc, "STATICVARIABLES"), "SVCURRENTCOMPANY", masters.company)
    request_data = _sub(import_data, "REQUESTDATA")
    for voucher in vouchers:
        msg = _sub(request_data, "TALLYMESSAGE")
        msg.set("xmlns:UDF", "TallyUDF")
        msg.append(build_voucher_element(voucher, masters))

    ET.indent(envelope)
    text = ET.tostring(envelope, encoding="unicode").replace("\n", "\r\n") + "\r\n"
    return b"\xff\xfe" + text.encode("utf-16-le")
