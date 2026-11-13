"""Human-readable messages (Telegram HTML): a slip as read, and the weekly export summary."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from html import escape

from .config import Masters
from .models import UNKNOWN, Slip, Voucher

CONFIDENCE_MARK = {"high": "", "medium": " ❓", "low": " ❓❓"}


def inr(x: Decimal) -> str:
    """Format with Indian digit grouping: 1234567.5 -> ₹12,34,567.50."""
    sign = "-" if x < 0 else ""
    whole, frac = f"{abs(x):.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups) + "," + tail
    return f"{sign}₹{whole}.{frac}"


def _qty(q: float | Decimal) -> str:
    return f"{Decimal(str(q)).normalize():f}"


def _date(iso: str | None) -> str:
    try:
        return date.fromisoformat(iso).strftime("%d-%m-%Y")
    except (TypeError, ValueError):
        return f"❓ {iso or 'no date'}"


def slip_heading(slip: Slip) -> str:
    party = slip.party.strip() if slip.party != UNKNOWN else f"❓ {slip.party_raw or 'unknown customer'}"
    bits = [_date(slip.date)]
    if slip.challan_number:
        bits.append(f"challan {slip.challan_number}")
    return f"{party} · " + " · ".join(bits)


def render_slip(slip: Slip, masters: Masters, blockers: list[str], warnings: list[str],
                missing: list[str]) -> str:
    """The slip as read, for checking. No prices: those come from the rates sheet when the
    weekly file is made."""
    if not slip.is_slip:
        return "🤔 This doesn't look like an order slip. Please send a clear photo of the slip."

    meta = [f"📅 {escape(_date(slip.date))}"]
    if slip.challan_number:
        meta.append(f"Challan {escape(slip.challan_number)}")
    if slip.bags:
        meta.append(f"{slip.bags} bag{'s' if slip.bags != 1 else ''}")
    party = slip.party.strip() if slip.party != UNKNOWN else f"❓ {slip.party_raw or 'unknown customer'}"
    out = [f"🧾 <b>{escape(party)}</b>", " · ".join(meta), ""]

    for n, line in enumerate(slip.lines, 1):
        item = masters.item(line.item)
        name = item.name if item else "❓ not matched"
        unit = item.unit if item else "pcs"
        flag = " ⚠️ no rate" if item and item.name in missing else ""
        out.append(f"{n}. <b>{escape(name)}</b>: {_qty(line.quantity)} {escape(unit)}"
                   f"{CONFIDENCE_MARK[line.confidence]}{flag}")
        out.append(f"    <i>written: “{escape(line.item_text)}”</i>")

    if slip.notes:
        out += ["", f"📝 {escape(slip.notes)}"]
    if warnings:
        out += [""] + [f"⚠️ {escape(w)}" for w in warnings]
    if blockers:
        out += ["", "🛑 <b>Fix before saving:</b>"] + [f"• {escape(b)}" for b in blockers]

    out += ["", "To fix anything, reply to this message (type or send a voice note), e.g. <i>“line 2 is 300”</i> or <i>“date is 17/7”</i>."]
    return "\n".join(out)


def render_voucher(v: Voucher) -> str:
    """One priced invoice (used for the weekly summary and the command-line tool)."""
    head = f"<b>{escape(v.party.strip())}</b> · {v.date:%d-%m-%Y}"
    if v.challan_number:
        head += f" · challan {escape(v.challan_number)}"
    rows = [head]
    rows += [f"  {escape(l.item)}: {_qty(l.quantity)} {escape(l.unit)} × {inr(l.rate)} = {inr(l.amount)}"
             + ("" if l.rate_note.startswith("sheet") else f"  <i>({escape(l.rate_note)})</i>")
             for l in v.lines]
    rows += [f"  {escape(t.ledger)}: {inr(t.amount)}" for t in v.taxes]
    rows.append(f"  <b>Total {inr(v.total)}</b>")
    return "\n".join(rows)


def render_batch_summary(vouchers: list[Voucher]) -> str:
    total = sum((v.total for v in vouchers), Decimal("0"))
    out = [f"📦 <b>{len(vouchers)} invoice{'s' if len(vouchers) != 1 else ''}, total {inr(total)}</b>", ""]
    out += [render_voucher(v) + "\n" for v in vouchers]
    out.append("In Tally: Gateway → Import of Data → Vouchers → give the file's path.")
    return "\n".join(out)


def _file_time(name: str) -> str:
    m = re.match(r"Tally_Sales_(\d{4})-(\d\d)-(\d\d)_(\d\d)(\d\d)", name)
    return f"{m[3]}-{m[2]}-{m[1]} {m[4]}:{m[5]}" if m else name


def render_see(files, ready: list[tuple[object, Voucher]], held: list[tuple[object, list[str]]]) -> str:
    """/see: the latest Tally files with the challans in each, and what's waiting for the next one."""
    out = ["📦 <b>Tally files</b> (latest first)"]
    if not files:
        out.append("None yet.")
    for f in files:
        total = sum((s.total for s in f.slips), Decimal("0"))
        out += ["", f"<b>{_file_time(f.name)}</b> · {len(f.slips)} invoice{'s' if len(f.slips) != 1 else ''} · {inr(total)}"]
        out += [f"• {_date(s.date)} · Ch {escape(s.challan or '–')} · {escape(s.party.strip())} · {inr(s.total)}"
                + ("" if s.in_tally else "  ↩️ <i>taken back, not in Tally</i>") for s in f.slips]

    out += ["", f"⏳ <b>Waiting for the next file</b> ({len(ready) + len(held)})"]
    if not ready and not held:
        out.append("Nothing waiting.")
    out += [f"• {v.date:%d-%m-%Y} · Ch {escape(v.challan_number or '–')} · {escape(v.party.strip())} · {inr(v.total)}"
            for _, v in ready]
    out += [f"• {escape(_date(p.slip.date))} · Ch {escape(p.slip.challan_number or '–')} · "
            f"{escape(p.slip.party.strip())} · ⚠️ {escape(why[0] if why else 'not ready')}" for p, why in held]
    return "\n".join(out)
