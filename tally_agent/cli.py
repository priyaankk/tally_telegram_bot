"""Run the slip → Tally XML pipeline on image files, without Telegram.

    uv run python -m tally_agent.cli photo1.jpg photo2.jpg --out out/

With --check DayBook.xml, each reading is compared with the invoice actually entered in
Tally for that customer and challan number (the Ref starts with the challan number).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

from .config import load_masters, load_settings
from .extract import SlipReader, load_image
from .models import Slip, apply_learned, check_slip, missing_rates, price_slip, unusual_items
from .preview import render_slip, render_voucher
from .rates import customer_items, make_rate_source
from .store import make_store, xml_filename
from .tally_data import _objects, parse_tally_xml
from .tally_xml import build_import_xml


def _plain(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


def tally_invoices(daybook: Path) -> dict[tuple[str, str], list[dict]]:
    """(party, challan number) -> [{"date": iso, "lines": Counter{(item, qty)}}, ...]
    A list because challan books restart, so the same number recurs months apart."""
    out: dict[tuple[str, str], list[dict]] = {}
    for v in _objects(parse_tally_xml(daybook.read_bytes()), "VOUCHER"):
        ref = (v.findtext("REFERENCE") or "").split("/")[0].strip()
        if v.get("VCHTYPE") != "Sales" or not ref:
            continue
        lines = Counter()
        for inv in v.findall("ALLINVENTORYENTRIES.LIST"):
            qty = float((inv.findtext("BILLEDQTY") or "0").split()[0])
            lines[(inv.findtext("STOCKITEMNAME"), qty)] += 1
        d = v.findtext("DATE")
        out.setdefault((v.findtext("PARTYLEDGERNAME"), ref), []).append(
            {"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "lines": lines})
    return out


def compare(slip: Slip, invoices: dict) -> tuple[int, int, list[str]]:
    """(correct lines, total lines in Tally, differences)"""
    candidates = invoices.get((slip.party, slip.challan_number or ""))
    if not candidates:
        return 0, 0, [f"no Tally invoice for {slip.party!r} challan {slip.challan_number!r}"]
    # Same challan number can recur; take the invoice whose items match best.
    read = Counter((l.item, l.quantity) for l in slip.lines)
    truth = max(candidates, key=lambda c: sum((read & c["lines"]).values()))
    diffs = []
    if slip.date != truth["date"]:
        diffs.append(f"date: read {slip.date}, Tally {truth['date']}")
    correct = sum((read & truth["lines"]).values())
    diffs += [f"read but not in Tally: {k[0]} × {k[1]:g}" for k in (read - truth["lines"])]
    diffs += [f"in Tally but not read: {k[0]} × {k[1]:g}" for k in (truth["lines"] - read)]
    return correct, sum(truth["lines"].values()), diffs


async def run(paths: list[Path], out: Path, check: Path | None) -> None:
    settings = load_settings()
    masters = load_masters(settings.masters_path, settings.hints_path)
    rates = make_rate_source(settings, masters)
    store = make_store(settings)
    reader = SlipReader(masters, settings, learned_spellings=store.learned_spellings,
                        customer_items=lambda: customer_items(rates.table()))
    out.mkdir(parents=True, exist_ok=True)
    invoices = tally_invoices(check) if check else None

    results = await asyncio.gather(*(reader.read(load_image(p.read_bytes())) for p in paths))
    total_correct = total_lines = 0
    for path, (_, slip) in zip(paths, results):
        print(f"\n===== {path.name} =====")
        slip = apply_learned(slip, store.learned_aliases(), masters)
        table = rates.table()
        blockers, warnings = check_slip(slip, masters)
        warnings += unusual_items(slip, customer_items(table))
        print(_plain(render_slip(slip, masters, blockers, warnings, missing_rates(slip, table))))
        voucher, problems = price_slip(slip, masters, table)
        print("\n--- priced from the rates sheet ---")
        print(_plain(render_voucher(voucher)) if voucher else "\n".join(f"✗ {p}" for p in problems))
        (out / f"{path.stem}.json").write_text(json.dumps(slip.model_dump(), ensure_ascii=False, indent=2))
        if voucher:
            xml_path = out / xml_filename(voucher)
            xml_path.write_bytes(build_import_xml([voucher], masters))
            print(f"\n→ wrote {xml_path}")
        if invoices is not None:
            correct, n, diffs = compare(slip, invoices)
            total_correct, total_lines = total_correct + correct, total_lines + n
            print(f"\n🔎 vs Tally: {correct}/{n} lines exactly right" + "".join(f"\n   ✗ {d}" for d in diffs))
    if invoices is not None:
        print(f"\n===== Overall: {total_correct}/{total_lines} lines match Tally =====")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--check", type=Path, help="Tally Day Book export to compare against")
    args = parser.parse_args()
    asyncio.run(run(args.images, args.out, args.check))


if __name__ == "__main__":
    main()
