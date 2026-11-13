"""Generate config/masters.toml from Tally's own XML exports.

    uv run python -m tally_agent.tally_import Master.xml [DayBook.xml]

Master.xml: Display → List of Accounts → Alt+E → XML (Data Interchange), All Masters.
DayBook.xml (optional): used only to detect which ledger sales are booked to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .tally_data import (export_kind, masters_toml, most_common_sales_ledger, parse_tally_xml,
                         read_masters, read_sales)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("config/masters.toml"))
    args = parser.parse_args()

    masters = None
    sales = []
    for path in args.exports:
        root = parse_tally_xml(path.read_bytes())
        kind = export_kind(root)
        print(f"{path}: {kind}")
        if kind == "masters":
            masters = read_masters(root)
        elif kind == "vouchers":
            sales += read_sales(root)
    if masters is None:
        raise SystemExit("No masters export given.")

    sales_ledger = most_common_sales_ledger(sales) or next(
        (l for l in masters.sales_ledgers if "RETURN" not in l.upper()), "")
    if not sales_ledger:
        raise SystemExit("Couldn't work out the sales ledger; pass a Day Book export too.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(masters_toml(masters, sales_ledger))
    print(f"Wrote {args.out}: company {masters.company!r}, {len(masters.items)} items, "
          f"{len(masters.parties)} customers, sales ledger {sales_ledger!r}, "
          f"godown {masters.godowns[0] if len(masters.godowns) == 1 else '-'!r}")


if __name__ == "__main__":
    main()
