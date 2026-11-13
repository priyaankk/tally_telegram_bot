"""Runtime settings (from environment) and Tally masters (from TOML)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from datetime import time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


@dataclass(frozen=True)
class Item:
    name: str
    unit: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Party:
    ledger: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Tax:
    ledger: str
    percent: Decimal


@dataclass(frozen=True)
class Masters:
    company: str
    voucher_type: str
    sales_ledger: str
    godown: str
    parties: tuple[Party, ...]
    items: tuple[Item, ...]
    taxes: tuple[Tax, ...] = ()
    glossary: tuple[str, ...] = ()   # business shorthand that helps reading slips
    _items_by_name: dict[str, Item] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        self._items_by_name.update({i.name: i for i in self.items})

    def item(self, name: str) -> Item | None:
        return self._items_by_name.get(name)

    def party(self, ledger: str) -> Party | None:
        return next((p for p in self.parties if p.ledger == ledger), None)


def load_masters(path: Path, hints_path: Path | None = None) -> Masters:
    """Tally names from `path` (generated from a Tally export), plus optional hand-written
    hints (spellings seen on slips, glossary) from `hints_path`."""
    with path.open("rb") as f:
        raw = tomllib.load(f)
    hints: dict = {}
    if hints_path and hints_path.exists():
        with hints_path.open("rb") as f:
            hints = tomllib.load(f)
    item_aliases: dict[str, list[str]] = hints.get("item_aliases", {})
    party_aliases: dict[str, list[str]] = hints.get("party_aliases", {})

    items = tuple(
        Item(name=i["name"], unit=i["unit"],
             aliases=tuple(i.get("aliases", ())) + tuple(item_aliases.get(i["name"], ())))
        for i in raw.get("items", ())
    )
    names = [i.name for i in items]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate stock item names in {path}: {sorted(duplicates)}")
    if not items:
        raise ValueError(f"No [[items]] defined in {path}")
    parties = tuple(
        Party(ledger=p["ledger"],
              aliases=tuple(p.get("aliases", ())) + tuple(party_aliases.get(p["ledger"], ())))
        for p in raw.get("parties", ())
    )
    known = set(names) | {p.ledger for p in parties}
    unknown = (set(item_aliases) | set(party_aliases)) - known
    if unknown:
        raise ValueError(f"{hints_path} names things that aren't in Tally: {sorted(unknown)}")

    return Masters(
        company=raw.get("company", ""),
        voucher_type=raw.get("voucher_type", "Sales"),
        sales_ledger=raw["sales_ledger"],
        godown=raw.get("godown", ""),
        parties=parties,
        items=items,
        taxes=tuple(
            Tax(ledger=t["ledger"], percent=Decimal(str(t["percent"])))
            for t in raw.get("taxes", ())
        ),
        glossary=tuple(hints.get("glossary", ())),
    )


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    allowed_user_ids: frozenset[int]
    masters_path: Path
    hints_path: Path
    data_dir: Path
    store_backend: str            # "file" (data_dir) or "dynamodb"
    dynamodb_table: str
    rates_sheet_url: str          # Google Sheet with the rates; "" to use rates_csv instead
    google_credentials_file: Path
    rates_csv: Path
    batch_weekday: int            # weekly export day, Monday=0 … Saturday=5
    batch_time: time              # local time of the weekly export
    timezone: ZoneInfo
    openrouter_api_key: str
    llm_model: str
    llm_provider: str           # OpenRouter provider slug to pin, "" to let OpenRouter choose
    llm_reasoning_effort: str
    webapp_url: str = ""          # the /see page (Lambda only; on a computer /see answers in text)


def load_settings() -> Settings:
    load_dotenv()
    allowed = os.environ.get("ALLOWED_TELEGRAM_USER_IDS", "")
    return Settings(
        telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        allowed_user_ids=frozenset(int(x) for x in allowed.replace(" ", "").split(",") if x),
        masters_path=Path(os.environ.get("MASTERS_PATH", "config/masters.toml")),
        hints_path=Path(os.environ.get("HINTS_PATH", "config/hints.toml")),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        store_backend=os.environ.get("STORE", "file"),
        dynamodb_table=os.environ.get("DYNAMODB_TABLE", "tally-bot"),
        rates_sheet_url=os.environ.get("RATES_SHEET_URL", ""),
        google_credentials_file=Path(os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE",
                                                    "config/google-service-account.json")),
        rates_csv=Path(os.environ.get("RATES_CSV", "sheets/rates.csv")),
        batch_weekday=int(os.environ.get("BATCH_WEEKDAY", "5")),
        batch_time=time.fromisoformat(os.environ.get("BATCH_TIME", "21:00")),
        timezone=ZoneInfo(os.environ.get("TIMEZONE", "Asia/Kolkata")),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        # Gemini read Hindi slips far better than DeepSeek v4.1 Flash (which also looped on
        # them) and matched it on English ones; about ₹0.70 a slip.
        llm_model=os.environ.get("LLM_MODEL", "google/gemini-3.5-flash-lite"),
        llm_provider=os.environ.get("LLM_PROVIDER", ""),
        llm_reasoning_effort=os.environ.get("LLM_REASONING_EFFORT", "high"),
        webapp_url=os.environ.get("WEBAPP_URL", ""),
    )
