"""The assistant: typed or spoken requests that aren't a reply to a slip, handled by an LLM with
business tools. Tools only look things up or *propose* a change. The bot shows each proposal
with buttons, and nothing changes until the owner taps one."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import openai

from .config import Masters, Settings
from .extract import ExtractionError
from .models import Names, Slip
from .preview import _file_time
from .rates import Rate
from .store import SlipRecord, _challan

log = logging.getLogger(__name__)

MAX_STEPS = 6            # model turns per request: a lookup or two, then an answer or proposal
MAX_RATE_ROWS = 80       # rows one lookup returns (a customer has a few dozen at most)
MAX_RATE = Decimal(100000)
NON_LATIN = re.compile(r"[\u0900-\u0AFF]")   # Devanagari and Gujarati script
MAX_CHANGES = 60         # per proposal, so the confirmation fits in one Telegram message

SYSTEM_PROMPT = """\
You are the assistant in a Telegram bot for a small manufacturing business in India (Tally \
company "{company}"). The owner writes or speaks in Gujarati, Hindi or English, often with item \
codes such as S5 or H4.

What you can do with your tools:
- Look up rates in the rates sheet (find_rates).
- Find slips sent to the bot and say exactly where each one is (find_slips): read but not saved \
yet, saved and waiting for the next Tally file, in a Tally file (say which: its date and time), \
or taken back out of a file because it didn't import. The bot only knows slips sent to it; \
anything older or entered straight into Tally isn't there, so say "not found in the bot", never \
"not in Tally".
- Change or add rates (propose_rate_changes). The bot shows the change to the owner with \
buttons and only writes the sheet when they confirm, so propose straight away; don't ask \
"are you sure?" first.
{slip_tool}
Rules:
- Rates are per customer and per item, in rupees per unit. Use the exact customer and item \
names from the lists below.
- If the request leaves it unclear which customer, which item or what rate, ask one short \
question instead of guessing.
- For "all customers" or a percentage change, look up the current rates first, then propose \
the exact new rates (2 decimals).
- To change or add a rate, always call propose_rate_changes; the buttons only appear through it. \
Never write that you changed or proposed a rate in a reply: if you didn't call the tool, nothing \
happened.
- A customer with a bare number ("RK 71?", "ACME 25 aayu?") asks about that challan: use \
find_slips. A rate question names an item or says rate/bhav.
- Follow-ups ("and B 1?", "make it 2.75") refer to the customer and item just discussed.
- If asked for something you have no tool for, say briefly what you can do.
- Reply briefly, in the owner's language but always in English letters:
  - message in English -> reply in English ("The B 1 BOLT rate for ACME is 2.50.")
  - message in Gujarati or Hindi -> reply in Gujlish, Gujarati written in English letters \
("ACME no B 1 bolt no bhav 2.50 che.") Never use \
Gujarati or Devanagari script. Plain text: no Markdown, no asterisks or # headings (Telegram \
shows them as they are).
{slip}
Customers (exact Tally names):
{parties}

Stock items (exact Tally names):
{items}
{glossary}"""

SLIP_TOOL = ("- Correct the slip that is open for review (correct_open_slip), when the message is a "
             "correction to it, e.g. \"line 2 is 300\" or \"date is 17/10\".\n")

SLIP_STATUS = {
    "checking": "read, not saved yet (waiting for ✅ Save)",
    "waiting": "saved, waiting for the next Tally file",
    "in_tally_file": "in a Tally file",
    "taken_back": "was in a Tally file but taken back out (it didn't import); not in Tally",
}
MAX_SLIP_ROWS = 30

TOOLS = [
    {"type": "function", "function": {
        "name": "find_slips",
        "description": "Slips sent to the bot, with where each one is now. Filter by any of "
                       "customer, challan number, item and date range; give at least one.",
        "parameters": {"type": "object", "properties": {
            "customer": {"type": "string", "description": "Exact customer name"},
            "challan": {"type": "string", "description": "Challan number"},
            "item": {"type": "string", "description": "Exact stock item name"},
            "from_date": {"type": "string", "description": "YYYY-MM-DD"},
            "to_date": {"type": "string", "description": "YYYY-MM-DD"},
        }},
    }},
    {"type": "function", "function": {
        "name": "find_rates",
        "description": "Current rates from the rates sheet, for a customer, an item, or both. "
                       "Give at least one.",
        "parameters": {"type": "object", "properties": {
            "customer": {"type": "string", "description": "Exact customer name"},
            "item": {"type": "string", "description": "Exact stock item name"},
        }},
    }},
    {"type": "function", "function": {
        "name": "propose_rate_changes",
        "description": "Show the owner rate changes or new rates to confirm. Nothing is written "
                       "until they tap the button.",
        "parameters": {"type": "object", "properties": {
            "changes": {"type": "array", "items": {"type": "object", "properties": {
                "customer": {"type": "string", "description": "Exact customer name"},
                "item": {"type": "string", "description": "Exact stock item name"},
                "rate": {"type": "number", "description": "New rate in rupees per unit"},
            }, "required": ["customer", "item", "rate"]}},
        }, "required": ["changes"]},
    }},
]
SLIP_TOOLS = [
    {"type": "function", "function": {
        "name": "correct_open_slip",
        "description": "Apply a correction to the slip that is open for review.",
        "parameters": {"type": "object", "properties": {
            "instruction": {"type": "string", "description": "The correction, in English"},
        }, "required": ["instruction"]},
    }},
]


@dataclass(frozen=True)
class RateChange:
    party: str
    item: str
    old: Decimal | None      # None: a new row in the sheet
    new: Decimal


@dataclass
class Outcome:
    """What the bot should do. A proposal or a slip correction ends the turn."""
    reply: str = ""
    rate_changes: list[RateChange] = field(default_factory=list)
    slip_correction: str | None = None

    def summary(self) -> str:
        """What the assistant did, for the conversation history."""
        if self.slip_correction:
            return f"(Corrected the open slip: {self.slip_correction})"
        if self.rate_changes:
            changes = "; ".join(f"{c.party.strip()} {c.item} {c.old if c.old is not None else 'new'} -> {c.new}"
                                for c in self.rate_changes)
            return f"(Proposed rate changes for the owner to confirm: {changes})"[:2000]
        return self.reply


def build_system_prompt(masters: Masters, open_slip: Slip | None) -> str:
    def line(name: str, aliases: tuple[str, ...]) -> str:
        return f"- {name.strip()}" + (f" (also: {', '.join(aliases)})" if aliases else "")

    slip = ""
    if open_slip is not None:
        lines = "\n".join(f"  {n}. {l.quantity:g} {l.item}" for n, l in enumerate(open_slip.lines, 1))
        slip = (f"\nThe slip open for review: customer {open_slip.party.strip()}, date {open_slip.date}, "
                f"challan {open_slip.challan_number}, lines:\n{lines}\n")
    glossary = ("\nWords used in item names:\n" + "\n".join(f"- {g}" for g in masters.glossary) + "\n"
                if masters.glossary else "")
    return SYSTEM_PROMPT.format(
        company=masters.company, slip_tool=SLIP_TOOL if open_slip else "", slip=slip, glossary=glossary,
        parties="\n".join(line(p.ledger, p.aliases) for p in masters.parties),
        items="\n".join(line(i.name, i.aliases) for i in masters.items))


class Assistant:
    def __init__(self, masters: Masters, settings: Settings, rates: Callable[[], dict[tuple[str, str], Rate]],
                 client: openai.AsyncOpenAI, provider: dict,
                 slips: Callable[[], list[SlipRecord]] = list) -> None:
        self.masters, self.rates, self.client, self.provider = masters, rates, client, provider
        self.slips = slips
        self.model = settings.llm_model
        self.names = Names(masters)

    async def handle(self, text: str, open_slip: Slip | None,
                     history: list[tuple[str, str]] = ()) -> Outcome:
        """history: the recent conversation, as (role, text), oldest first."""
        messages: list[dict] = [{"role": "system", "content": build_system_prompt(self.masters, open_slip)},
                                *({"role": role, "content": said} for role, said in history),
                                {"role": "user", "content": text}]
        tools = TOOLS + (SLIP_TOOLS if open_slip else [])
        outcome, rewritten = Outcome(), False
        for _ in range(MAX_STEPS):
            response = await self.client.chat.completions.create(
                model=self.model, max_tokens=4000, messages=messages, tools=tools,
                extra_body={"reasoning": {"effort": "low"}, **self.provider})
            message = response.choices[0].message if response.choices else None
            if message is None:
                raise ExtractionError("the assistant gave no answer")
            if not message.tool_calls:
                reply = (message.content or "").strip()
                if NON_LATIN.search(reply) and not rewritten:   # the owner reads English letters only
                    rewritten = True
                    messages += [{"role": "assistant", "content": reply},
                                 {"role": "user", "content": "Write that again in English letters only "
                                                             "(English or Gujlish), no Gujarati or Devanagari script."}]
                    continue
                outcome.reply = reply
                return outcome
            # The whole message goes back, including any reasoning the model needs to continue.
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                result = await self._run_tool(call.function.name, call.function.arguments, outcome)
                log.info("Assistant tool %s(%s) -> %s", call.function.name, call.function.arguments, result[:200])
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if outcome.rate_changes or outcome.slip_correction:
                return outcome
        raise ExtractionError("the assistant didn't finish")

    async def _run_tool(self, name: str, arguments: str, outcome: Outcome) -> str:
        try:
            args = json.loads(arguments or "{}")
            if name == "find_slips":
                return await self._find_slips(**{k: v for k, v in args.items() if v})
            if name == "find_rates":
                return await self._find_rates(args.get("customer"), args.get("item"))
            if name == "propose_rate_changes":
                outcome.rate_changes = await self._rate_changes(args.get("changes") or [])
                return "Shown to the owner to confirm."
            if name == "correct_open_slip":
                outcome.slip_correction = str(args["instruction"])
                return "The corrected slip is being shown to the owner."
            return f"Unknown tool {name}."
        except (ValueError, KeyError, TypeError) as e:   # told to the model, which can fix its call
            return f"Error: {e}"

    async def _find_rates(self, customer: str | None, item: str | None) -> str:
        if not customer and not item:
            raise ValueError("Give a customer, an item, or both.")
        party = self.names.party(customer) if customer else None
        stock = self.names.item(item) if item else None
        table = await asyncio.to_thread(self.rates)
        rows = [{"customer": p.strip(), "item": i, "rate": f"{r.rate:.2f}", "updated": r.updated}
                for (p, i), r in sorted(table.items())
                if (party is None or p == party) and (stock is None or i == stock)]
        if not rows:
            return "No rates in the sheet for that."
        more = f" (first {MAX_RATE_ROWS} of {len(rows)})" if len(rows) > MAX_RATE_ROWS else ""
        return f"Rates{more}:\n" + json.dumps(rows[:MAX_RATE_ROWS], ensure_ascii=False)

    async def _find_slips(self, customer: str | None = None, challan: str | None = None, item: str | None = None,
                          from_date: str | None = None, to_date: str | None = None) -> str:
        if not any((customer, challan, item, from_date, to_date)):
            raise ValueError("Give at least one of customer, challan, item, from_date, to_date.")
        party = self.names.party(customer) if customer else None
        stock = self.names.item(item) if item else None
        number = _challan(challan) if challan else None
        found = [r for r in await asyncio.to_thread(self.slips)
                 if (party is None or r.party == party)
                 and (number is None or _challan(r.challan or "") == number)
                 and (stock is None or (r.lines is not None and any(i == stock for _, i in r.lines)))
                 and (not from_date or (r.date or "") >= from_date) and (not to_date or (r.date or "") <= to_date)]
        if not found:
            return "No slip like that has been sent to the bot."
        found.sort(key=lambda r: (r.date or "", r.challan or ""))
        rows = [{"customer": r.party.strip(), "date": r.date, "challan": r.challan,
                 "where": SLIP_STATUS[r.status] + (f" made {_file_time(r.file)}" if r.file else ""),
                 **({"lines": [f"{q:g} {i}" for q, i in r.lines]} if r.lines is not None else {}),
                 **({"total": f"{r.total:.2f}"} if r.total is not None else {})} for r in found]
        more = f" (first {MAX_SLIP_ROWS} of {len(rows)})" if len(rows) > MAX_SLIP_ROWS else ""
        return f"Slips{more}:\n" + json.dumps(rows[:MAX_SLIP_ROWS], ensure_ascii=False)

    async def _rate_changes(self, changes: list[dict]) -> list[RateChange]:
        if not changes:
            raise ValueError("No changes given.")
        if len(changes) > MAX_CHANGES:
            raise ValueError(f"At most {MAX_CHANGES} changes at once: ask the owner to split the request.")
        table = await asyncio.to_thread(self.rates)
        out, problems = {}, []
        for change in changes:
            try:
                party, item = self.names.party(change["customer"]), self.names.item(change["item"])
                rate = Decimal(str(change["rate"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            except (ValueError, KeyError, InvalidOperation) as e:
                problems.append(str(e))
                continue
            if not 0 < rate < MAX_RATE:
                problems.append(f"{rate} is not a sensible rate for {item}.")
                continue
            old = table.get((party, item))
            out[(party, item)] = RateChange(party, item, old.rate if old else None, rate)
        if problems:
            raise ValueError(" ".join(problems))
        return list(out.values())
