import asyncio
import json
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from PIL import Image, ImageDraw

from tally_agent.config import Item, Masters, Party, Tax, load_masters
from tally_agent.extract import build_system_prompt, ruled_lines_horizontal
from tally_agent.models import (UNKNOWN, Slip, SlipLine, apply_learned, check_slip, missing_rates,
                                price_slip, slip_json_schema)
from tally_agent.preview import inr, render_slip
from tally_agent.rates import CsvRates, Rate, customer_items, rows_to_table
from tally_agent.kv import FileKV
from tally_agent.store import Draft, Store, slip_key
from tally_agent.tally_data import _objects, parse_tally_xml
from tally_agent.tally_xml import build_import_xml
from tally_agent.weekly import build_batch, is_due, last_scheduled

ROOT = Path(__file__).parent.parent
ACME = "ACME TRADERS "   # trailing space, exactly as in Tally


def masters(**overrides) -> Masters:
    base = dict(
        company="Demo Co 26/27",
        voucher_type="Sales",
        sales_ledger="SALES A/C",
        godown="Main Location",
        parties=(Party(ACME, ("M/s ACME",)), Party("B&C STORES")),
        items=(Item("S3 SCREW", "PCS"), Item("N3 NUT", "PCS"), Item("CLIPS", "PCS")),
    )
    return Masters(**(base | overrides))


RATES = {(ACME, "S3 SCREW"): Rate(Decimal("3.00"), "09-10-2026"),
         (ACME, "N3 NUT"): Rate(Decimal("3.00"), "09-10-2026"),
         ("B&C STORES", "S3 SCREW"): Rate(Decimal("2.80"), "")}


def line(item: str, qty: float, rate: float | None = None, text: str | None = None) -> SlipLine:
    return SlipLine(item_text=text or item.lower(), quantity=qty, item=item,
                    confidence="high", rate_override=rate)


def slip(*lines: SlipLine, party=ACME, d="2026-07-16", challan="14") -> Slip:
    return Slip(is_slip=True, party=party, party_raw="M/s ACME", date=d,
                challan_number=challan, bags=1, lines=list(lines), notes="")


def vouchers_in(xml: bytes):
    return _objects(parse_tally_xml(xml), "VOUCHER")


def test_hints_match_real_masters():
    # The committed hints must only name things that exist in the generated Tally masters.
    generated = ROOT / "config" / "masters.toml"
    if not generated.exists():
        pytest.skip("config/masters.toml not generated (needs a Tally export)")
    m = load_masters(generated, ROOT / "config" / "hints.toml")
    assert m.party(ACME) and "Pan Screw" in m.item("S3 SCREW").aliases


def test_prices_from_sheet_and_user_rate():
    v, problems = price_slip(slip(line("S3 SCREW", 250), line("N3 NUT", 500, rate=2.75)), masters(), RATES)
    assert problems == []
    assert [l.amount for l in v.lines] == [Decimal("750.00"), Decimal("1375.00")]
    assert [l.rate_note for l in v.lines] == ["sheet, 09-10-2026", "rate you gave"]
    assert v.total == Decimal("2125.00")


def test_rates_are_per_customer():
    v, _ = price_slip(slip(line("S3 SCREW", 10), party="B&C STORES"), masters(), RATES)
    assert v.lines[0].rate == Decimal("2.80")


def test_problems_block_export():
    s = slip(line("CLIPS", 2000), line(UNKNOWN, 10), party=UNKNOWN, d=None)
    v, problems = price_slip(s, masters(), RATES)
    assert v is None
    assert len(problems) == 3  # party, date, unmatched item (rates only checked once the customer is known)


def test_missing_rate_asks_for_it():
    _, problems = price_slip(slip(line("CLIPS", 2000)), masters(), RATES)
    assert problems == ["No rate for CLIPS for ACME TRADERS in the rates sheet."]


def test_xml_matches_tally9_shape():
    m = masters()
    v, _ = price_slip(slip(line("S3 SCREW", 250.0)), m, RATES)
    xml = build_import_xml([v], m)
    assert xml.startswith(b"\xff\xfe<\x00E\x00N\x00V\x00")    # UTF-16 LE, BOM, no <?xml?> declaration
    assert b"\r\x00\n\x00" in xml                           # CRLF like Tally's own exports
    vch = vouchers_in(xml)[0]
    assert vch.get("VCHTYPE") == "Sales" and vch.get("OBJVIEW") is None
    assert vch.findtext("DATE") == "20260716"
    assert vch.findtext("REFERENCE") == "14"
    assert vch.findtext("PARTYLEDGERNAME") == ACME            # trailing space kept
    inv = vch.find("ALLINVENTORYENTRIES.LIST")
    assert inv.findtext("RATE") == "3.00/PCS"
    assert inv.findtext("ACTUALQTY") == " 250 PCS"
    assert inv.findtext("ACCOUNTINGALLOCATIONS.LIST/LEDGERNAME") == "SALES A/C"
    assert inv.findtext("BATCHALLOCATIONS.LIST/GODOWNNAME") == "Main Location"
    assert inv.findtext("BATCHALLOCATIONS.LIST/BATCHNAME") == "Primary Batch"


def test_xml_balances_with_taxes():
    m = masters(taxes=(Tax("18% Tax", Decimal("18")),))
    v, _ = price_slip(slip(line("S3 SCREW", 333), line("N3 NUT", 7)), m, RATES)
    vch = vouchers_in(build_import_xml([v], m))[0]
    amounts = [Decimal(e.findtext("AMOUNT")) for e in vch.iter()
               if e.tag in ("LEDGERENTRIES.LIST", "ACCOUNTINGALLOCATIONS.LIST")]
    assert sum(amounts) == 0                                  # debits == credits
    party = vch.find("LEDGERENTRIES.LIST")
    assert party.findtext("ISDEEMEDPOSITIVE") == "Yes" and Decimal(party.findtext("AMOUNT")) == -v.total


def test_xml_escapes_names():
    m = masters()
    v, _ = price_slip(slip(line("S3 SCREW", 1), party="B&C STORES"), m, RATES)
    assert vouchers_in(build_import_xml([v], m))[0].findtext("PARTYLEDGERNAME") == "B&C STORES"


def test_schema_restricts_names_to_masters():
    schema = slip_json_schema(masters())
    assert schema["properties"]["lines"]["items"]["properties"]["item"]["enum"] == \
        ["S3 SCREW", "N3 NUT", "CLIPS", UNKNOWN]
    assert "challan_number" in schema["required"]


def test_sheet_names_match_tally_loosely():
    rows = [["Customer", "Item", "Rate", "Last updated", "Notes"],
            ["acme  traders", "s3 screw", "₹3.25", "01-10-2026", ""],   # case/space differences
            ["Nobody", "S3 SCREW", "1", "", ""],                         # unknown customer: ignored
            [ACME.strip(), "N3 NUT", "", "", ""]]                       # no rate: ignored
    assert rows_to_table(rows, masters()) == {(ACME, "S3 SCREW"): Rate(Decimal("3.25"), "01-10-2026")}


def test_csv_rates_add_and_update(tmp_path):
    src = CsvRates(tmp_path / "rates.csv", masters())
    src.set_rate(ACME, "CLIPS", Decimal("0.90"))
    src.set_rate(ACME, "CLIPS", Decimal("1.00"))
    table = src.table()
    assert table[(ACME, "CLIPS")].rate == Decimal("1.00") and len(table) == 1
    assert customer_items(table) == {ACME: ["CLIPS"]}


@pytest.mark.parametrize("x, expected", [
    ("0", "₹0.00"), ("999.5", "₹999.50"), ("1234", "₹1,234.00"),
    ("123456.789", "₹1,23,456.79"), ("12345678", "₹1,23,45,678.00"),
])
def test_inr(x, expected):
    assert inr(Decimal(x)) == expected


def test_slip_preview_has_no_prices_and_flags_missing_rates():
    s = slip(line("S3 SCREW", 250), line("CLIPS", 2000))
    blockers, warnings = check_slip(s, masters(), today=date(2026, 7, 20))
    text = render_slip(s, masters(), blockers, warnings, missing_rates(s, RATES))
    assert "16-07-2026" in text and "Challan 14" in text and "₹" not in text
    assert "CLIPS</b>: 2000 PCS ⚠️ no rate" in text and "Fix before saving" not in text


def test_date_checks():
    m = masters()
    def blockers(d, today=date(2026, 10, 17)):
        return check_slip(slip(line("S3 SCREW", 1), d=d), m, today=today)[0]
    assert "future" in blockers("2026-10-18")[0]
    assert "more than a month ago" in blockers("2026-07-16")[0]          # a day read as the wrong month
    assert "financial year" in blockers("2020-09-28")[0]                 # year misread
    assert not blockers("2026-09-17") and not blockers("2026-10-17")     # a month back to today: fine
    assert "more than a month ago" in blockers("2026-09-16")[0]
    assert "financial year" in blockers("2026-03-25", today=date(2026, 4, 2))[0]   # last year's books
    assert not blockers("2026-02-28", today=date(2026, 3, 31))                       # 31 Mar -> 28 Feb
    assert check_slip(slip(line("S3 SCREW", 1), d=None), m)[0]                                    # missing: blocks


def test_pending_duplicates_and_archive(tmp_path):
    m = masters()
    store = Store(FileKV(tmp_path))
    s = slip(line("S3 SCREW", 250))
    draft = store.new_draft(b"jpeg", 42, s)
    assert store.find_duplicate(s) is None
    store.add_pending(draft, s, chat_id=42)
    assert "already saved" in store.find_duplicate(s)
    # Challan books restart: the same number on another date is not a duplicate.
    assert store.find_duplicate(slip(line("S3 SCREW", 250), d="2026-10-03")) is None
    # A number is used once a day: same date and number is the same slip, however it was read.
    misread = slip(line("S5 SCREW", 500), party="SOMEONE ELSE", challan="014")
    assert "already saved" in store.find_duplicate(misread)
    assert store.find_duplicate(slip(line("S3 SCREW", 250), challan="15")) is None

    batch = build_batch(store.pending(), m, RATES)
    assert [v.total for _, v in batch.ready] == [Decimal("750.00")] and not batch.held
    xml = build_import_xml([v for _, v in batch.ready], m)
    store.archive_batch("week1", batch.ready, xml)
    assert store.export_xml("week1") == xml and store.photo(draft) == b"jpeg"
    assert store.pending() == []
    assert "already sent to Tally" in store.find_duplicate(s)


def test_batch_is_in_date_then_challan_order():
    from tally_agent.store import Pending
    slips = [("a", "2026-07-16", "10"), ("b", "2026-07-15", "30"), ("c", "2026-07-16", None),
             ("d", "2026-07-16", "9"), ("e", "2026-07-15", "4")]
    pending = [Pending(i, slip(line("S3 SCREW", 10), d=d, challan=c), 1, "2026-07-17T10:00:00")
               for i, d, c in slips]
    assert [p.draft_id for p, _ in build_batch(pending, masters(), RATES).ready] == ["e", "b", "d", "a", "c"]


def test_batch_holds_slips_without_rates():
    from tally_agent.store import Pending
    ok = Pending("a", slip(line("S3 SCREW", 10)), 1, "2026-07-16T10:00:00")
    no_rate = Pending("b", slip(line("CLIPS", 10), challan="15"), 1, "2026-07-16T11:00:00")
    batch = build_batch([ok, no_rate], masters(), RATES)
    assert [p.draft_id for p, _ in batch.ready] == ["a"]
    assert batch.held[0][0].draft_id == "b" and "No rate for CLIPS" in batch.held[0][1][0]


IST = ZoneInfo("Asia/Kolkata")
SAT, NINE_PM = 5, time(21, 0)


@pytest.mark.parametrize("now, last_run, due", [
    (datetime(2026, 10, 17, 20, 59, tzinfo=IST), datetime(2026, 10, 10, 21, 1, tzinfo=IST), False),  # Sat, before 9
    (datetime(2026, 10, 17, 21, 0, tzinfo=IST), datetime(2026, 10, 10, 21, 1, tzinfo=IST), True),    # Sat 9 PM
    (datetime(2026, 10, 17, 21, 5, tzinfo=IST), datetime(2026, 10, 17, 21, 0, tzinfo=IST), False),   # already ran
    (datetime(2026, 10, 19, 9, 0, tzinfo=IST), datetime(2026, 10, 10, 21, 1, tzinfo=IST), True),     # laptop off Sat: Monday catch-up
    (datetime(2026, 10, 19, 9, 0, tzinfo=IST), datetime(2026, 10, 17, 21, 0, tzinfo=IST), False),    # Monday, Sat done
])
def test_weekly_schedule(now, last_run, due):
    assert is_due(now, last_run, SAT, NINE_PM) is due


def test_last_scheduled_is_saturday_night():
    assert last_scheduled(datetime(2026, 10, 15, 12, 0, tzinfo=IST), SAT, NINE_PM) == \
        datetime(2026, 10, 10, 21, 0, tzinfo=IST)


def test_learned_spellings_are_per_customer_and_corrections_stick(tmp_path):
    m = masters()
    store = Store(FileKV(tmp_path))
    store.learn(slip(line("S3 SCREW", 250, text="Pan Screw")))            # ACME: Pan Screw = S3

    read = slip(line(UNKNOWN, 100, text="pan-screw"))
    assert apply_learned(read, store.learned_aliases(), m).lines[0].item == "S3 SCREW"
    other = slip(line(UNKNOWN, 100, text="Pan Screw"), party="B&C STORES")
    assert apply_learned(other, store.learned_aliases(), m).lines[0].item == UNKNOWN

    store.learn(slip(line("N3 NUT", 1, text="Pan Screw")))              # a later correction wins
    assert apply_learned(read, store.learned_aliases(), m).lines[0].item == "N3 NUT"
    assert store.learned_spellings() == {"N3 NUT": ["Pan Screw"]}


def test_prompt_contains_spellings_and_customer_items():
    prompt = build_system_prompt(masters(glossary=("S = pan screw",)), {"N3 NUT": ["Hex Nutt"]},
                                 {ACME: ["N3 NUT", "S3 SCREW"]})
    assert "- N3 NUT (also written: Hex Nutt)" in prompt
    assert f"- {ACME}: N3 NUT, S3 SCREW" in prompt
    assert "- S = pan screw" in prompt


def _ruled_page(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for y in range(40, h, 40):
        d.line([(0, y), (w, y)], fill=(90, 90, 160), width=2)
    for i in range(30):   # some "handwriting" scribbles
        x, y = 30 + (i * 37) % (w - 60), 50 + (i * 53) % (h - 100)
        d.line([(x, y), (x + 15, y - 20), (x + 25, y)], fill=(20, 20, 120), width=3)
    return img


def test_ruled_lines_direction():
    page = _ruled_page(600, 800)
    assert ruled_lines_horizontal(page)
    assert not ruled_lines_horizontal(page.rotate(90, expand=True))


def test_unusual_item_warning_suggests_what_the_customer_buys():
    from tally_agent.models import unusual_items
    s = slip(line("S3 SCREW", 10), line("CLIPS", 2000))
    warnings = unusual_items(s, {ACME: ["S3 SCREW", "CLIPS (BIG)"]})
    assert warnings == [f"Line 2: {ACME.strip()} hasn't bought CLIPS before (they usually get CLIPS (BIG)). Please check."]
    assert unusual_items(s, {}) == []      # new customer: no history, no warning


def test_similar_items_prefer_the_same_words():
    from tally_agent.models import _similar_items
    assert _similar_items("B 1 BOLT", ["B 9 BOLT", "B 1 BOLT (STEEL)", "H4 HINGE"]) == ["B 1 BOLT (STEEL)", "B 9 BOLT"]
    assert _similar_items("6MM HOOK", ["6MM BIG HOOK", "N3 NUT"]) == ["6MM BIG HOOK"]


class MemoryKV:
    """Stands in for DynamoDB in tests: same interface as FileKV/DynamoKV."""
    def __init__(self):
        self.data = {}
    def get(self, ns, key):
        return self.data.get((ns, key))
    def put(self, ns, key, value):
        import json
        self.data[(ns, key)] = json.loads(json.dumps(value))   # like a real store: no shared objects
    def delete(self, ns, key):
        self.data.pop((ns, key), None)
    def items(self, ns):
        return sorted((k, v) for (n, k), v in self.data.items() if n == ns and isinstance(v, dict))
    def get_bytes(self, ns, key):
        v = self.data.get((ns, key))
        return v if isinstance(v, bytes) else None
    def put_bytes(self, ns, key, value):
        self.data[(ns, key)] = value


@pytest.mark.parametrize("make_kv", [lambda p: FileKV(p), lambda p: MemoryKV()], ids=["files", "memory"])
def test_store_drafts_links_and_rate_updates(tmp_path, make_kv):
    store = Store(make_kv(tmp_path))
    s = slip(line("S3 SCREW", 250))
    d = store.new_draft(b"photo", 7, s)
    store.save_draft(Draft(d, s, 7, 99))
    store.link_message(7, 99, kind="draft", draft_id=d)
    store.link_message(7, 100, kind="rate_q", party=ACME, item="CLIPS")
    store.set_last_draft(7, d)
    assert store.draft(d).preview_message_id == 99 and store.photo(d) == b"photo"
    assert store.message_link(7, 99) == {"kind": "draft", "draft_id": d}
    assert store.message_link(7, 100)["party"] == ACME          # trailing space survives
    assert store.last_draft(7) == d
    store.put_rate_updates(d, [(ACME, "CLIPS", "1.00")])
    assert store.pop_rate_updates(d) == [(ACME, "CLIPS", "1.00")] and store.pop_rate_updates(d) is None
    store.drop_draft(d, keep_photo=False)
    assert store.draft(d) is None and store.photo(d) is None


def test_incoming_photos_are_kept_until_read(tmp_path):
    store = Store(FileKV(tmp_path))
    first = store.keep_incoming(7, 42, b"photo 1")
    second = store.keep_incoming(7, 42, b"photo 2")
    store.incoming_read(second)
    assert [pid for pid, _ in store.unread_photos()] == [first]   # the one that failed stays
    assert store.incoming_photo(first) == b"photo 1" and store.incoming_photo(second) is None
    assert store.unread_photos()[0][1]["user_id"] == 42


def test_retries_think_less_hard():
    from tally_agent.extract import EFFORT_LADDER, SlipReader

    class Calls:
        efforts = []

        async def create(self, **kw):
            self.efforts.append(kw["extra_body"]["reasoning"]["effort"])
            msg = type("M", (), {"refusal": None, "content": None})
            return type("R", (), {"choices": [type("C", (), {"finish_reason": "length", "message": msg})]})

    reader = SlipReader.__new__(SlipReader)
    reader.client = type("Client", (), {"chat": type("Chat", (), {"completions": Calls()})})
    reader.model, reader.provider = "m", {}
    with pytest.raises(Exception, match="cut off"):
        asyncio.run(reader._json_call([], {}, effort="high", max_tokens=10, attempts=4))
    assert Calls.efforts == ["high", "medium", "low", "low"] and EFFORT_LADDER[0] == "high"


def test_file_keys_with_odd_characters_round_trip(tmp_path):
    kv = FileKV(tmp_path)
    key = "ACME TRADERS |2026-07-16|challan 14/a"
    kv.put("exported", key, {"x": 1})
    assert kv.items("exported") == [(key, {"x": 1})]


def test_big_photos_are_shrunk_to_fit_one_record():
    import io
    from PIL import Image
    from tally_agent.store import MAX_PHOTO_BYTES, _fit_jpeg
    noisy = Image.effect_noise((2048, 2048), 100).convert("RGB")
    raw = io.BytesIO()
    noisy.save(raw, format="JPEG", quality=95)
    assert len(raw.getvalue()) > MAX_PHOTO_BYTES
    assert len(_fit_jpeg(raw.getvalue())) <= MAX_PHOTO_BYTES


def test_a_late_button_acknowledgement_is_not_fatal():
    from telegram.error import BadRequest

    from tally_agent.bot import _answer

    sent = []

    class Message:
        async def reply_text(self, text):
            sent.append(text)

    class TooOldQuery:
        message = Message()

        async def answer(self, text=None, show_alert=False):
            raise BadRequest("Query is too old and response timeout expired or query id is invalid")

    asyncio.run(_answer(TooOldQuery(), "Saved"))
    assert sent == []
    asyncio.run(_answer(TooOldQuery(), "Please fix the problems first.", show_alert=True))
    assert sent == ["Please fix the problems first."]   # a pop-up that never showed goes in the chat


class FakeChat:
    """Plays back model turns: each is a reply text or a list of (tool name, arguments)."""

    def __init__(self, turns):
        from types import SimpleNamespace as NS
        self.NS, self.turns, self.seen = NS, list(turns), []
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kwargs):
        NS = self.NS
        self.seen.append(kwargs["messages"][:])
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            message = NS(content=turn, tool_calls=None)
        else:
            calls = [NS(id=f"c{n}", function=NS(name=name, arguments=json.dumps(args)))
                     for n, (name, args) in enumerate(turn)]
            message = NS(content=None, tool_calls=calls)
        message.model_dump = lambda **_: {"role": "assistant", "tool_calls": []}
        return NS(choices=[NS(message=message)])


def assistant_with(turns):
    from tally_agent.assistant import Assistant
    from tally_agent.config import Settings
    client = FakeChat(turns)
    settings = Settings.__new__(Settings)
    object.__setattr__(settings, "llm_model", "test")
    return Assistant(masters(), settings, lambda: RATES, client, {}), client


def test_assistant_proposes_rate_changes_with_the_old_rate():
    assistant, client = assistant_with([
        [("find_rates", {"customer": "acme traders"})],
        [("propose_rate_changes", {"changes": [{"customer": "m/s  acme", "item": "S3 Screw", "rate": 3.5},
                                               {"customer": "ACME TRADERS", "item": "CLIPS", "rate": 1.234}]})],
    ])
    outcome = asyncio.run(assistant.handle("ACME S3 screw 3.50, clips 1.23", None))
    assert [(c.party, c.item, c.old, c.new) for c in outcome.rate_changes] == [
        (ACME, "S3 SCREW", Decimal("3.00"), Decimal("3.50")), (ACME, "CLIPS", None, Decimal("1.23"))]
    assert '"rate": "3.00"' in client.seen[1][-1]["content"]          # the lookup reached the model
    assert "correct_open_slip" not in json.dumps(client.seen[0])      # no slip open: no slip tool


def test_assistant_names_must_be_tally_names():
    assistant, client = assistant_with([
        [("propose_rate_changes", {"changes": [{"customer": "ACME TRADERS", "item": "S9 SCREW", "rate": 3}]})],
        "Which screw did you mean?",
    ])
    outcome = asyncio.run(assistant.handle("ACME S9 screw 3", None))
    assert outcome.rate_changes == [] and outcome.reply == "Which screw did you mean?"
    assert "no stock item called “S9 SCREW”" in client.seen[1][-1]["content"]


def test_assistant_hands_slip_corrections_back():
    assistant, _ = assistant_with([[("correct_open_slip", {"instruction": "line 1 is 300"})]])
    outcome = asyncio.run(assistant.handle("line 1 is 300", slip(line("S3 SCREW", 250))))
    assert outcome.slip_correction == "line 1 is 300" and not outcome.rate_changes


def test_assistant_sees_the_recent_conversation(tmp_path):
    store = Store(FileKV(tmp_path))
    store.remember_exchange(7, "rates for ACME", "S3 SCREW 3.00, N3 NUT 3.00")
    assert store.conversation(7) == [("user", "rates for ACME"), ("assistant", "S3 SCREW 3.00, N3 NUT 3.00")]
    assert store.conversation(8) == []
    assistant, client = assistant_with(["N3 NUT is 3.00 for ACME TRADERS."])
    asyncio.run(assistant.handle("and N3?", None, store.conversation(7)))
    assert [m["role"] for m in client.seen[0]] == ["system", "user", "assistant", "user"]


def test_assistant_replies_in_english_letters():
    assistant, client = assistant_with(["ACME નો ભાવ 2.50 છે", "ACME no bhav 2.50 che"])
    assert asyncio.run(assistant.handle("acme no bhav?", None)).reply == "ACME no bhav 2.50 che"
    assert "English letters only" in client.seen[1][-1]["content"]


def test_see_lists_each_file_and_whats_waiting(tmp_path):
    from tally_agent.preview import render_see
    from tally_agent.store import Pending
    m = masters()
    store = Store(FileKV(tmp_path))
    first, second = slip(line("S3 SCREW", 250), challan="14"), slip(line("N3 NUT", 10), challan="15")
    for n, s in enumerate((first, second)):
        store.add_pending(f"d{n}", s, chat_id=7)
    batch = build_batch(store.pending(), m, RATES)
    store.archive_batch("Tally_Sales_2026-10-17_1540", batch.ready, build_import_xml([v for _, v in batch.ready], m))
    store.kv.delete("exported", slip_key(second))                      # taken back: it didn't import
    store.add_pending("d2", slip(line("CLIPS", 5), challan="16"), chat_id=7)
    store.add_pending("other", slip(line("S3 SCREW", 1), challan="99"), chat_id=8)

    files = store.exports(7, limit=5)
    assert [(f.challan, f.total, f.in_tally) for f in files[0].slips] == [
        ("14", Decimal("750.00"), True), ("15", Decimal("30.00"), False)]
    assert store.exports(8, limit=5) == []                             # another person's files
    waiting = build_batch([p for p in store.pending() if p.chat_id == 7], m, RATES)
    text = render_see(files, waiting.ready, waiting.held)
    assert "17-10-2026 15:40</b> · 2 invoices · ₹780.00" in text
    assert "Ch 15 · ACME TRADERS · ₹30.00  ↩️" in text and "Ch 16" in text and "⚠️" in text and "Ch 99" not in text


def signed_init_data(user_id: int, token: str, auth_date: int) -> str:
    import hashlib
    import hmac
    from urllib.parse import urlencode
    fields = {"auth_date": str(auth_date), "user": json.dumps({"id": user_id, "first_name": "Test"})}
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    return urlencode(fields | {"hash": hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()})


def test_see_page_only_trusts_telegram_signed_requests():
    from tally_agent.webapp import verify_init_data
    good = signed_init_data(42, "token", auth_date=1_000_000)
    assert verify_init_data(good, "token", now=1_000_100) == 42
    for bad, now in ((good, 1_000_000 + 2 * 86400),                       # too old
                     (signed_init_data(42, "other bot", 1_000_000), 1_000_100),   # someone else's signature
                     (good.replace("42", "43"), 1_000_100), ("", 1_000_100)):     # tampered, missing
        with pytest.raises(PermissionError):
            verify_init_data(bad, "token", now=now)


def see_app(tmp_path):
    from types import SimpleNamespace as NS
    from zoneinfo import ZoneInfo
    from tally_agent.webapp import SeeApp
    store = Store(FileKV(tmp_path))
    return SeeApp(NS(timezone=ZoneInfo("Asia/Kolkata")), masters(), store, NS(table=lambda: RATES)), store


def test_see_page_edits_deletes_and_takes_back(tmp_path):
    from datetime import date as day
    from tally_agent.webapp import Refused
    app, store = see_app(tmp_path)
    today = day.today().isoformat()
    store.add_pending("a", slip(line("S3 SCREW", 250), d=today, challan="14"), chat_id=7)
    store.add_pending("b", slip(line("N3 NUT", 10), d=today, challan="15"), chat_id=7)
    store.add_pending("c", slip(line("S3 SCREW", 1), d=today, challan="16"), chat_id=8)

    state = app.state(7)
    assert [w["challan"] for w in state["waiting"]] == ["14", "15"]            # only this chat's
    assert state["waiting"][0]["total"] == "750.00" and "CLIPS" in state["items"]

    note = app.edit(7, "a", {"party": "acme traders", "date": today, "challan": "14",
                             "lines": [{"item": "S3 SCREW", "qty": "300"}, {"item": "n3 nut", "qty": 5}]})
    assert "items: 300 S3 SCREW, 5 N3 NUT" in note
    assert [(l.item, l.quantity) for l in store.pending()[0].slip.lines] == [("S3 SCREW", 300), ("N3 NUT", 5)]
    with pytest.raises(Refused, match="no stock item"):
        app.edit(7, "a", {"party": ACME, "date": today, "challan": "14", "lines": [{"item": "S9", "qty": 1}]})
    with pytest.raises(Refused, match="already saved"):                       # challan 15 is taken
        app.edit(7, "a", {"party": ACME, "date": today, "challan": "15", "lines": [{"item": "CLIPS", "qty": 1}]})
    with pytest.raises(Refused, match="can.t be saved"):
        app.edit(7, "a", {"party": ACME, "date": "2026-01-01", "challan": "14", "lines": [{"item": "CLIPS", "qty": 1}]})
    with pytest.raises(Refused, match="no longer waiting"):                   # someone else's slip
        app.delete(7, "c")

    assert "Deleted" in app.delete(7, "b") and [p.draft_id for p in store.pending()] == ["a", "c"]

    mine = [p for p in store.pending() if p.chat_id == 7]
    batch = build_batch(mine, masters(), RATES)
    store.archive_batch("Tally_Sales_2026-10-17_1540", batch.ready, build_import_xml([v for _, v in batch.ready], masters()))
    filed = app.state(7)["files"][0]["slips"][0]
    assert filed["in_tally"] and filed["lines"][0]["item"] == "S3 SCREW"
    assert "waiting for the next Tally file again" in app.take_back(
        7, "Tally_Sales_2026-10-17_1540", filed["ledger"], filed["date"], filed["challan"])
    state = app.state(7)
    assert not state["files"][0]["slips"][0]["in_tally"] and [w["id"] for w in state["waiting"]] == ["a"]
    with pytest.raises(Refused, match="already taken back"):
        app.take_back(7, "Tally_Sales_2026-10-17_1540", filed["ledger"], filed["date"], filed["challan"])
