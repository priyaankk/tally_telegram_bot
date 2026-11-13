"""Everything the bot remembers: slips being reviewed, slips saved for the weekly file, the
export archive, and spellings learned from confirmed slips.

Every record is stored on its own (one per slip, per message, per spelling), so two photos
handled at the same moment, e.g. by two Lambda invocations, never overwrite each other."""

from __future__ import annotations

import io
import re
import time
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from PIL import Image

from .config import Settings
from .kv import KV, DynamoKV, FileKV
from .models import UNKNOWN, Slip, Voucher, learned_key, normalize_text

MAX_PHOTO_BYTES = 350_000   # DynamoDB items are capped at 400 KB


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-") or "x"


def xml_filename(voucher: Voucher) -> str:
    parts = ["Sales", _safe(voucher.party), voucher.date.isoformat()]
    if voucher.challan_number:
        parts.append(f"challan{_safe(voucher.challan_number)}")
    return "_".join(parts) + ".xml"


CONVERSATION_TURNS = 12    # messages (6 questions and answers)
CONVERSATION_HOURS = 2


def same_slip(a: Slip, b: Slip) -> bool:
    """A challan number is used only once a day (books restart, so it recurs on other days):
    the same date and number is the same slip, even if the customer was read differently."""
    if a.challan_number and b.challan_number:
        return a.date == b.date and _challan(a.challan_number) == _challan(b.challan_number)
    return slip_key(a) == slip_key(b)


def _challan(number: str) -> str:
    return number.strip().lstrip("0").casefold()


def _day(timestamp: str) -> str:
    return datetime.fromisoformat(timestamp[:10]).strftime("%d-%m-%Y")


def slip_key(slip: Slip) -> str:
    """Identifies a slip for duplicate checks. Challan books restart, so the challan number
    alone isn't unique: customer + date + challan. Without a challan number, the lines."""
    if slip.challan_number:
        tail = f"challan {slip.challan_number}"
    else:
        tail = ";".join(sorted(f"{l.item}x{l.quantity:g}" for l in slip.lines))
    return f"{slip.party}|{slip.date}|{tail}"


def _fit_jpeg(jpeg: bytes) -> bytes:
    """Re-encode smaller until it fits in one record (only large photos are touched)."""
    if len(jpeg) <= MAX_PHOTO_BYTES:
        return jpeg
    img = Image.open(io.BytesIO(jpeg))
    for edge, quality in ((2048, 80), (1800, 75), (1600, 70), (1400, 65), (1200, 60), (1000, 55),
                          (800, 50), (640, 45), (480, 40)):
        small = img.copy()
        small.thumbnail((edge, edge))
        out = io.BytesIO()
        small.save(out, format="JPEG", quality=quality)
        if out.tell() <= MAX_PHOTO_BYTES:
            return out.getvalue()
    return out.getvalue()


@dataclass(frozen=True)
class Pending:
    draft_id: str
    slip: Slip
    chat_id: int
    saved_at: str


@dataclass(frozen=True)
class FiledSlip:
    """One invoice in a Tally file the bot made."""
    date: str               # ISO
    challan: str | None
    party: str
    total: Decimal
    in_tally: bool          # False: taken back out (it didn't import), so it can be sent again
    lines: tuple[tuple[str, str, str, str], ...] = ()   # (item, quantity, rate, amount) as in the file


@dataclass(frozen=True)
class ExportFile:
    name: str
    slips: list[FiledSlip]


@dataclass(frozen=True)
class SlipRecord:
    """Where a slip the bot has seen is now."""
    status: str             # checking | waiting | in_tally_file | taken_back
    date: str | None
    challan: str | None
    party: str
    lines: list[tuple[float, str]] | None    # None: only the file's total is known
    total: Decimal | None   # once priced into a Tally file
    file: str | None        # the Tally file's name
    chat_id: int | None


@dataclass(frozen=True)
class Draft:
    draft_id: str
    slip: Slip
    chat_id: int
    preview_message_id: int | None


class Store:
    def __init__(self, kv: KV) -> None:
        self.kv = kv

    # -- learned spellings (one record per customer + spelling) ------------------
    def learned_aliases(self) -> dict[str, str]:
        """learned_key(party, spelling) -> stock item name"""
        return {k: v["item"] for k, v in self.kv.items("learned")}

    def learned_spellings(self) -> dict[str, list[str]]:
        """stock item name -> spellings seen on confirmed slips"""
        by_item: dict[str, list[str]] = {}
        for _, row in self.kv.items("learned"):
            by_item.setdefault(row["item"], []).append(row["text"])
        return by_item

    def learn(self, slip: Slip) -> None:
        """Remember how each item was written on a slip a person has confirmed.
        A later confirmation of the same spelling wins, so corrections stick."""
        for line in slip.lines:
            if normalize_text(line.item_text) and line.item != UNKNOWN and slip.party != UNKNOWN:
                self.kv.put("learned", learned_key(slip.party, line.item_text),
                            {"party": slip.party, "item": line.item, "text": line.item_text})

    # -- incoming photos ------------------------------------------------------
    # Every photo is kept from the moment it arrives until it has been read. Whatever is left
    # here is a photo that failed (or the bot stopped mid-read): it can be looked at and re-read.
    def keep_incoming(self, chat_id: int, user_id: int, jpeg: bytes) -> str:
        photo_id = uuid.uuid4().hex[:12]
        self.kv.put_bytes("incoming_photo", photo_id, _fit_jpeg(jpeg))
        self.kv.put("incoming", photo_id, {"chat_id": chat_id, "user_id": user_id,
                                           "received_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        return photo_id

    def incoming_read(self, photo_id: str) -> None:
        self.kv.delete("incoming", photo_id)
        self.kv.delete("incoming_photo", photo_id)

    def unread_photos(self) -> list[tuple[str, dict]]:
        return sorted(self.kv.items("incoming"), key=lambda kv: kv[1]["received_at"])

    def incoming_photo(self, photo_id: str) -> bytes:
        return self.kv.get_bytes("incoming_photo", photo_id)

    # -- drafts (slips being reviewed) ------------------------------------------
    def new_draft(self, jpeg: bytes, chat_id: int, slip: Slip) -> str:
        draft_id = uuid.uuid4().hex[:12]
        self.kv.put_bytes("photo", draft_id, _fit_jpeg(jpeg))
        self.save_draft(Draft(draft_id, slip, chat_id, None))
        return draft_id

    def save_draft(self, draft: Draft) -> None:
        self.kv.put("draft", draft.draft_id, {"slip": draft.slip.model_dump(), "chat_id": draft.chat_id,
                                              "preview_message_id": draft.preview_message_id})

    def draft(self, draft_id: str) -> Draft | None:
        row = self.kv.get("draft", draft_id)
        if row is None:
            return None
        return Draft(draft_id, Slip.model_validate(row["slip"]), row["chat_id"], row["preview_message_id"])

    def drop_draft(self, draft_id: str, keep_photo: bool) -> None:
        self.kv.delete("draft", draft_id)
        if not keep_photo:
            self.kv.delete("photo", draft_id)

    def photo(self, draft_id: str) -> bytes:
        return self.kv.get_bytes("photo", draft_id)

    # -- what a bot message is about (replies are matched through these) -----------
    def link_message(self, chat_id: int, message_id: int, **info) -> None:
        self.kv.put("msg", f"{chat_id}:{message_id}", info)

    def message_link(self, chat_id: int, message_id: int) -> dict | None:
        return self.kv.get("msg", f"{chat_id}:{message_id}")

    def unlink_message(self, chat_id: int, message_id: int) -> None:
        self.kv.delete("msg", f"{chat_id}:{message_id}")

    def set_last_draft(self, chat_id: int, draft_id: str | None) -> None:
        if draft_id is None:
            self.kv.delete("last_draft", str(chat_id))
        else:
            self.kv.put("last_draft", str(chat_id), {"draft_id": draft_id})

    def last_draft(self, chat_id: int) -> str | None:
        row = self.kv.get("last_draft", str(chat_id))
        return row["draft_id"] if row else None

    # -- the conversation with the assistant ------------------------------------------
    # Recent exchanges only, so "and B 1?" follows on from "rates for ACME" without old
    # topics leaking into new questions.
    def conversation(self, chat_id: int) -> list[tuple[str, str]]:
        cutoff = time.time() - CONVERSATION_HOURS * 3600
        row = self.kv.get("convo", str(chat_id)) or {"turns": []}
        return [(t["role"], t["text"]) for t in row["turns"] if t["at"] >= cutoff]

    def remember_exchange(self, chat_id: int, asked: str, answered: str) -> None:
        now = time.time()
        row = self.kv.get("convo", str(chat_id)) or {"turns": []}
        turns = [t for t in row["turns"] if t["at"] >= now - CONVERSATION_HOURS * 3600]
        turns += [{"role": "user", "text": asked, "at": now}, {"role": "assistant", "text": answered, "at": now}]
        self.kv.put("convo", str(chat_id), {"turns": turns[-CONVERSATION_TURNS:]})

    def note_in_conversation(self, chat_id: int, note: str) -> None:
        """Something done outside the chat (e.g. in /see), so the assistant knows about it."""
        row = self.kv.get("convo", str(chat_id)) or {"turns": []}
        turns = row["turns"] + [{"role": "assistant", "text": note, "at": time.time()}]
        self.kv.put("convo", str(chat_id), {"turns": turns[-CONVERSATION_TURNS:]})

    def put_rate_updates(self, draft_id: str, updates: list[tuple[str, str, str]]) -> None:
        self.kv.put("rate_update", draft_id, {"updates": [list(u) for u in updates]})

    def pop_rate_updates(self, draft_id: str) -> list[tuple[str, str, str]] | None:
        row = self.kv.get("rate_update", draft_id)
        if row is None:
            return None
        self.kv.delete("rate_update", draft_id)
        return [tuple(u) for u in row["updates"]]

    # -- pending (confirmed, waiting for the weekly export) ----------------------
    def add_pending(self, draft_id: str, slip: Slip, chat_id: int) -> None:
        self.kv.put("pending", draft_id, {"slip": slip.model_dump(), "chat_id": chat_id,
                                          "saved_at": datetime.now().isoformat(timespec="seconds")})

    def update_pending(self, draft_id: str, slip: Slip) -> None:
        row = self.kv.get("pending", draft_id)
        if row is not None:
            row["slip"] = slip.model_dump()
            self.kv.put("pending", draft_id, row)

    def remove_pending(self, draft_id: str) -> None:
        self.kv.delete("pending", draft_id)
        self.kv.delete("photo", draft_id)

    def take_back(self, export: str, party: str, day: str, challan: str | None, chat_id: int) -> Slip | None:
        """A slip in a Tally file that didn't import: no longer counted as in Tally, and waiting
        for the next file again. None if it isn't (or is no longer) in that file."""
        for key, row in self.kv.items("exported"):
            if (row["export"], row["party"], row["date"], row.get("challan_number")) == (export, party, day, challan):
                slip = Slip.model_validate(row["slip"])
                self.add_pending(row["draft_id"], slip, chat_id)
                self.kv.delete("exported", key)
                return slip
        return None

    def pending(self) -> list[Pending]:
        out = [Pending(k, Slip.model_validate(v["slip"]), v["chat_id"], v["saved_at"])
               for k, v in self.kv.items("pending")]
        return sorted(out, key=lambda p: (p.slip.date or "", p.saved_at))

    # -- exports --------------------------------------------------------------
    def find_duplicate(self, slip: Slip, ignore: str | None = None) -> str | None:
        """Describe an earlier copy of this slip (waiting for export, or already exported), if any.
        ignore: a waiting slip's own id, when it is the one being checked."""
        for p in self.pending():
            if p.draft_id != ignore and same_slip(p.slip, slip):
                return f"already saved on {_day(p.saved_at)} ({p.slip.party.strip()})"
        for _, row in self.kv.items("exported"):
            if "slip" in row and same_slip(Slip.model_validate(row["slip"]), slip):
                return f"already sent to Tally on {_day(row['exported_at'])} ({row['party'].strip()})"
        return None

    def archive_batch(self, name: str, entries: list[tuple[Pending, Voucher]], xml: bytes) -> None:
        """File an export: the XML (compressed) and which slips went in. Removes them from pending;
        their photos are kept under the same draft ids."""
        now = datetime.now().isoformat(timespec="seconds")
        self.kv.put_bytes("export_xml", name, zlib.compress(xml))
        self.kv.put("export", name, {"exported_at": now, "chat_id": entries[0][0].chat_id if entries else None,
                                     "slips": [p.draft_id for p, _ in entries],
                                     "total": f"{sum(v.total for _, v in entries):.2f}"})
        for p, voucher in entries:
            self.kv.put("exported", slip_key(p.slip), {
                "party": voucher.party, "date": voucher.date.isoformat(), "challan_number": voucher.challan_number,
                "total": f"{voucher.total:.2f}", "exported_at": now, "export": name, "draft_id": p.draft_id,
                "slip": p.slip.model_dump()})
            self.kv.delete("pending", p.draft_id)

    def exports(self, chat_id: int | None, limit: int | None) -> list[ExportFile]:
        """The latest Tally files made for this chat (None: everyone's), newest first, with what
        is in each (read from the file itself). Files from before chats were recorded are
        shown to everyone."""
        from .tally_data import _objects, parse_tally_xml

        kept = {(r["export"], r["party"], r["date"], r.get("challan_number")) for _, r in self.kv.items("exported")}
        files = sorted(((n, r) for n, r in self.kv.items("export")
                        if chat_id is None or r.get("chat_id", chat_id) == chat_id),
                       key=lambda nr: nr[1]["exported_at"], reverse=True)[:limit]
        out = []
        for name, _ in files:
            slips = []
            for v in _objects(parse_tally_xml(self.export_xml(name) or b"<ENVELOPE/>"), "VOUCHER"):
                d, party, challan = v.findtext("DATE") or "", v.findtext("PARTYLEDGERNAME") or "", v.findtext("REFERENCE")
                iso = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                amount = next((e.findtext("AMOUNT") for e in v.findall("LEDGERENTRIES.LIST")
                               if e.findtext("ISPARTYLEDGER") == "Yes"), "0")
                lines = tuple((e.findtext("STOCKITEMNAME") or "", (e.findtext("BILLEDQTY") or "").strip(),
                               (e.findtext("RATE") or "").split("/")[0], e.findtext("AMOUNT") or "")
                              for e in v.findall("ALLINVENTORYENTRIES.LIST"))
                slips.append(FiledSlip(iso, challan, party, abs(Decimal(amount)), (name, party, iso, challan) in kept,
                                       lines))
            out.append(ExportFile(name, slips))
        return out

    def slip_records(self) -> list[SlipRecord]:
        """Every slip the bot knows of, for everyone: being checked, waiting for the next file,
        in a Tally file, or taken back out of one."""
        def lines(slip: Slip) -> list[tuple[float, str]]:
            return [(l.quantity, l.item) for l in slip.lines]

        out = [SlipRecord("checking", d.slip.date, d.slip.challan_number, d.slip.party, lines(d.slip), None, None,
                          d.chat_id)
               for d in (self.draft(k) for k, _ in self.kv.items("draft")) if d is not None and d.slip.is_slip]
        out += [SlipRecord("waiting", p.slip.date, p.slip.challan_number, p.slip.party, lines(p.slip), None, None,
                           p.chat_id) for p in self.pending()]
        exported = [r for _, r in self.kv.items("exported")]
        out += [SlipRecord("in_tally_file", r["date"], r.get("challan_number"), r["party"],
                           lines(Slip.model_validate(r["slip"])) if "slip" in r else None,
                           Decimal(r["total"]), r["export"], None) for r in exported]
        out += [SlipRecord("taken_back", f.date, f.challan, f.party, None, f.total, file.name, None)
                for file in self.exports(None, None) for f in file.slips if not f.in_tally]
        return out

    def export_xml(self, name: str) -> bytes | None:
        data = self.kv.get_bytes("export_xml", name)
        return zlib.decompress(data) if data else None

    # -- weekly schedule state ----------------------------------------------------
    def last_batch_run(self) -> datetime | None:
        row = self.kv.get("state", "batch")
        return datetime.fromisoformat(row["last_run"]) if row else None

    def set_last_batch_run(self, when: datetime) -> None:
        self.kv.put("state", "batch", {"last_run": when.isoformat(timespec="seconds")})


def make_store(settings: Settings) -> Store:
    if settings.store_backend == "dynamodb":
        return Store(DynamoKV(settings.dynamodb_table))
    return Store(FileKV(settings.data_dir))
