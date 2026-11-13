"""/see as a Telegram Mini App: a page inside Telegram listing the Tally files (with the challans
in each) and the slips waiting for the next one. Waiting slips can be edited or deleted; a slip
in a file that didn't import can be taken back. Every change is checked like a slip being saved,
posted in the chat, and noted for the assistant.

Served by the same Lambda function URL: GET /app is the page, POST /api its calls. Each call
carries Telegram's signed initData, which proves who is asking. Kept free of the Telegram and
LLM libraries so a call doesn't wait for them to load.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import date, datetime
from html import escape
from pathlib import Path
from urllib.parse import parse_qsl

from .config import Masters, Settings
from .models import UNKNOWN, Names, Slip, SlipLine, check_slip
from .notify import tell_chat
from .preview import _file_time, slip_heading
from .rates import RateSource
from .store import Store
from .weekly import build_batch

log = logging.getLogger(__name__)

PAGE = Path(__file__).parent / "webapp" / "see.html"
MAX_AGE = 24 * 3600          # how long an opened page may keep making changes
SEE_FILES = 10


class Refused(Exception):
    """A request that can't be done, with the reason to show."""


def verify_init_data(init_data: str, bot_token: str, now: float | None = None) -> int:
    """The Telegram user id from a Mini App's initData, after checking Telegram's signature."""
    try:
        fields = dict(parse_qsl(init_data, strict_parsing=True))
        given = fields.pop("hash")
        user = json.loads(fields["user"])["id"]
        signed_at = int(fields["auth_date"])
    except (ValueError, KeyError, TypeError) as e:
        raise PermissionError("not opened from Telegram") from e
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(hmac.new(secret, check.encode(), hashlib.sha256).hexdigest(), given):
        raise PermissionError("bad signature")
    if (now or time.time()) - signed_at > MAX_AGE:
        raise PermissionError("expired: open /see again")
    return int(user)


class SeeApp:
    def __init__(self, settings: Settings, masters: Masters, store: Store, rates: RateSource) -> None:
        self.settings, self.masters, self.store, self.rates = settings, masters, store, rates
        self.names = Names(masters)

    def _today(self) -> date:
        return datetime.now(self.settings.timezone).date()

    # -- what the page shows ---------------------------------------------------------
    def state(self, chat_id: int) -> dict:
        units = {i.name: i.unit for i in self.masters.items}
        pending = [p for p in self.store.pending() if p.chat_id == chat_id]
        batch = build_batch(pending, self.masters, self.rates.table())
        totals = {p.draft_id: f"{v.total:.2f}" for p, v in batch.ready}
        problems = {p.draft_id: why[0] if why else "not ready" for p, why in batch.held}

        def lines(slip: Slip) -> list[dict]:
            return [{"item": l.item, "qty": l.quantity, "unit": units.get(l.item, "")} for l in slip.lines]

        def slip_fields(slip: Slip) -> dict:
            return {"date": slip.date, "challan": slip.challan_number,
                    "party": slip.party.strip() if slip.party != UNKNOWN else f"? {slip.party_raw}"}

        waiting = [{"id": p.draft_id, **slip_fields(p.slip), "lines": lines(p.slip),
                    "total": totals.get(p.draft_id), "problem": problems.get(p.draft_id)}
                   for p in sorted(pending, key=lambda p: (p.slip.date or "", p.slip.challan_number or ""))]
        drafts = (self.store.draft(k) for k, _ in self.store.kv.items("draft"))
        checking = [{**slip_fields(d.slip), "lines": lines(d.slip)}
                    for d in drafts if d is not None and d.chat_id == chat_id and d.slip.is_slip]
        files = [{"name": f.name, "when": _file_time(f.name), "total": f"{sum(s.total for s in f.slips):.2f}",
                  "slips": [{"date": s.date, "challan": s.challan, "party": s.party.strip(), "ledger": s.party,
                             "total": f"{s.total:.2f}", "in_tally": s.in_tally,
                             "lines": [{"item": i, "qty": q, "rate": r, "amount": a} for i, q, r, a in s.lines]}
                            for s in f.slips]}
                 for f in self.store.exports(chat_id, limit=SEE_FILES)]
        return {"waiting": waiting, "checking": checking, "files": files,
                "parties": sorted(p.ledger.strip() for p in self.masters.parties),
                "items": sorted(i.name for i in self.masters.items)}

    # -- changes -----------------------------------------------------------------------
    def _waiting(self, chat_id: int, slip_id: str):
        for p in self.store.pending():
            if p.draft_id == slip_id and p.chat_id == chat_id:
                return p
        raise Refused("This slip is no longer waiting (it may already be in a Tally file).")

    def edit(self, chat_id: int, slip_id: str, change: dict) -> str:
        """Replace a waiting slip's customer, date, challan and lines. Returns the chat note."""
        old = self._waiting(chat_id, slip_id).slip
        try:
            party = self.names.party(str(change["party"]))
            new_lines = []
            for n, line in enumerate(change["lines"]):
                item, qty = self.names.item(str(line["item"])), float(line["qty"])
                before = old.lines[n] if n < len(old.lines) else None
                same = before is not None and before.item == item
                new_lines.append(SlipLine(
                    item_text=before.item_text if same else "changed in /see", quantity=qty, item=item,
                    confidence="high", rate_override=before.rate_override if same else None))
        except (ValueError, KeyError, TypeError) as e:
            raise Refused(str(e)) from e
        slip = old.model_copy(update={"party": party, "date": change.get("date") or None,
                                      "challan_number": str(change.get("challan") or "").strip() or None,
                                      "lines": new_lines})
        blockers, _ = check_slip(slip, self.masters, today=self._today())
        if slip.challan_number and (dup := self.store.find_duplicate(slip, ignore=slip_id)):
            blockers.insert(0, f"Challan {slip.challan_number} of this date is {dup}.")
        if blockers:
            raise Refused(" ".join(blockers))
        if slip == old:
            raise Refused("Nothing changed.")
        self.store.update_pending(slip_id, slip)
        return f"✏️ Changed in /see: {escape(slip_heading(slip))}\n" + "\n".join(_differences(old, slip))

    def delete(self, chat_id: int, slip_id: str) -> str:
        slip = self._waiting(chat_id, slip_id).slip
        self.store.remove_pending(slip_id)
        return f"🗑 Deleted in /see: {escape(slip_heading(slip))}. It won't go in a Tally file."

    def take_back(self, chat_id: int, file: str, ledger: str, day: str, challan: str | None) -> str:
        if file not in {f.name for f in self.store.exports(chat_id, limit=None)}:
            raise Refused("That Tally file isn't one of yours.")
        slip = self.store.take_back(file, ledger, day, challan, chat_id)
        if slip is None:
            raise Refused("That slip was already taken back.")
        return (f"↩️ Taken back in /see: {escape(slip_heading(slip))} didn't import from the "
                f"{_file_time(file)} file. It's waiting for the next Tally file again.")


def _differences(old: Slip, new: Slip) -> list[str]:
    out = []
    if old.party != new.party:
        out.append(f"• customer: {escape(old.party.strip())} → {escape(new.party.strip())}")
    if old.date != new.date:
        out.append(f"• date: {old.date} → {new.date}")
    if old.challan_number != new.challan_number:
        out.append(f"• challan: {escape(old.challan_number or '–')} → {escape(new.challan_number or '–')}")
    before = [(l.item, l.quantity) for l in old.lines]
    after = [(l.item, l.quantity) for l in new.lines]
    if before != after:
        out.append("• items: " + ", ".join(f"{q:g} {escape(i)}" for i, q in after))
    return out


# -- HTTP (called from lambda_app) ----------------------------------------------------------

_app: SeeApp | None = None


def _see_app() -> SeeApp:
    global _app
    if _app is None:
        from .config import load_masters, load_settings
        from .rates import make_rate_source
        from .store import make_store

        settings = load_settings()
        masters = load_masters(settings.masters_path, settings.hints_path)
        _app = SeeApp(settings, masters, make_store(settings), make_rate_source(settings, masters))
    return _app


def _json(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, ensure_ascii=False)}


def page() -> dict:
    return {"statusCode": 200, "headers": {"Content-Type": "text/html; charset=utf-8",
                                           "Cache-Control": "no-cache"},
            "body": PAGE.read_text(encoding="utf-8")}


def api(body: str) -> dict:
    app = _see_app()
    try:
        request = json.loads(body or "{}")
        user = verify_init_data(request.get("initData", ""), app.settings.telegram_token)
    except (PermissionError, ValueError) as e:
        log.warning("/see call refused: %s", e)
        return _json(401, {"error": "Please open this from the /see button in the bot."})
    if user not in app.settings.allowed_user_ids:
        return _json(403, {"error": "Sorry, this bot is private."})
    chat_id = user    # a private chat's id is the user's id
    action = request.get("action", "state")
    log.info("/see %s from user %s", action, user)
    try:
        if action == "edit":
            note = app.edit(chat_id, request["id"], request["slip"])
        elif action == "delete":
            note = app.delete(chat_id, request["id"])
        elif action == "take_back":
            note = app.take_back(chat_id, request["file"], request["ledger"], request["date"], request.get("challan"))
        elif action == "state":
            note = None
        else:
            return _json(400, {"error": f"Unknown action {action}"})
    except Refused as e:
        return _json(409, {"error": str(e), "state": app.state(chat_id)})
    except KeyError as e:
        return _json(400, {"error": f"Missing {e}"})
    if note:
        tell_chat(app.settings.telegram_token, chat_id, note)
        app.store.note_in_conversation(chat_id, f"(The owner changed this in the /see app: {note})")
    return _json(200, {"state": app.state(chat_id), "done": note})
