"""Telegram bot.

During the week: photo of a slip in → the bot shows what it read → the user fixes anything
by replying → ✅ saves it for the weekly file. Missing rates are asked for straight away.
Saturday night: every saved slip is priced from the rates sheet as it is then, and one Tally
XML file is sent.

Runs two ways with the same handlers (build_application):
- on a computer, polling Telegram:  uv run python -m tally_agent.bot
- on AWS Lambda, via a webhook:     tally_agent.lambda_app.handler
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from functools import wraps
from html import escape

import openai
from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
    WebAppInfo,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, NetworkError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .assistant import Assistant, RateChange
from .config import Masters, Settings, load_masters, load_settings
from .extract import ExtractionError, SlipReader, encode_jpeg, load_image
from .models import Slip, apply_learned, check_slip, missing_rates, unusual_items
from .preview import (
    _date,
    _file_time,
    inr,
    render_batch_summary,
    render_see,
    render_slip,
    slip_heading,
)
from .rates import RateSource, customer_items, make_rate_source
from .store import Draft, Store, make_store
from .tally_xml import build_import_xml
from .weekly import batch_name, build_batch, is_due

log = logging.getLogger("tally_agent")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TELEGRAM_LIMIT = 4000
HELP = (
    "📷 Send me a photo of a delivery slip.\n"
    "I'll show you what I read. Reply to fix anything (type it or send a voice note), then tap ✅ Save.\n\n"
    "💰 You can also ask me to look up or change a rate, e.g. “Sharma S5 screw rate 3.50”.\n\n"
    "/pending: slips saved so far\n"
    "/see: the challans in each Tally file, and what's waiting for the next one\n"
    "/export: make the Tally file now"
)


UNREADABLE_PHOTO = "😕 I couldn't read this one. Please send the photo again, ideally flat, in good light."
CAN_HANDLE = "I can read photos of slips, typed messages and voice notes. Please send one of those."
MAX_VOICE_SECONDS = 120
SEE_FILES = 5   # Tally files /see lists


def allowed_only(handler):
    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        settings: Settings = context.bot_data["settings"]
        user = update.effective_user
        if user is None or user.id not in settings.allowed_user_ids:
            log.warning("Ignoring update from unauthorised user %s", user and user.id)
            if update.effective_message:
                await update.effective_message.reply_text("Sorry, this bot is private.")
            return
        m = update.message
        kind = ("button" if m is None else "photo" if m.photo or m.document else "voice" if m.voice
                else "text" if m.text else "other message")
        log.info("%s from user %s: %s", kind, user.id, handler.__name__)
        return await handler(update, context)
    return wrapper


# -- small helpers ---------------------------------------------------------------

def _store(context) -> Store:
    return context.bot_data["store"]


def _today(context) -> date:
    """Today where the slips are written (the server's clock is UTC)."""
    return datetime.now(context.bot_data["settings"].timezone).date()


async def _rate_table(bot_data: dict) -> dict:
    source: RateSource = bot_data["rates"]
    return await asyncio.to_thread(source.table)   # may be a network call (Google Sheets)


async def _set_rate(bot_data: dict, party: str, item: str, rate: Decimal) -> None:
    source: RateSource = bot_data["rates"]
    await asyncio.to_thread(source.set_rate, party, item, rate)


async def _clear_buttons(bot: Bot, chat_id: int, message_id: int) -> None:
    try:
        await bot.edit_message_reply_markup(chat_id, message_id, reply_markup=None)
    except BadRequest:  # already cleared, or the message is too old to edit
        pass


async def _answer(query, text: str | None = None, *, show_alert: bool = False) -> None:
    """Acknowledge a button tap. Telegram only accepts this for a short while; after a slow
    (cold) start it refuses, and that mustn't stop the tap from doing its work."""
    try:
        await query.answer(text, show_alert=show_alert)
    except BadRequest as e:
        log.warning("Couldn't acknowledge the button: %s", e)
        if show_alert:   # the pop-up never showed, so say it in the chat instead
            await query.message.reply_text(text)


async def _send_long(bot: Bot, chat_id: int, text: str, **kwargs) -> None:
    """Send HTML text, split on line breaks to stay under Telegram's message size limit."""
    chunk = ""
    for line in text.split("\n"):
        if len(chunk) + len(line) + 1 > TELEGRAM_LIMIT:
            await bot.send_message(chat_id, chunk, parse_mode=ParseMode.HTML)
            chunk = ""
        chunk += line + "\n"
    await bot.send_message(chat_id, chunk, parse_mode=ParseMode.HTML, **kwargs)


def _parse_rate(text: str) -> Decimal | None:
    m = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
    try:
        return Decimal(m.group()) if m else None
    except InvalidOperation:
        return None


async def _model_call(message: Message, coro, unreadable: str):
    """Run an LLM call, turning failures into a friendly reply. Returns None on failure."""
    try:
        return await coro
    except ExtractionError as e:
        log.warning("Reading failed: %s", e)
        await message.reply_text(unreadable)
    except openai.RateLimitError:
        await message.reply_text("⏳ Too many requests right now. Please send it again in a minute.")
    except (openai.APIConnectionError, openai.InternalServerError):
        await message.reply_text("📶 Couldn't reach the reading service. Please try again.")
    except openai.APIStatusError as e:
        log.error("LLM API error %s: %s", e.status_code, e.message)
        await message.reply_text("⚠️ Something went wrong reading the slip. Please try again.")
    return None


# -- reviewing a slip --------------------------------------------------------------

def _already_have(store: Store, slip: Slip) -> str | None:
    """Why a slip can't be saved: its challan number and date are already in (a challan number
    is used once a day). Slips without a number are only questioned at Save."""
    if not slip.challan_number or not (dup := store.find_duplicate(slip)):
        return None
    return (f"Challan {slip.challan_number} of {_date(slip.date)} is {dup}. "
            "No need to send it again. If I read the number or date wrong, reply with the right one.")


async def _show_slip(message: Message, context, draft_id: str, slip: Slip) -> None:
    masters: Masters = context.bot_data["masters"]
    store = _store(context)
    blockers, warnings = check_slip(slip, masters, today=_today(context))
    if slip.is_slip and (have := _already_have(store, slip)):
        blockers.insert(0, have)
    table = await _rate_table(context.bot_data)
    warnings += unusual_items(slip, customer_items(table))
    missing = missing_rates(slip, table)
    buttons = [InlineKeyboardButton("❌ Cancel", callback_data=f"no:{draft_id}")]
    if not blockers:
        buttons.insert(0, InlineKeyboardButton("✅ Save", callback_data=f"ok:{draft_id}"))
    sent = await message.reply_text(render_slip(slip, masters, blockers, warnings, missing),
                                    parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup([buttons]))

    old = store.draft(draft_id)
    if old and old.preview_message_id:
        store.unlink_message(message.chat_id, old.preview_message_id)
        await _clear_buttons(context.bot, message.chat_id, old.preview_message_id)
    store.save_draft(Draft(draft_id, slip, message.chat_id, sent.message_id))
    store.link_message(message.chat_id, sent.message_id, kind="draft", draft_id=draft_id)
    store.set_last_draft(message.chat_id, draft_id)


def _forget(store: Store, draft: Draft, keep_photo: bool) -> None:
    store.drop_draft(draft.draft_id, keep_photo=keep_photo)
    if draft.preview_message_id:
        store.unlink_message(draft.chat_id, draft.preview_message_id)
    if store.last_draft(draft.chat_id) == draft.draft_id:
        store.set_last_draft(draft.chat_id, None)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    user = update.effective_user
    if user.id in settings.allowed_user_ids:
        await update.message.reply_text(HELP)
    else:
        await update.message.reply_text(
            f"This bot is private. Your Telegram user id is {user.id}. "
            "Ask the owner to add it to ALLOWED_TELEGRAM_USER_IDS."
        )


@allowed_only
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP)


@allowed_only
async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    reader: SlipReader = context.bot_data["reader"]

    tg_file = await (message.photo[-1] if message.photo else message.document).get_file()
    await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
    try:
        image = load_image(bytes(await tg_file.download_as_bytearray()))
    except Exception:
        log.exception("Could not open image")
        await message.reply_text("😕 I couldn't open that image. Please send it as a normal photo.")
        return

    store = _store(context)
    try:   # a copy in case the reading fails; not having one mustn't stop the reading
        photo_id = store.keep_incoming(message.chat_id, update.effective_user.id, encode_jpeg(image))
        log.info("Kept incoming photo %s", photo_id)
    except Exception:
        log.exception("Couldn't keep a copy of the incoming photo")
        photo_id = None
    result = await _model_call(message, reader.read(image), UNREADABLE_PHOTO)
    if result is None:
        return
    if photo_id:
        try:
            store.incoming_read(photo_id)
        except Exception:
            log.exception("Couldn't drop the copy of incoming photo %s", photo_id)
    jpeg, slip = result  # jpeg is the photo turned upright, kept for corrections
    slip = apply_learned(slip, store.learned_aliases(), context.bot_data["masters"])
    await _show_slip(message, context, store.new_draft(jpeg, message.chat_id, slip), slip)


def _reply_target(message: Message, store: Store) -> tuple[dict | None, str | None]:
    """(what the replied-to bot message is about, the slip a correction applies to): the slip
    whose message was replied to, else the latest slip if the message isn't a reply."""
    reply_to = message.reply_to_message
    link = store.message_link(message.chat_id, reply_to.message_id) if reply_to else None
    if link and link["kind"] == "draft":
        return link, link["draft_id"]
    return link, None if reply_to else store.last_draft(message.chat_id)


async def _handle_reply(message: Message, context, text: str) -> None:
    """Typed text, or what a voice note said: a rate answer or a correction to a slip."""
    store = _store(context)
    link, draft_id = _reply_target(message, store)

    # An answer to "what's the rate of X for Y?"
    if link and link["kind"] == "rate_q":
        rate = _parse_rate(text)
        if rate is None or rate <= 0:
            await message.reply_text("Please reply with just the rate, e.g. 2.50")
            return
        await _set_rate(context.bot_data, link["party"], link["item"], rate)
        store.unlink_message(message.chat_id, message.reply_to_message.message_id)
        await message.reply_text(
            f"📒 Saved in the rates sheet: {link['item']} for {link['party'].strip()} = {inr(rate)}")
        return

    # A reply to a slip: a correction to it
    if link and link["kind"] == "draft":
        draft = store.draft(draft_id)
        if draft is None:
            await message.reply_text("This slip was already handled.")
            return
        await _revise(message, context, draft_id, draft, text)
        return

    await _ask_assistant(message, context, text)


async def _revise(message: Message, context, draft_id: str, draft: Draft, text: str) -> None:
    reader: SlipReader = context.bot_data["reader"]
    await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
    slip = await _model_call(message, reader.revise(_store(context).photo(draft_id), draft.slip, text),
                             "😕 I couldn't apply that change. Please try again.")
    if slip is not None:
        await _show_slip(message, context, draft_id, slip)


async def _ask_assistant(message: Message, context, text: str) -> None:
    """Anything else: the assistant looks things up, answers, or proposes a change."""
    store = _store(context)
    draft_id = store.last_draft(message.chat_id)
    draft = store.draft(draft_id) if draft_id else None
    assistant: Assistant = context.bot_data["assistant"]
    await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
    history = store.conversation(message.chat_id)
    outcome = await _model_call(message, assistant.handle(text, draft.slip if draft else None, history),
                                "😕 I couldn't work that out. Please try again.")
    if outcome is None:
        return
    store.remember_exchange(message.chat_id, text, outcome.summary() or HELP)
    if outcome.slip_correction and draft:
        await _revise(message, context, draft_id, draft, text)
    if outcome.rate_changes:
        await _propose_rates(message, store, outcome.rate_changes)
    if not (outcome.slip_correction or outcome.rate_changes):
        await message.reply_text(outcome.reply or HELP)


async def _propose_rates(message: Message, store: Store, changes: list[RateChange]) -> None:
    change_id = uuid.uuid4().hex[:12]
    store.put_rate_updates(change_id, [(c.party, c.item, str(c.new)) for c in changes])
    lines = "\n".join(
        f"• {escape(c.party.strip())} · <b>{escape(c.item)}</b>: "
        + (f"{inr(c.old)} → {inr(c.new)}" if c.old is not None else f"new rate {inr(c.new)}")
        for c in changes)
    await message.reply_text(
        f"💰 Change the rates sheet?\n{lines}\n\n<i>Slips waiting for the next Tally file will use the new "
        f"rate{'s' if len(changes) > 1 else ''}.</i>",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Change", callback_data=f"rc_ok:{change_id}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"rc_no:{change_id}"),
        ]]))


@allowed_only
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_reply(update.message, context, update.message.text)


@allowed_only
async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A voice note in any language: written down, then handled like typed text."""
    message = update.message
    if message.voice.duration > MAX_VOICE_SECONDS:
        await message.reply_text("🎤 Please keep voice notes under 2 minutes.")
        return
    await context.bot.send_chat_action(message.chat_id, ChatAction.TYPING)
    ogg = bytes(await (await message.voice.get_file()).download_as_bytearray())
    store = _store(context)
    _, draft_id = _reply_target(message, store)
    draft = store.draft(draft_id) if draft_id else None
    reader: SlipReader = context.bot_data["reader"]
    heard = await _model_call(message, reader.transcribe(ogg, draft.slip if draft else None),
                              "😕 I couldn't understand that voice note. Please try again, or type it.")
    if heard is None:
        return
    transcript, english = heard
    if not transcript:
        await message.reply_text("🎤 I couldn't make out that voice note. Please try again, or type it.")
        return
    log.info("Voice note heard as: %s", english or transcript)
    await _handle_reply(message, context, f"{transcript} (in English: {english})" if english else transcript)


@allowed_only
async def on_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stickers, videos, files... anything else gets an answer rather than silence."""
    if update.message:
        await update.message.reply_text(CAN_HANDLE)


@allowed_only
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    action, draft_id = query.data.split(":", 1)
    store = _store(context)
    draft = store.draft(draft_id)
    if draft is None:
        await _answer(query, "This slip was already handled.")
        await query.edit_message_reply_markup(reply_markup=None)
        return
    masters: Masters = context.bot_data["masters"]

    if action == "no":
        await _answer(query, "Cancelled")
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("❌ Cancelled. This slip won't be exported.")
        _forget(store, draft, keep_photo=False)
        return

    slip = draft.slip
    blockers, _ = check_slip(slip, masters, today=_today(context))
    if blockers:
        await _answer(query, "Please fix the problems first.", show_alert=True)
        return
    if have := _already_have(store, slip):   # e.g. the same photo sent twice, both saved
        await _answer(query, have, show_alert=True)
        return
    if action == "ok" and (dup := store.find_duplicate(slip)):
        await _answer(query)
        await query.message.reply_text(
            f"⚠️ This slip ({slip_heading(slip)}) was {dup}. Save it again?",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("Yes, save again", callback_data=f"force:{draft_id}"),
                InlineKeyboardButton("❌ Cancel", callback_data=f"no:{draft_id}"),
            ]]),
        )
        return

    store.add_pending(draft_id, slip, query.message.chat_id)
    store.learn(slip)
    await _answer(query, "Saved")
    if draft.preview_message_id:
        await _clear_buttons(context.bot, query.message.chat_id, draft.preview_message_id)
    if action == "force":
        await query.edit_message_reply_markup(reply_markup=None)
    _forget(store, draft, keep_photo=True)   # the photo stays with the saved slip
    await query.message.reply_text("✅ Saved.")
    await _handle_rates(query.message, context, draft_id, slip)


async def _handle_rates(message: Message, context, draft_id: str, slip: Slip) -> None:
    """Rates typed in a correction go to the sheet (asking first if they change an existing
    rate); items with no rate at all are asked for now, so Saturday's file isn't held up."""
    store = _store(context)
    table = await _rate_table(context.bot_data)
    given = {l.item: Decimal(str(l.rate_override)) for l in slip.lines if l.rate_override is not None}
    new = {i: r for i, r in given.items() if (slip.party, i) not in table}
    changed = {i: r for i, r in given.items() if (slip.party, i) in table and table[(slip.party, i)].rate != r}

    for item, rate in new.items():
        await _set_rate(context.bot_data, slip.party, item, rate)
    if new:
        store.update_pending(draft_id, _drop_overrides(slip, set(new)))   # the sheet is the source now
        await message.reply_text("📒 Added to the rates sheet:\n" + "\n".join(
            f"• {item}: {inr(rate)}" for item, rate in new.items()))

    if changed:
        store.put_rate_updates(draft_id, [(slip.party, item, str(rate)) for item, rate in changed.items()])
        lines = "\n".join(f"• {item}: {inr(table[(slip.party, item)].rate)} → {inr(rate)}"
                          for item, rate in changed.items())
        await message.reply_text(
            f"You gave a different rate than the sheet:\n{lines}\n\nSave as the new rate for {slip.party.strip()}?",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("💾 Save as new rate", callback_data=f"save:{draft_id}"),
                InlineKeyboardButton("Just this slip", callback_data=f"keep:{draft_id}"),
            ]]),
        )

    for item in missing_rates(slip, table):
        asked = await message.reply_text(
            f"❓ What's the rate of <b>{escape(item)}</b> for <b>{escape(slip.party.strip())}</b>?\n"
            "Reply to this message with the rate, e.g. 2.50",
            parse_mode=ParseMode.HTML)
        store.link_message(message.chat_id, asked.message_id, kind="rate_q", party=slip.party, item=item)


def _drop_overrides(slip: Slip, items: set[str]) -> Slip:
    return slip.model_copy(update={"lines": [
        l.model_copy(update={"rate_override": None}) if l.item in items else l for l in slip.lines]})


@allowed_only
async def on_rate_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    action, draft_id = query.data.split(":", 1)
    store = _store(context)
    updates = store.pop_rate_updates(draft_id)
    await query.edit_message_reply_markup(reply_markup=None)
    if updates is None:
        await _answer(query, "Already done.")
        return
    if action == "keep":
        await _answer(query, "Sheet left unchanged")
        await query.message.reply_text("OK, that rate is used for this slip only.")
        return
    for party, item, rate in updates:
        await _set_rate(context.bot_data, party, item, Decimal(rate))
    for p in store.pending():
        if p.draft_id == draft_id:
            store.update_pending(draft_id, _drop_overrides(p.slip, {item for _, item, _ in updates}))
    await _answer(query, "Saved")
    await query.message.reply_text("💾 Rates sheet updated.")


@allowed_only
async def on_rate_change_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """✅/❌ on rate changes the assistant proposed."""
    query = update.callback_query
    action, change_id = query.data.split(":", 1)
    updates = _store(context).pop_rate_updates(change_id)
    await _clear_buttons(context.bot, query.message.chat_id, query.message.message_id)
    if updates is None:
        await _answer(query, "Already done.")
        return
    if action == "rc_no":
        await _answer(query, "Cancelled")
        await query.message.reply_text("OK, the rates sheet is unchanged.")
        return
    for party, item, rate in updates:
        await _set_rate(context.bot_data, party, item, Decimal(rate))
    await _answer(query, "Saved")
    await query.message.reply_text(f"💾 Rates sheet updated ({len(updates)} rate{'s' if len(updates) > 1 else ''}).")


# -- the weekly file ---------------------------------------------------------------

async def run_batch(bot: Bot, bot_data: dict, *, empty_notice_to: list[int]) -> None:
    """Price every saved slip from the rates sheet now and send one Tally file per chat."""
    async with bot_data["batch_lock"]:
        store: Store = bot_data["store"]
        masters: Masters = bot_data["masters"]
        settings: Settings = bot_data["settings"]
        pending = store.pending()
        if not pending:
            for chat_id in empty_notice_to:
                await bot.send_message(chat_id, "📭 No slips this week, so no Tally file.")
            return

        batch = build_batch(pending, masters, await _rate_table(bot_data))
        now = datetime.now(settings.timezone)
        chats = sorted({p.chat_id for p in pending})
        for chat_id in chats:
            ready = [(p, v) for p, v in batch.ready if p.chat_id == chat_id]
            held = [(p, why) for p, why in batch.held if p.chat_id == chat_id]
            if ready:
                name = batch_name(now) + (f"_{chat_id}" if len(chats) > 1 else "")
                xml = build_import_xml([v for _, v in ready], masters)
                await bot.send_document(chat_id, document=xml, filename=f"{name}.xml",
                                        caption=f"🧾 Tally file: {len(ready)} invoice(s). Details below.")
                await _send_long(bot, chat_id, render_batch_summary([v for _, v in ready]))
                store.archive_batch(name, ready, xml)
            if held:
                text = ["⏸ <b>Not in the file yet</b> (fix the rates sheet, then tap Try again):", ""]
                for p, why in held:
                    text.append(f"• {escape(slip_heading(p.slip))}")
                    text += [f"    – {escape(w)}" for w in why]
                await _send_long(bot, chat_id, "\n".join(text), reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("🔄 Try again", callback_data="retry:batch")]]))


async def weekly_run(bot: Bot, bot_data: dict) -> None:
    settings: Settings = bot_data["settings"]
    await run_batch(bot, bot_data, empty_notice_to=sorted(settings.allowed_user_ids))
    bot_data["store"].set_last_batch_run(datetime.now(settings.timezone))


async def weekly_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    """(Computer only) Runs every few minutes: make the weekly file if it's due, including a
    run missed while the computer was off. On Lambda a schedule calls weekly_run instead."""
    bot_data = context.application.bot_data
    settings: Settings = bot_data["settings"]
    store: Store = bot_data["store"]
    if is_due(datetime.now(settings.timezone), store.last_batch_run(), settings.batch_weekday, settings.batch_time):
        log.info("Weekly export due (last run: %s)", store.last_batch_run())
        await weekly_run(context.bot, bot_data)


@allowed_only
async def export_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Making the Tally file from the rates sheet now…")
    await run_batch(context.bot, context.bot_data, empty_notice_to=[update.effective_chat.id])


@allowed_only
async def on_retry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await _answer(query, "Trying again…")
    await query.edit_message_reply_markup(reply_markup=None)
    await run_batch(context.bot, context.bot_data, empty_notice_to=[query.message.chat_id])


@allowed_only
async def see_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The /see page inside Telegram; in text where there's no page (the bot on a computer)."""
    chat_id = update.effective_chat.id
    store = _store(context)
    settings: Settings = context.bot_data["settings"]
    pending = [p for p in store.pending() if p.chat_id == chat_id]
    if settings.webapp_url and update.effective_chat.type == "private":
        last = store.exports(chat_id, limit=1)
        summary = (f"⏳ {len(pending)} slip{'s' if len(pending) != 1 else ''} waiting for the next Tally file"
                   + (f"\n📦 Last file: {_file_time(last[0].name)}" if last else ""))
        await update.message.reply_text(
            f"{summary}\n\nOpen to see the challans in each file, and to fix or delete waiting slips.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(
                "📋 Open", web_app=WebAppInfo(settings.webapp_url))]]))
        return
    batch = build_batch(pending, context.bot_data["masters"], await _rate_table(context.bot_data))
    files = store.exports(chat_id, limit=SEE_FILES)
    await _send_long(context.bot, chat_id, render_see(files, batch.ready, batch.held))


@allowed_only
async def pending_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pending = [p for p in _store(context).pending() if p.chat_id == update.effective_chat.id]
    if not pending:
        await update.message.reply_text("No slips saved yet.")
        return
    table = await _rate_table(context.bot_data)
    rows = [f"🗂 <b>{len(pending)} slip(s) saved:</b>"]
    for p in pending:
        missing = missing_rates(p.slip, table)
        rows.append(f"• {escape(slip_heading(p.slip))}: {len(p.slip.lines)} item(s)"
                    + (f" ⚠️ no rate: {escape(', '.join(missing))}" if missing else ""))
    await _send_long(context.bot, update.effective_chat.id, "\n".join(rows))


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    if isinstance(context.error, NetworkError) and update is None:
        log.warning("Telegram network hiccup (will retry): %s", context.error)
        return
    log.exception("Unhandled error", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("⚠️ Something went wrong. Please try again.")


# -- wiring ------------------------------------------------------------------------

def build_application(settings: Settings, *, polling: bool) -> Application:
    """The bot with all its handlers. polling=True for a computer (uses Telegram long polling
    and checks the weekly schedule itself); False for a webhook (Lambda)."""
    if not settings.telegram_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set (see .env.example).")
    if not settings.openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (see .env.example).")
    masters = load_masters(settings.masters_path, settings.hints_path)
    store = make_store(settings)
    rates = make_rate_source(settings, masters)

    builder = Application.builder().token(settings.telegram_token)
    if polling:
        builder = builder.concurrent_updates(True)   # several photos at once are read in parallel
    else:
        builder = builder.updater(None)
    app = builder.build()
    reader = SlipReader(masters, settings, learned_spellings=store.learned_spellings,
                        customer_items=lambda: customer_items(rates.table()))
    app.bot_data.update(
        settings=settings,
        masters=masters,
        store=store,
        rates=rates,
        batch_lock=asyncio.Lock(),
        reader=reader,
        assistant=Assistant(masters, settings, rates.table, reader.client, reader.provider,
                            slips=store.slip_records),
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("pending", pending_cmd))
    app.add_handler(CommandHandler("export", export_cmd))
    app.add_handler(CommandHandler("see", see_cmd))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^(ok|no|force):"))
    app.add_handler(CallbackQueryHandler(on_rate_button, pattern=r"^(save|keep):"))
    app.add_handler(CallbackQueryHandler(on_rate_change_button, pattern=r"^rc_(ok|no):"))
    app.add_handler(CallbackQueryHandler(on_retry, pattern=r"^retry:"))
    app.add_handler(MessageHandler(filters.ALL, on_other))   # last: whatever nothing above took
    app.add_error_handler(on_error)
    if polling:
        # Check every 5 minutes; the first check shortly after start-up catches a missed Saturday.
        app.job_queue.run_repeating(weekly_check, interval=300, first=30)
    return app


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs the bot token in URLs
    logging.getLogger("apscheduler").setLevel(logging.WARNING)  # a line per 5-minute check

    settings = load_settings()
    if not settings.allowed_user_ids:
        log.warning("ALLOWED_TELEGRAM_USER_IDS is empty: the bot will refuse everyone. "
                    "Send /start to the bot to see your user id.")
    app = build_application(settings, polling=True)
    store: Store = app.bot_data["store"]
    if store.last_batch_run() is None:
        # First ever start: nothing was missed, so the first weekly file is the next Saturday's.
        store.set_last_batch_run(datetime.now(settings.timezone))

    log.info("Bot running (polling): %d stock items, %d customers, %d rates; weekly file %s %s %s.",
             len(app.bot_data["masters"].items), len(app.bot_data["masters"].parties),
             len(app.bot_data["rates"].table()), DAYS[settings.batch_weekday], settings.batch_time,
             settings.timezone)
    # Messages sent while the bot was off (up to Telegram's 24-hour limit) are processed now.
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=False)


if __name__ == "__main__":
    main()
