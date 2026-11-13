# Tally Telegram Bot

A private Telegram bot that turns photos of handwritten delivery slips (challans) into
Tally Sales invoices.

**During the week:** send a photo of a slip. The bot turns the photo upright, reads it with a
vision LLM (customer, date, challan number, items, quantities) and matches every line to an
exact Tally stock item. To fix anything, reply to its message by typing or with a voice note
("line 2 is 300", "date is 18/8"), then tap **✅ Save**. If an item has no rate for that
customer yet, the bot asks for it and writes it into the rates sheet.

**Weekly (Saturday 9 PM by default):** every saved slip is priced from the rates sheet *as it
is at that moment*, and one Tally XML file with all the week's invoices is sent on Telegram.
If the bot was off at that time, the file is sent as soon as it's back.

You can also just ask it things like "ACME S3 screw rate 3.50" or "where is ACME challan 14?".
An LLM assistant with tools looks up rates and slips, and proposes rate changes that only
happen after you tap a button.

Commands: `/pending` (slips waiting), `/export` (make the file now), `/see` (Tally files and
waiting slips, as a Telegram Mini App when running on Lambda).

> All customer and item names in this repo (ACME TRADERS, S3 SCREW, …) are made-up examples.

## How it works

| File | What it does |
|---|---|
| `tally_agent/bot.py` | Telegram handlers: photos, replies, voice notes, buttons, weekly job |
| `tally_agent/extract.py` | Reads a slip photo with a vision LLM (OpenRouter), using a JSON schema |
| `tally_agent/models.py` | The `Slip` model, checks, learned spellings, pricing into a `Voucher` |
| `tally_agent/assistant.py` | The tool-using assistant for rate lookups, rate changes and slip lookups |
| `tally_agent/rates.py` | The rates sheet: Google Sheets or a local CSV |
| `tally_agent/tally_xml.py` | Builds the Tally import XML (UTF-16, the same format Tally 9 exports) |
| `tally_agent/tally_data.py`, `tally_import.py` | Reads Tally's own XML exports to generate `config/masters.toml` |
| `tally_agent/store.py`, `kv.py` | Bot state, kept in files (laptop) or DynamoDB (Lambda) |
| `tally_agent/lambda_app.py`, `webapp.py` | AWS Lambda entry point (webhook) and the `/see` Mini App |
| `tally_agent/weekly.py` | The weekly batch |
| `tally_agent/cli.py` | Run the pipeline on image files without Telegram |
| `deploy/aws.py` | One-command deploy to AWS Lambda + DynamoDB + EventBridge Scheduler |

## Where things come from

| What | Source |
|---|---|
| Item, customer, ledger names | `config/masters.toml`, generated from Tally's own export (don't hand-edit). See `config/masters.example.toml` |
| Rates | a Google Sheet (tab **Rates**: Customer, Item, Rate, Last updated, Notes), or a CSV with the same columns (see `sheets/rates.example.csv`) |
| Slip spellings / glossary | `config/hints.toml` (hand-edited) plus spellings learned from confirmed slips, per customer |

Generate the masters from Tally: Display → List of Accounts → Alt+E → XML (Data Interchange),
All Masters. Then:

```bash
uv run python -m tally_agent.tally_import tally_export/Master.xml tally_export/DayBook.xml
```

## Setup

```bash
uv sync
cp .env.example .env        # fill in your own keys
cp config/masters.example.toml config/masters.toml   # or generate it from Tally (above)
cp sheets/rates.example.csv sheets/rates.csv         # if you're not using a Google Sheet
uv run python -m tally_agent.bot                     # runs with long polling
```

You need a Telegram bot token from @BotFather and an OpenRouter API key. Send `/start` to the
bot to see your Telegram user id, then put it in `ALLOWED_TELEGRAM_USER_IDS`.

To run it as a service on a Linux machine, see `deploy/tally-bot.service` (systemd user
unit). To run it on AWS Lambda instead, see `deploy/aws.py` (set your own AWS profile and region
at the top).

## Importing into Tally 9

Save the file, then Gateway → **Import of Data → Vouchers** → type its path. Each invoice keeps
its slip's date, the challan number goes in *Ref*, and Tally assigns voucher numbers.

## Development

```bash
uv run pytest
# Read slips without Telegram, optionally comparing with what was entered in Tally:
uv run python -m tally_agent.cli photo1.jpg photo2.jpg --check tally_export/DayBook.xml
```
