"""AWS Lambda entry point. One function, three kinds of calls:

1. Through the function URL:
   - from Telegram (the webhook): check the secret header, hand the update to a background run
     of this same function, and answer 200 straight away. Reading a slip takes up to a minute,
     and Telegram resends updates it doesn't get a quick answer for.
   - GET /app and POST /api: the /see page inside Telegram (see webapp.py).
2. The background run ({"update": ...}): process the update with the normal bot handlers.
3. From the weekly schedule ({"tally": "weekly"}): make and send the weekly Tally file.

The webhook part imports only what it needs and shows "typing…" itself, so a cold start
(the bot's libraries take ~3 s to load) doesn't delay the first sign of life.

Secrets come from SSM Parameter Store under $SSM_PREFIX (see deploy/aws.py).
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import urllib.request

import boto3

logging.getLogger().setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)   # it logs the bot token in URLs
log = logging.getLogger("tally_agent")

SECRET_NAMES = ("TELEGRAM_BOT_TOKEN", "OPENROUTER_API_KEY", "GOOGLE_SERVICE_ACCOUNT_JSON", "WEBHOOK_SECRET")
_secrets_loaded = False


def _load_secrets() -> None:
    """Once per container: secrets from SSM into the environment (load_settings reads them)."""
    global _secrets_loaded
    if _secrets_loaded:
        return
    prefix = os.environ["SSM_PREFIX"].rstrip("/")
    response = boto3.client("ssm").get_parameters(
        Names=[f"{prefix}/{n}" for n in SECRET_NAMES], WithDecryption=True)
    for p in response["Parameters"]:
        os.environ[p["Name"].rsplit("/", 1)[-1]] = p["Value"]
    if "GOOGLE_SERVICE_ACCOUNT_JSON" in os.environ:
        path = "/tmp/google-service-account.json"
        with open(path, "w") as f:
            f.write(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
        os.chmod(path, 0o600)
        os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"] = path
    _secrets_loaded = True


async def _process(update_json: dict, public_url: str | None) -> None:
    from telegram import Update

    from .bot import build_application
    from .config import load_settings

    if public_url:   # where the /see page is served
        os.environ.setdefault("WEBAPP_URL", f"{public_url}app")
    app = build_application(load_settings(), polling=False)
    async with app:
        await app.process_update(Update.de_json(update_json, app.bot))


async def _weekly() -> None:
    from .bot import build_application, weekly_run
    from .config import load_settings

    app = build_application(load_settings(), polling=False)
    async with app:
        await weekly_run(app.bot, app.bot_data)


def _show_typing(update: dict) -> None:
    """Show "typing…" for a message from an allowed user. Best effort: never blocks the hand-off."""
    message = update.get("message") or {}
    user_id = str((message.get("from") or {}).get("id", ""))
    allowed = {u.strip() for u in os.environ.get("ALLOWED_TELEGRAM_USER_IDS", "").split(",")}
    if not message or user_id not in allowed:
        return
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendChatAction",
        data=json.dumps({"chat_id": message["chat"]["id"], "action": "typing"}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=3).close()
    except Exception as e:  # noqa: BLE001
        log.warning("Couldn't show typing: %s", type(e).__name__)   # the URL holds the token


def _webhook(event: dict) -> dict:
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    given = headers.get("x-telegram-bot-api-secret-token", "")
    if not hmac.compare_digest(given, os.environ.get("WEBHOOK_SECRET", "")):
        log.warning("Rejected a webhook call without the right secret")
        return {"statusCode": 403, "body": "forbidden"}
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode()
    update = json.loads(body)
    _show_typing(update)
    public_url = f"https://{event['requestContext']['domainName']}/"
    boto3.client("lambda").invoke(
        FunctionName=os.environ["AWS_LAMBDA_FUNCTION_NAME"], InvocationType="Event",
        Payload=json.dumps({"update": update, "public_url": public_url}).encode())
    return {"statusCode": 200, "body": "ok"}


def _http(event: dict) -> dict:
    method = event["requestContext"].get("http", {}).get("method", "POST")
    path = event.get("rawPath") or "/"
    if method == "GET" and path == "/app":
        from .webapp import page
        return page()
    if method == "POST" and path == "/api":
        from .webapp import api
        body = event.get("body") or ""
        return api(base64.b64decode(body).decode() if event.get("isBase64Encoded") else body)
    return _webhook(event)


def handler(event: dict, context) -> dict | None:
    _load_secrets()
    if "requestContext" in event:            # 1. function URL: Telegram, or the /see page
        return _http(event)
    if "update" in event:                    # 2. background run
        asyncio.run(_process(event["update"], event.get("public_url")))
        return None
    if event.get("tally") == "weekly":       # 3. weekly schedule
        asyncio.run(_weekly())
        return None
    log.warning("Unknown event: %s", list(event))
    return None
