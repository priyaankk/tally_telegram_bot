"""Post a message in a chat with plain HTTP, for code that doesn't load the Telegram library
(the /see page)."""

from __future__ import annotations

import json
import logging
import urllib.request

log = logging.getLogger(__name__)


def tell_chat(bot_token: str, chat_id: int, text: str, *, quiet: bool = False) -> None:
    """Best effort: a failure is logged, never raised. quiet: no notification sound."""
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data=json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                         "disable_notification": quiet}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=5).close()
    except Exception as e:  # noqa: BLE001
        log.warning("Couldn't post in the chat: %s", type(e).__name__)   # the URL holds the token
