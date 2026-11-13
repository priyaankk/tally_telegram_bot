"""Read a slip photo with a vision LLM (via OpenRouter) and return a structured Slip."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from collections.abc import Callable

import numpy as np
import openai
from PIL import Image, ImageOps

from .config import OPENROUTER_BASE_URL, Masters, Settings
from .models import Slip, slip_json_schema

log = logging.getLogger(__name__)

MAX_IMAGE_EDGE = 2048
ORIENTATION_IMAGE_EDGE = 768
# Includes the model's reasoning. Normal readings use a few thousand; the cap also limits the
# credit OpenRouter reserves per request. A run-away answer hits it and is retried once.
MAX_OUTPUT_TOKENS = 8000
# A model that reasons in circles until it hits the cap tends to do it again on the same
# request, so each retry steps down this ladder instead of repeating it.
EFFORT_LADDER = ("high", "medium", "low")

SYSTEM_PROMPT = """\
You read handwritten delivery slips (challans) from a small manufacturing business in \
India and turn them into structured data that will be booked as Sales invoices in Tally.

How the slips are written:
- Small lined notepad pages. The first line is the customer, usually written \
"M/s <NAME>" (the "M/s" can look like "17/s").
- The date is day-first (DD/MM/YY): "16/7/26" is 2026-07-16. Never read it month-first. \
Return it as YYYY-MM-DD.
- The number PRINTED on the notepad page (small, typeset, at the top) is the challan \
number: put it in challan_number. Numbers written by hand at the top (often circled) are \
not the challan number; ignore them.
- Each item line is: quantity, unit, item description. "Pcs", "Ps", "Pc", "P." and similar \
squiggles after the quantity all mean pieces.
- The writer draws the digit 0 as a small "U"- or "∪"-shaped loop, often joined to the \
next zero: "25U" is 250, "100U" or "10UU" is 1000, "200U" is 2000. Count every loop as a \
zero. Quantities are almost always round numbers such as 100, 250, 400, 500, 1000, 2000.
- "(1 bag)", "(2 bags)" near the bottom is the number of bags.
- Ignore the signature.

How to fill the fields:
- item_text: the item description exactly as written, without the quantity or unit, \
e.g. "Hinge H4". Keep the writer's spelling; do not correct it.
- item: the stock item below that the line refers to. Names on slips are misspelt, \
abbreviated, reordered, or written in Hindi/Marathi/Gujarati words. Variant codes \
(S3/S5, N3/N4, H1/H4, 10MM/12MM ...) distinguish different products and must \
never be swapped or guessed. If no stock item clearly fits, use UNKNOWN rather than \
forcing a match.
- Strongly prefer the items the customer currently buys (listed below). A plain \
description such as "Pan Screw" or "Hex Nut" usually means the one matching item that \
customer buys.
- When the writing matches an "also written" spelling listed for a stock item, that \
spelling has been confirmed by a person: use that item.
- party: the customer ledger below, or UNKNOWN if none fits. party_raw is the name as written.
- confidence: high = clearly legible and clearly matched; medium = some doubt about the \
reading or the match; low = mostly a guess.
- rate_override: null unless a rate is explicitly written on that line.
- notes: a short note on anything a person should double-check. Empty string if nothing.
- If the image is not such a slip, set is_slip to false and leave lines empty.

Slips in Hindi:
- Some slips are written in Hindi (Devanagari script). Then the customer name and the item \
words are in Devanagari, while item codes (S3, N3, H4, B 1 ...) and quantities \
are in Latin letters and digits. The date and quantities follow the same rules as above.
- Read the Devanagari phonetically and match it to the English names below. Common words: \
स्क्रू = Screw, नट = Nut, बोल्ट = Bolt, क्लैंप = Clamp, हुक = Hook, क्लिप = Clip, \
पैन = Pan (S), हेक्स = Hex (N), भाई = Bhai.
- Customer names: transliterate and match the customer list, e.g. रमेश भाई = RAMESH BHAI. \
Text in brackets after the name is usually a place or a person; ignore it for matching.
- Item codes decide the item. Use the item with exactly that code (B 1 → B 1 BOLT) even \
when the customer has not bought it before; never change a code to fit the customer's list.
- Do not deliberate at length over hard handwriting: make your best reading, lower the \
confidence, and say what is uncertain in notes.
{glossary}
Items each customer currently buys:
{customer_items}

All stock items:
{items}

Customers:
{parties}
"""

REVISE_INSTRUCTIONS = """\
Here is the slip as it was read earlier:

{current}

The person who sent the slip says:
"{correction}"

Apply their correction and return the complete updated slip. Their word overrides what \
is on the image. If they give a rate for an item, put it in rate_override on that item's \
line(s). If they name a different item, pick the matching stock item. Keep every other \
field as it was unless the correction says otherwise.\
"""

TRANSCRIBE_PROMPT = """\
This voice note is from a person checking a handwritten delivery slip that was read for \
them. They may speak Gujarati, Hindi, English or a mix. Item codes are letters and numbers \
such as S3, N3, H4, B 1.

{slip}

Return what they said word for word, in the language and script they spoke (transcript), \
and an English translation (english). Write numbers, dates and item codes in digits and \
Latin letters, e.g. S3, 250, 10/10/26. The slip above is only to help you recognise \
names and codes: never fill in words from it that you cannot clearly hear. If you cannot \
make out what is said, return empty strings.\
"""
TRANSCRIPT_SCHEMA = {"type": "object",
                     "properties": {"transcript": {"type": "string"}, "english": {"type": "string"}},
                     "required": ["transcript", "english"], "additionalProperties": False}

UPRIGHT_QUESTION = ("Is the handwritten text in this photo upright, i.e. can it be read "
                    "normally left-to-right without turning the image?")
UPRIGHT_SCHEMA = {"type": "object", "properties": {"upright": {"type": "boolean"}},
                  "required": ["upright"], "additionalProperties": False}


class ExtractionError(Exception):
    pass


def encode_jpeg(img: Image.Image, edge: int = MAX_IMAGE_EDGE) -> bytes:
    img = img.copy()
    img.thumbnail((edge, edge))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


def load_image(data: bytes) -> Image.Image:
    """Decode, apply the camera's EXIF rotation, convert to RGB."""
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def ruled_lines_horizontal(img: Image.Image) -> bool:
    """True if the notepad's ruled lines run left-right (text upright or upside down),
    False if they run top-bottom (photo taken sideways).

    Long straight lines make whole rows (or columns) of the edge map uniformly strong,
    so the per-row edge strength varies much more across rows than across columns."""
    g = np.asarray(ImageOps.grayscale(img).resize((800, 800)), dtype=np.float32)
    rows = np.abs(np.diff(g, axis=0)).mean(axis=1)
    cols = np.abs(np.diff(g, axis=1)).mean(axis=0)
    return rows.std() / (rows.mean() + 1e-6) >= cols.std() / (cols.mean() + 1e-6)


def build_system_prompt(masters: Masters, learned: dict[str, list[str]],
                        customer_items: dict[str, list[str]]) -> str:
    def line(name: str, spellings: list[str]) -> str:
        seen = list(dict.fromkeys(spellings))  # dedupe, keep order
        return f"- {name}" + (f" (also written: {', '.join(seen)})" if seen else "")

    items = "\n".join(line(i.name, [*i.aliases, *learned.get(i.name, [])]) for i in masters.items)
    parties = "\n".join(line(p.ledger, list(p.aliases)) for p in masters.parties)
    by_customer = "\n".join(f"- {party}: {', '.join(sorted(names))}"
                            for party, names in sorted(customer_items.items()))
    glossary = ("\nWords used in item names:\n" + "\n".join(f"- {g}" for g in masters.glossary) + "\n"
                if masters.glossary else "")
    return SYSTEM_PROMPT.format(items=items, parties=parties or "- (none configured)",
                                customer_items=by_customer or "- (none yet)", glossary=glossary)


class SlipReader:
    def __init__(self, masters: Masters, settings: Settings, *,
                 learned_spellings: Callable[[], dict[str, list[str]]] = dict,
                 customer_items: Callable[[], dict[str, list[str]]] = dict) -> None:
        self.client = openai.AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=settings.openrouter_api_key)
        self.masters = masters
        self.learned_spellings = learned_spellings
        self.customer_items = customer_items
        self.model = settings.llm_model
        self.effort = settings.llm_reasoning_effort
        self.schema = slip_json_schema(masters)
        # require_parameters: only route to providers that honour the JSON schema.
        self.provider: dict = {"provider": {"require_parameters": True}}
        if settings.llm_provider:
            self.provider["provider"] |= {"order": [settings.llm_provider], "allow_fallbacks": False}

    # -- orientation ------------------------------------------------------------
    async def _is_upright(self, img: Image.Image) -> bool | None:
        try:
            answer = await self._json_call(
                [{"role": "user", "content": [self._image_part(encode_jpeg(img, ORIENTATION_IMAGE_EDGE)),
                                              {"type": "text", "text": UPRIGHT_QUESTION}]}],
                UPRIGHT_SCHEMA, effort="low", max_tokens=3000)
            return answer["upright"]
        except ExtractionError:
            return None

    async def upright(self, img: Image.Image) -> Image.Image:
        """Turn the photo so the writing is upright. The ruled lines tell sideways from not;
        the model then picks between the two remaining turns, which it does reliably."""
        if ruled_lines_horizontal(img):
            candidates = [img, img.rotate(180)]
        else:
            candidates = [img.rotate(-90, expand=True), img.rotate(90, expand=True)]  # 90° cw, ccw
        verdicts = await asyncio.gather(*(self._is_upright(c) for c in candidates))
        if verdicts == [False, True]:
            return candidates[1]
        return candidates[0]   # also the fallback when the model is unsure

    # -- reading ----------------------------------------------------------------
    async def read(self, image: Image.Image) -> tuple[bytes, Slip]:
        """Returns (the upright JPEG that was read, the slip). Keep the JPEG for revisions."""
        jpeg = encode_jpeg(await self.upright(image))
        return jpeg, await self._read_slip([self._image_part(jpeg), {"type": "text", "text": "Read this slip."}])


    async def revise(self, jpeg: bytes, current: Slip, correction: str) -> Slip:
        text = REVISE_INSTRUCTIONS.format(
            current=json.dumps(current.model_dump(), ensure_ascii=False, indent=2),
            correction=correction,
        )
        return await self._read_slip([self._image_part(jpeg), {"type": "text", "text": text}])

    async def transcribe(self, ogg: bytes, current: Slip | None) -> tuple[str, str]:
        """A Telegram voice note (OGG/Opus) -> (what was said, as spoken; English translation)."""
        if current is None:
            context = "No slip is open right now."
        else:
            lines = "\n".join(f"- {l.quantity:g} {l.item} (written \"{l.item_text}\")" for l in current.lines)
            context = (f"The slip currently says: customer {current.party.strip()}, date {current.date}, "
                       f"challan {current.challan_number}, lines:\n{lines}")
        audio = {"type": "input_audio",
                 "input_audio": {"data": base64.standard_b64encode(ogg).decode("ascii"), "format": "ogg"}}
        answer = await self._json_call(
            [{"role": "user", "content": [audio, {"type": "text", "text": TRANSCRIBE_PROMPT.format(slip=context)}]}],
            TRANSCRIPT_SCHEMA, effort="low", max_tokens=3000)
        return answer["transcript"].strip(), answer["english"].strip()

    @staticmethod
    def _image_part(jpeg: bytes) -> dict:
        b64 = base64.standard_b64encode(jpeg).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    async def _read_slip(self, content: list[dict]) -> Slip:
        system = build_system_prompt(self.masters, self.learned_spellings(), self.customer_items())
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        data = await self._json_call(messages, self.schema, effort=self.effort,
                                     max_tokens=MAX_OUTPUT_TOKENS, attempts=3)
        try:
            return Slip.model_validate(data)
        except ValueError as e:
            raise ExtractionError("The model's answer was not in the expected format.") from e

    async def _json_call(self, messages: list[dict], schema: dict, *, effort: str,
                         max_tokens: int, attempts: int = 2) -> dict:
        """One structured-output call. Retries when the model runs out of tokens (it occasionally
        reasons in circles) or returns something unparseable, each time with less reasoning."""
        problem = "no answer"
        lower = EFFORT_LADDER[EFFORT_LADDER.index(effort) + 1:] if effort in EFFORT_LADDER else EFFORT_LADDER[1:]
        efforts = [effort, *lower]
        for attempt in range(attempts):
            effort = efforts[min(attempt, len(efforts) - 1)]
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "answer", "strict": True, "schema": schema}},
                extra_body={"reasoning": {"effort": effort}, **self.provider},
            )
            choice = response.choices[0] if response.choices else None
            if choice is None:
                problem = "no answer"
            elif choice.finish_reason == "length":
                problem = "the answer was cut off"
            elif choice.message.refusal or not choice.message.content:
                problem = "an empty answer"
            else:
                try:
                    return json.loads(choice.message.content)
                except json.JSONDecodeError:
                    problem = "an unreadable answer"
            log.warning("LLM attempt %d/%d (effort %s) gave %s", attempt + 1, attempts, effort, problem)
        raise ExtractionError(f"the reading service gave {problem}")
