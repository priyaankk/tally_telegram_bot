"""A tiny key-value store with two backends, so the bot runs the same on a laptop (files)
and on AWS Lambda (DynamoDB). Records live under a namespace; each is a JSON dict or bytes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol


class KV(Protocol):
    def get(self, ns: str, key: str) -> dict | None: ...
    def put(self, ns: str, key: str, value: dict) -> None: ...
    def delete(self, ns: str, key: str) -> None: ...
    def items(self, ns: str) -> list[tuple[str, dict]]: ...
    def get_bytes(self, ns: str, key: str) -> bytes | None: ...
    def put_bytes(self, ns: str, key: str, value: bytes) -> None: ...


def _file_name(key: str) -> str:
    # Keys can contain anything (customer names, "|"); keep file names safe and reversible.
    return re.sub(r"[^A-Za-z0-9._-]", lambda m: f"%{ord(m.group()):04x}", key)


def _key_from_file(name: str) -> str:
    return re.sub(r"%([0-9a-f]{4})", lambda m: chr(int(m.group(1), 16)), name)


class FileKV:
    """data/<namespace>/<key>.json (or .bin)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, ns: str, key: str, ext: str) -> Path:
        folder = self.root / ns
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{_file_name(key)}.{ext}"

    def get(self, ns, key):
        p = self._path(ns, key, "json")
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, ns, key, value):
        p = self._path(ns, key, "json")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
        tmp.replace(p)

    def delete(self, ns, key):
        for ext in ("json", "bin"):
            self._path(ns, key, ext).unlink(missing_ok=True)

    def items(self, ns):
        folder = self.root / ns
        if not folder.exists():
            return []
        return sorted((_key_from_file(p.stem), json.loads(p.read_text())) for p in folder.glob("*.json"))

    def get_bytes(self, ns, key):
        p = self._path(ns, key, "bin")
        return p.read_bytes() if p.exists() else None

    def put_bytes(self, ns, key, value):
        self._path(ns, key, "bin").write_bytes(value)


class DynamoKV:
    """One DynamoDB table with key (pk = namespace, sk = key). JSON goes in `data`,
    bytes in `blob` (DynamoDB items are limited to 400 KB)."""

    def __init__(self, table_name: str, region: str | None = None) -> None:
        import boto3  # only needed on AWS

        self.table = boto3.resource("dynamodb", region_name=region).Table(table_name)

    def get(self, ns, key):
        item = self.table.get_item(Key={"pk": ns, "sk": key}).get("Item")
        return json.loads(item["data"]) if item and "data" in item else None

    def put(self, ns, key, value):
        self.table.put_item(Item={"pk": ns, "sk": key, "data": json.dumps(value, ensure_ascii=False)})

    def delete(self, ns, key):
        self.table.delete_item(Key={"pk": ns, "sk": key})

    def items(self, ns):
        from boto3.dynamodb.conditions import Key

        out, kwargs = [], {"KeyConditionExpression": Key("pk").eq(ns)}
        while True:
            page = self.table.query(**kwargs)
            out += [(i["sk"], json.loads(i["data"])) for i in page["Items"] if "data" in i]
            if "LastEvaluatedKey" not in page:
                return out
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def get_bytes(self, ns, key):
        item = self.table.get_item(Key={"pk": ns, "sk": key}).get("Item")
        return bytes(item["blob"]) if item and "blob" in item else None

    def put_bytes(self, ns, key, value):
        self.table.put_item(Item={"pk": ns, "sk": key, "blob": value})
