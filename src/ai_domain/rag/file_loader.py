from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_text_from_file(path: str | Path) -> str:
    file_path = Path(path)
    data = file_path.read_bytes()
    text = _decode_bytes(data)

    suffix = file_path.suffix.lower()
    if suffix in {".json"}:
        return _json_to_text(text)
    if suffix in {".jsonl"}:
        return _jsonl_to_text(text)
    return text


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "cp1251", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _jsonl_to_text(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            lines.append(raw)
            continue
        lines.append(_extract_text(obj))
    return "\n".join([line for line in lines if line])


def _json_to_text(text: str) -> str:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text
    return _extract_text(obj)


def _extract_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        parts = [_extract_text(v) for v in obj.values()]
        return "\n".join([p for p in parts if p])
    if isinstance(obj, Iterable):
        parts = [_extract_text(v) for v in obj]
        return "\n".join([p for p in parts if p])
    return str(obj)
