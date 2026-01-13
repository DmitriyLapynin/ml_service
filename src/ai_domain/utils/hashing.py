from __future__ import annotations

import hashlib
from typing import Iterable, Sequence, Dict


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_message_contents(contents: Iterable[str]) -> str:
    joined = "\n".join(contents)
    return hash_text(joined)


def hash_text_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def messages_fingerprint(messages: Sequence[Dict[str, str]]) -> Dict[str, object]:
    total_chars = 0
    roles = []
    parts = []
    for m in messages:
        role = m.get("role") or ""
        content = m.get("content") or ""
        total_chars += len(content)
        roles.append(role)
        parts.append(f"{role}:{content}")
    digest = hash_text_short("|".join(parts))
    return {"count": len(messages), "total_chars": total_chars, "roles": roles, "digest": digest}
