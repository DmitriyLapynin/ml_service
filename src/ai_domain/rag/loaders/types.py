from __future__ import annotations

from typing import Any, Dict, TypedDict


class DocumentLike(TypedDict):
    text: str
    metadata: Dict[str, Any]
