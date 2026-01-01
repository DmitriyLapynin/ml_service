from __future__ import annotations

import time
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Callable, Any


@dataclass(frozen=True)
class CachedClient:
    client: Any
    expires_at: float


class TTLRUClientCache:
    """
    In-memory TTL + LRU cache.
    - TTL: клиент живёт ограниченное время
    - LRU: если ключей слишком много, выкидываем самые старые по использованию
    """
    def __init__(self, *, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, CachedClient]" = OrderedDict()

    @staticmethod
    def _fingerprint(api_key: str) -> str:
        # Не логировать и не хранить “сырой” ключ — только отпечаток.
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    def get_or_create(self, *, api_key: str, factory: Callable[[str], Any]) -> Any:
        now = time.time()
        fp = self._fingerprint(api_key)

        with self._lock:
            # очистка протухших
            if fp in self._items:
                entry = self._items[fp]
                if entry.expires_at > now:
                    # LRU: поднимаем наверх
                    self._items.move_to_end(fp)
                    return entry.client
                else:
                    # expired
                    del self._items[fp]

            # создать новый
            client = factory(api_key)
            self._items[fp] = CachedClient(client=client, expires_at=now + self.ttl_seconds)
            self._items.move_to_end(fp)

            # LRU eviction
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

            return client

    def invalidate(self, api_key: str) -> None:
        fp = self._fingerprint(api_key)
        with self._lock:
            self._items.pop(fp, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()