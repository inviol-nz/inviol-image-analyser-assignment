from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Generic, MutableMapping, TypeVar

T = TypeVar("T")


class AnalysisCache(Generic[T]):
    """
    In-memory LRU cache to avoid re-running inference on identical images in the short term.
    """

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self._store: MutableMapping[str, T] = OrderedDict()

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get(self, data: bytes) -> T | None:
        key = self._hash_bytes(data)
        try:
            value = self._store.pop(key)
        except KeyError:
            return None
        # Mark as recently used
        self._store[key] = value
        return value

    def set(self, data: bytes, value: T) -> None:
        key = self._hash_bytes(data)
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self.max_size:
            # Evict least recently used
            self._store.popitem(last=False)
        self._store[key] = value
