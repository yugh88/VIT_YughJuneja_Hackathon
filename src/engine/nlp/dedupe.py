"""Stage 3 - near-duplicate detection with MinHash LSH.

128 permutations, 5-word shingles (5-character shingles for scripts written
without spaces), Jaccard >= 0.8 counts as a duplicate. Syndicated wire copies
collapse onto the first story: the first gets a growing ``syndication_count`` and
later copies are reported as duplicates of it. Entries expire after ``ttl_s`` so
memory stays bounded on a long-running stream.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

_WORD = re.compile(r"\w+", re.UNICODE)
NO_SPACE_LANGS = {"zh", "ja", "th", "lo", "km", "my"}


def shingles(text: str, lang: str = "en", k: int = 5) -> set[str]:
    t = text.lower()
    if lang in NO_SPACE_LANGS:
        chars = "".join(_WORD.findall(t))
        return {chars[i : i + k] for i in range(max(1, len(chars) - k + 1))}
    words = _WORD.findall(t)
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


@dataclass(frozen=True)
class DedupResult:
    is_duplicate: bool
    cluster_id: str  # doc id of the first story in the cluster
    syndication_count: int  # copies seen so far, including this one
    similarity: float  # estimated Jaccard to the cluster head (1.0 for a new story)


class Deduplicator:
    def __init__(
        self, threshold: float = 0.8, num_perm: int = 128, ttl_s: float = 72 * 3600, max_items: int = 200_000
    ) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self.ttl_s = ttl_s
        self.max_items = max_items
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._items: OrderedDict[str, tuple[float, MinHash]] = OrderedDict()
        self._cluster_of: dict[str, str] = {}
        self._count: dict[str, int] = {}
        self.seen = 0
        self.duplicates = 0

    def _minhash(self, sh: set[str]) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        for s in sh:
            m.update(s.encode("utf-8"))
        return m

    def _evict(self, now: float) -> None:
        while self._items:
            key, (ts, _) = next(iter(self._items.items()))
            if now - ts < self.ttl_s and len(self._items) <= self.max_items:
                break
            self._items.popitem(last=False)
            self._lsh.remove(key)
            self._count.pop(key, None)
            self._cluster_of.pop(key, None)

    def check(self, doc_id: str, text: str, lang: str = "en", now: float | None = None) -> DedupResult:
        now = time.time() if now is None else now
        self.seen += 1
        self._evict(now)
        sh = shingles(text, lang)
        if not sh:
            return DedupResult(False, doc_id, 1, 1.0)
        m = self._minhash(sh)
        best, best_sim = None, 0.0
        for cand in self._lsh.query(m):
            sim = m.jaccard(self._items[cand][1])
            if sim > best_sim:
                best, best_sim = cand, sim
        if best is not None and best_sim >= self.threshold:
            head = self._cluster_of.get(best, best)
            self._count[head] = self._count.get(head, 1) + 1
            self.duplicates += 1
            return DedupResult(True, head, self._count[head], best_sim)
        if doc_id in self._items:  # same id re-sent (replay lap): treat as duplicate of itself
            return DedupResult(True, doc_id, self._count.get(doc_id, 1), 1.0)
        self._lsh.insert(doc_id, m)
        self._items[doc_id] = (now, m)
        self._cluster_of[doc_id] = doc_id
        self._count[doc_id] = 1
        return DedupResult(False, doc_id, 1, 1.0)

    def count_of(self, doc_id: str) -> int:
        """Copies seen so far of the cluster headed by ``doc_id``."""
        return self._count.get(self._cluster_of.get(doc_id, doc_id), 1)

    @property
    def duplicate_ratio(self) -> float:
        return self.duplicates / self.seen if self.seen else 0.0
