from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List

from .sokoban import SokobanState


@dataclass
class ReplayItem:
    level_id: str
    state: SokobanState


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.rng = random.Random(seed)
        self._items: List[ReplayItem] = []
        self._by_level: Dict[str, List[int]] = {}

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: ReplayItem) -> None:
        if len(self._items) >= self.capacity:
            old = self._items.pop(0)
            self._rebuild_index_after_pop(old.level_id)
        self._items.append(item)
        idx = len(self._items) - 1
        self._by_level.setdefault(item.level_id, []).append(idx)

    def _rebuild_index_after_pop(self, removed_level: str) -> None:
        for level_id, ids in list(self._by_level.items()):
            shifted = [i - 1 for i in ids if i != 0]
            if shifted:
                self._by_level[level_id] = shifted
            else:
                del self._by_level[level_id]
        if removed_level not in self._by_level:
            return

    def sample_any(self) -> ReplayItem:
        if not self._items:
            raise RuntimeError("Replay buffer is empty")
        return self.rng.choice(self._items)

    def sample_for_level(self, level_id: str) -> ReplayItem:
        ids = self._by_level.get(level_id, [])
        if not ids:
            return self.sample_any()
        idx = self.rng.choice(ids)
        return self._items[idx]

    def sample_state_for_level(self, level_id: str) -> SokobanState:
        return self.sample_for_level(level_id).state
