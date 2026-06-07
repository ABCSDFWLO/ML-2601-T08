from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

Coord = Tuple[int, int]

ACTION_TO_DELTA = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (0, -1),  # left
    3: (1, 0),   # down
}


@dataclass(frozen=True)
class SokobanState:
    player: Coord
    boxes: frozenset[Coord]


@dataclass
class SokobanLevel:
    height: int
    width: int
    walls: Set[Coord]
    goals: Set[Coord]
    initial_state: SokobanState
    level_id: str = ""

    @staticmethod
    def from_ascii(rows: Sequence[str], level_id: str = "") -> "SokobanLevel":
        h = len(rows)
        w = max(len(r) for r in rows)
        walls: Set[Coord] = set()
        goals: Set[Coord] = set()
        boxes: Set[Coord] = set()
        player: Coord | None = None

        for r in range(h):
            row = rows[r].ljust(w)
            for c, ch in enumerate(row):
                if ch == "#":
                    walls.add((r, c))
                elif ch == ".":
                    goals.add((r, c))
                elif ch == "$":
                    boxes.add((r, c))
                elif ch == "@":
                    player = (r, c)
                elif ch == "*":
                    goals.add((r, c))
                    boxes.add((r, c))
                elif ch == "+":
                    goals.add((r, c))
                    player = (r, c)

        if player is None:
            raise ValueError("Level has no player")

        return SokobanLevel(
            height=h,
            width=w,
            walls=walls,
            goals=goals,
            initial_state=SokobanState(player=player, boxes=frozenset(boxes)),
            level_id=level_id,
        )

    def inside(self, pos: Coord) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def free_cells(self) -> List[Coord]:
        cells: List[Coord] = []
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) not in self.walls:
                    cells.append((r, c))
        return cells

    def apply_action(self, state: SokobanState, action: int) -> SokobanState:
        if action == 4:
            return state
        if action not in ACTION_TO_DELTA:
            return state

        dr, dc = ACTION_TO_DELTA[action]
        pr, pc = state.player
        npos = (pr + dr, pc + dc)
        if not self.inside(npos) or npos in self.walls:
            return state

        boxes = set(state.boxes)
        if npos in boxes:
            push_to = (npos[0] + dr, npos[1] + dc)
            if (not self.inside(push_to)) or (push_to in self.walls) or (push_to in boxes):
                return state
            boxes.remove(npos)
            boxes.add(push_to)
            return SokobanState(player=npos, boxes=frozenset(boxes))
        return SokobanState(player=npos, boxes=state.boxes)

    def rollout(self, state: SokobanState, actions: Sequence[int]) -> SokobanState:
        cur = state
        for action in actions:
            cur = self.apply_action(cur, int(action))
        return cur

    def solved(self, state: SokobanState) -> bool:
        return all(b in self.goals for b in state.boxes)

    def encode_state(self, state: SokobanState) -> np.ndarray:
        # Channels: player, wall, box, goal
        x = np.zeros((4, self.height, self.width), dtype=np.float32)
        for (r, c) in self.walls:
            x[1, r, c] = 1.0
        for (r, c) in self.goals:
            x[3, r, c] = 1.0
        pr, pc = state.player
        x[0, pr, pc] = 1.0
        for (r, c) in state.boxes:
            x[2, r, c] = 1.0
        return x

    def decode_state_tensor(self, tensor: np.ndarray, template: SokobanState) -> SokobanState:
        # Decode player as argmax on free cells, and boxes as top-k scores.
        n_boxes = len(template.boxes)
        free = self.free_cells()

        p_scores = np.array([tensor[0, r, c] for (r, c) in free], dtype=np.float32)
        p_idx = int(np.argmax(p_scores))
        player = free[p_idx]

        b_scores = np.array([tensor[2, r, c] for (r, c) in free], dtype=np.float32)
        order = np.argsort(-b_scores)
        picked: List[Coord] = []
        for idx in order:
            cell = free[int(idx)]
            if cell == player:
                continue
            picked.append(cell)
            if len(picked) >= n_boxes:
                break

        if len(picked) < n_boxes:
            for cell in free:
                if cell != player and cell not in picked:
                    picked.append(cell)
                    if len(picked) >= n_boxes:
                        break

        return SokobanState(player=player, boxes=frozenset(picked[:n_boxes]))

    def distance(self, state: SokobanState, target: SokobanState) -> float:
        # Dense proxy used for planning and training selection.
        player_dist = abs(state.player[0] - target.player[0]) + abs(state.player[1] - target.player[1])
        box_mismatch = len(state.boxes.symmetric_difference(target.boxes))
        goal_miss = sum(1 for b in state.boxes if b not in self.goals)
        return float(player_dist + 2.0 * box_mismatch + 3.0 * goal_miss)

    def random_walk(self, rng: random.Random, steps: int) -> SokobanState:
        s = self.initial_state
        for _ in range(steps):
            s = self.apply_action(s, rng.randint(0, 3))
        return s

    def state_with_boxes_on_goals(self, rng: random.Random, player_mode: str = "random") -> List[SokobanState]:
        boxes = frozenset(self.goals)
        free = [c for c in self.free_cells() if c not in boxes]
        if not free:
            return []
        if player_mode == "all":
            return [SokobanState(player=cell, boxes=boxes) for cell in free]
        return [SokobanState(player=rng.choice(free), boxes=boxes)]
