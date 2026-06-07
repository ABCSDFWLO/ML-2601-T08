from __future__ import annotations

from dataclasses import dataclass
import itertools
import random
from typing import List, Sequence

import numpy as np
import torch

from .models import ActionModel, StateModel
from .replay import ReplayBuffer
from .sokoban import SokobanLevel, SokobanState


@dataclass
class PlannerConfig:
    d: int = 4
    R: int = 5
    ndss: int = 32
    device: str = "cpu"


class HalfWegPlanner:
    def __init__(
        self,
        action_model: ActionModel,
        state_model: StateModel,
        cfg: PlannerConfig,
        replay: ReplayBuffer,
        seed: int = 0,
    ) -> None:
        self.action_model = action_model
        self.state_model = state_model
        self.cfg = cfg
        self.replay = replay
        self.rng = random.Random(seed)
        self.device = torch.device(cfg.device)

        self.action_model.to(self.device)
        self.state_model.to(self.device)

    def _input_tensor(self, level: SokobanLevel, u: SokobanState, v: SokobanState, b: int, r: int | None = None) -> torch.Tensor:
        eu = level.encode_state(u)
        ev = level.encode_state(v)
        b_plane = np.full((1, level.height, level.width), float(b), dtype=np.float32)
        if r is None:
            r_norm = 0.0
        else:
            r_norm = float(r) / float(max(1, self.cfg.R))
        r_plane = np.full((1, level.height, level.width), r_norm, dtype=np.float32)
        x = np.concatenate([eu, ev, b_plane, r_plane], axis=0)
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _predict_actions(self, level: SokobanLevel, u: SokobanState, v: SokobanState, b: int) -> List[int]:
        x = self._input_tensor(level, u, v, b, r=None)
        logits = self.action_model(x)[0]
        actions = torch.argmax(logits, dim=-1).tolist()
        out: List[int] = []
        for a in actions:
            out.append(int(a))
            if a == 4:
                break
        return out

    @torch.no_grad()
    def _predict_subgoal(self, level: SokobanLevel, u: SokobanState, v: SokobanState, b: int, r: int) -> SokobanState:
        x = self._input_tensor(level, u, v, b, r=r)
        y = self.state_model(x)[0].detach().cpu().numpy()
        return level.decode_state_tensor(y, template=u)

    def _score(self, level: SokobanLevel, end_state: SokobanState, target: SokobanState, b: int) -> float:
        d = level.distance(end_state, target)
        return -d if b == 0 else d

    def _exhaustive_pl0(self, level: SokobanLevel, u: SokobanState, v: SokobanState, b: int) -> List[int]:
        best_score = float("-inf")
        best_plan: List[int] = [4]
        for seq in itertools.product([0, 1, 2, 3, 4], repeat=self.cfg.d):
            plan = list(seq)
            if 4 in plan:
                stop_idx = plan.index(4)
                plan = plan[: stop_idx + 1]
            end_state = level.rollout(u, plan)
            score = self._score(level, end_state, v, b)
            if score > best_score:
                best_score = score
                best_plan = plan
        return best_plan

    def plan_policy(self, policy_level: int, level: SokobanLevel, u: SokobanState, v: SokobanState, b: int) -> List[int]:
        if policy_level <= 0:
            return self._predict_actions(level, u, v, b)
        w = self._predict_subgoal(level, u, v, b, r=policy_level)
        a1 = self.plan_policy(policy_level - 1, level, u, w, 0)
        w_hat = level.rollout(u, a1)
        a2 = self.plan_policy(policy_level - 1, level, w_hat, v, b)
        return (a1 + a2)[: (2 ** policy_level) * self.cfg.d]

    def search_policy(self, policy_level: int, level: SokobanLevel, u: SokobanState, v: SokobanState, b: int) -> List[int]:
        if policy_level == 0:
            return self._exhaustive_pl0(level, u, v, b)

        best_score = float("-inf")
        best_plan: List[int] = [4]

        for _ in range(self.cfg.ndss):
            w = self.replay.sample_state_for_level(level.level_id)
            a1 = self.plan_policy(policy_level - 1, level, u, w, 0)
            u_hat = level.rollout(u, a1)
            a2 = self.plan_policy(policy_level - 1, level, u_hat, v, b)
            plan = a1 + a2
            end_state = level.rollout(u, plan)
            score = self._score(level, end_state, v, b)
            if score > best_score:
                best_score = score
                best_plan = plan

        max_len = (2 ** policy_level) * self.cfg.d
        return best_plan[:max_len]

    def best_plan_over_hierarchy(self, level: SokobanLevel, u: SokobanState, v: SokobanState, b: int) -> List[int]:
        best_score = float("-inf")
        best: List[int] = [4]
        for i in range(0, self.cfg.R + 1):
            plan = self.search_policy(i, level, u, v, b)
            end_state = level.rollout(u, plan)
            score = self._score(level, end_state, v, b)
            if score > best_score:
                best_score = score
                best = plan
        return best
