from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from .boxoban import BoxobanCorpus
from .models import ActionModel, ModelConfig, StateModel
from .planner import HalfWegPlanner, PlannerConfig
from .replay import ReplayBuffer, ReplayItem
from .sokoban import SokobanLevel, SokobanState


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cpu"
    d: int = 4
    R: int = 5
    ndss: int = 32
    filters: int = 64
    blocks: int = 4
    lr: float = 1e-3
    replay_capacity: int = 50000
    max_files: int = 4
    max_levels_per_file: int = 128
    self_play_levels: int = 64
    self_play_steps: int = 64
    planning_samples: int = 128
    epochs: int = 3
    grad_clip: float = 1.0


@dataclass
class PlanExample:
    level: SokobanLevel
    u: SokobanState
    v: SokobanState
    b: int
    plan: List[int]


class HalfWegTrainer:
    def __init__(self, root_path: str, split_glob: str, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.corpus = BoxobanCorpus(root=root_path, split_glob=split_glob, seed=cfg.seed)
        self.corpus.load(max_files=cfg.max_files, max_levels_per_file=cfg.max_levels_per_file)

        model_cfg = ModelConfig(channels=10, filters=cfg.filters, blocks=cfg.blocks, d=cfg.d, action_dim=5)
        self.ma = ActionModel(model_cfg)
        self.ms = StateModel(model_cfg)

        self.replay = ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)
        self.planner = HalfWegPlanner(
            action_model=self.ma,
            state_model=self.ms,
            cfg=PlannerConfig(d=cfg.d, R=cfg.R, ndss=cfg.ndss, device=cfg.device),
            replay=self.replay,
            seed=cfg.seed,
        )

        self.device = torch.device(cfg.device)
        self.ma.to(self.device)
        self.ms.to(self.device)

        self.opt = AdamW(list(self.ma.parameters()) + list(self.ms.parameters()), lr=cfg.lr)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def _seed_replay(self) -> None:
        if len(self.replay) > 0:
            return
        for level in self.corpus.iter_sokoban_levels():
            self.replay.add(ReplayItem(level_id=level.level_id, state=level.initial_state))
            for steps in range(1, self.cfg.self_play_steps + 1, max(1, self.cfg.self_play_steps // 8)):
                state = level.random_walk(self.rng, steps)
                self.replay.add(ReplayItem(level_id=level.level_id, state=state))

    def collect_self_play(self) -> None:
        self._seed_replay()
        levels = list(self.corpus.iter_sokoban_levels())
        for _ in range(self.cfg.self_play_levels):
            level = self.rng.choice(levels)
            cur = level.initial_state
            for _ in range(self.cfg.self_play_steps):
                if self.rng.random() < 0.5:
                    act = self.rng.randint(0, 3)
                    cur = level.apply_action(cur, act)
                else:
                    goal = self.replay.sample_state_for_level(level.level_id)
                    p_level = self.rng.randint(0, self.cfg.R)
                    plan = self.planner.plan_policy(p_level, level, cur, goal, 0)
                    cur = level.rollout(cur, plan)
                self.replay.add(ReplayItem(level_id=level.level_id, state=cur))

    def sample_planning_problems(self) -> List[Tuple[SokobanLevel, SokobanState, SokobanState, int]]:
        levels = list(self.corpus.iter_sokoban_levels())
        problems: List[Tuple[SokobanLevel, SokobanState, SokobanState, int]] = []
        for _ in range(self.cfg.planning_samples):
            level = self.rng.choice(levels)
            u = self.replay.sample_state_for_level(level.level_id)
            v = self.replay.sample_state_for_level(level.level_id)
            b = 0
            problems.append((level, u, v, b))
        return problems

    def build_training_examples(self) -> List[PlanExample]:
        rows: List[PlanExample] = []
        for level, u, v, b in self.sample_planning_problems():
            plan = self.planner.best_plan_over_hierarchy(level, u, v, b)
            rows.append(PlanExample(level=level, u=u, v=v, b=b, plan=plan))
        return rows

    def _planner_input(self, level: SokobanLevel, u: SokobanState, v: SokobanState, b: int, r: int | None) -> torch.Tensor:
        eu = level.encode_state(u)
        ev = level.encode_state(v)
        b_plane = np.full((1, level.height, level.width), float(b), dtype=np.float32)
        r_val = 0.0 if r is None else float(r) / float(max(1, self.cfg.R))
        r_plane = np.full((1, level.height, level.width), r_val, dtype=np.float32)
        x = np.concatenate([eu, ev, b_plane, r_plane], axis=0)
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    def _action_target(self, plan: Sequence[int]) -> torch.Tensor:
        target = [4] * self.cfg.d
        for i in range(min(len(plan), self.cfg.d)):
            target[i] = int(plan[i])
            if target[i] == 4:
                break
        return torch.tensor(target, dtype=torch.long, device=self.device)

    def _states_along(self, level: SokobanLevel, start: SokobanState, plan: Sequence[int]) -> List[SokobanState]:
        states: List[SokobanState] = []
        cur = start
        for a in plan:
            cur = level.apply_action(cur, int(a))
            states.append(cur)
        return states

    def train_epoch(self, examples: Sequence[PlanExample]) -> float:
        self.ma.train()
        self.ms.train()
        total_loss = 0.0

        for ex in examples:
            level, u, v, b, plan = ex.level, ex.u, ex.v, ex.b, ex.plan
            traj = self._states_along(level, u, plan)
            ud = traj[min(len(traj), self.cfg.d) - 1] if traj else u
            uend = traj[-1] if traj else u
            prefix_plan = list(plan[: self.cfg.d])

            self.opt.zero_grad()

            x_main = self._planner_input(level, u, v, b, r=None)
            logits_main = self.ma(x_main)[0]
            y_main = self._action_target(plan)
            loss_ma = self.ce(logits_main, y_main)

            x_ref = self._planner_input(level, u, ud, 0, r=None)
            logits_ref = self.ma(x_ref)[0]
            y_ref = self._action_target(prefix_plan)
            loss_ma_ref = self.ce(logits_ref, y_ref)

            loss_ms = torch.tensor(0.0, device=self.device)
            for r in range(1, self.cfg.R + 1):
                idx = min((2 ** (r - 1)) * self.cfg.d, len(traj)) - 1
                target_state = traj[idx] if idx >= 0 and traj else u
                y = torch.from_numpy(level.encode_state(target_state)).unsqueeze(0).to(self.device)

                x = self._planner_input(level, u, v, b, r=r)
                pred = self.ms(x)
                loss_ms = loss_ms + self.mse(pred, y)

                x_ref_s = self._planner_input(level, u, uend, 0, r=r)
                pred_ref = self.ms(x_ref_s)
                loss_ms = loss_ms + self.mse(pred_ref, y)

            loss = loss_ma + 0.5 * loss_ma_ref + 0.5 * loss_ms
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.ma.parameters()) + list(self.ms.parameters()), self.cfg.grad_clip)
            self.opt.step()

            total_loss += float(loss.item())

        if not examples:
            return 0.0
        return total_loss / float(len(examples))

    def evaluate_quick(self, levels: int = 20, player_mode: str = "random") -> float:
        self.ma.eval()
        self.ms.eval()
        lv = list(self.corpus.iter_sokoban_levels())
        solved = 0
        total = 0
        for _ in range(min(levels, len(lv))):
            level = self.rng.choice(lv)
            starts = [level.initial_state]
            targets = level.state_with_boxes_on_goals(self.rng, player_mode=player_mode)
            if not targets:
                continue
            ok = False
            for t in targets:
                for s in starts:
                    plan = self.planner.plan_policy(self.cfg.R, level, s, t, 0)
                    end = level.rollout(s, plan)
                    if level.solved(end):
                        ok = True
                        break
                if ok:
                    break
            total += 1
            solved += int(ok)
        if total == 0:
            return 0.0
        return solved / total

    def fit(self) -> None:
        for epoch in range(1, self.cfg.epochs + 1):
            self.collect_self_play()
            examples = self.build_training_examples()
            loss = self.train_epoch(examples)
            solve_rate = self.evaluate_quick(levels=30, player_mode="random")
            print(f"epoch={epoch} examples={len(examples)} loss={loss:.4f} quick_solved={solve_rate:.3f}")

    def save(self, path: str) -> None:
        payload = {
            "cfg": self.cfg.__dict__,
            "ma": self.ma.state_dict(),
            "ms": self.ms.state_dict(),
        }
        torch.save(payload, path)
