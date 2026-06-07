from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class SharedBackbone(nn.Module):
    def __init__(self, in_channels: int, filters: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


@dataclass
class ModelConfig:
    channels: int = 9
    filters: int = 64
    blocks: int = 4
    d: int = 4
    action_dim: int = 5


class ActionModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = SharedBackbone(cfg.channels, cfg.filters, cfg.blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.filters, cfg.d * cfg.action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        logits = self.head(h)
        return logits.view(-1, self.cfg.d, self.cfg.action_dim)


class StateModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = SharedBackbone(cfg.channels, cfg.filters, cfg.blocks)
        self.head = nn.Conv2d(cfg.filters, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)
