from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, List, Sequence

from .sokoban import SokobanLevel, SokobanState


@dataclass(frozen=True)
class LevelSpec:
    level_id: str
    rows: List[str]


def _split_level_blocks(text: str) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith(";"):
            if current:
                blocks.append(current)
                current = []
            continue
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def parse_level_file(path: Path) -> List[LevelSpec]:
    text = path.read_text(encoding="utf-8")
    blocks = _split_level_blocks(text)
    out: List[LevelSpec] = []
    for i, rows in enumerate(blocks):
        out.append(LevelSpec(level_id=f"{path.stem}:{i}", rows=rows))
    return out


class BoxobanCorpus:
    def __init__(self, root: Path, split_glob: str, seed: int = 0) -> None:
        self.root = Path(root)
        self.split_glob = split_glob
        self.rng = random.Random(seed)
        self._levels: List[LevelSpec] = []

    def load(self, max_files: int | None = None, max_levels_per_file: int | None = None) -> None:
        files = sorted(self.root.glob(self.split_glob))
        if max_files is not None:
            files = files[:max_files]
        levels: List[LevelSpec] = []
        for file_path in files:
            parsed = parse_level_file(file_path)
            if max_levels_per_file is not None:
                parsed = parsed[:max_levels_per_file]
            levels.extend(parsed)
        self._levels = levels

    @property
    def levels(self) -> Sequence[LevelSpec]:
        return self._levels

    def sample_level(self) -> LevelSpec:
        if not self._levels:
            raise RuntimeError("No levels loaded. Call load() first.")
        return self.rng.choice(self._levels)

    def to_sokoban_level(self, spec: LevelSpec) -> SokobanLevel:
        return SokobanLevel.from_ascii(spec.rows, level_id=spec.level_id)

    def sample_sokoban_level(self) -> SokobanLevel:
        return self.to_sokoban_level(self.sample_level())

    def iter_sokoban_levels(self) -> Iterable[SokobanLevel]:
        for spec in self._levels:
            yield self.to_sokoban_level(spec)
