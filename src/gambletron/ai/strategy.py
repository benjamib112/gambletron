"""Strategy storage: serialize/deserialize blueprint strategies."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class CheckpointStrategy:
    """Strategy backed by a C++ binary checkpoint — no full extraction needed.

    Loads the .bin checkpoint into the C++ engine and queries infosets
    on demand, avoiding the memory cost of a full Python dict.
    """

    def __init__(self, path: str | Path) -> None:
        import gambletron_engine as engine

        self._config = engine.MCCFRConfig()
        self._config.num_players = 6
        self._trainer = engine.MCCFRTrainer(self._config)
        self._trainer.load_checkpoint(str(path))

    def get(self, key: int) -> Optional[List[float]]:
        avg = self._trainer.get_average_strategy(key)
        if not avg:
            return None
        return list(avg)

    def get_or_uniform(self, key: int, num_actions: int) -> List[float]:
        avg = self._trainer.get_average_strategy(key)
        if avg and len(avg) > 0:
            return list(avg)
        return [1.0 / num_actions] * num_actions

    def __len__(self) -> int:
        return self._trainer.num_infosets()

    @classmethod
    def from_file(cls, path: str | Path) -> "CheckpointStrategy":
        return cls(path)


class Strategy:
    """Stores action probability distributions keyed by infoset."""

    def __init__(self) -> None:
        # infoset_key -> list of action probabilities
        self._strategies: Dict[int, List[float]] = {}

    def set(self, key: int, probs: List[float]) -> None:
        self._strategies[key] = probs

    def get(self, key: int) -> Optional[List[float]]:
        return self._strategies.get(key)

    def get_or_uniform(self, key: int, num_actions: int) -> List[float]:
        probs = self._strategies.get(key)
        if probs is not None:
            return probs
        return [1.0 / num_actions] * num_actions

    def __len__(self) -> int:
        return len(self._strategies)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._strategies, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._strategies = pickle.load(f)

    @classmethod
    def from_file(cls, path: str | Path) -> Strategy:
        s = cls()
        s.load(path)
        return s

    def merge(self, other: Strategy, weight: float = 1.0) -> None:
        """Merge another strategy into this one (for snapshot averaging)."""
        for key, probs in other._strategies.items():
            if key in self._strategies:
                existing = self._strategies[key]
                if len(existing) == len(probs):
                    merged = [
                        (e + weight * p) for e, p in zip(existing, probs)
                    ]
                    total = sum(merged)
                    if total > 0:
                        self._strategies[key] = [m / total for m in merged]
            else:
                self._strategies[key] = list(probs)
