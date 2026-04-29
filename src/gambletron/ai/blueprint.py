"""Blueprint strategy training via MCCFR.

Orchestrates the C++ MCCFR engine with proper abstraction configuration.
"""

from __future__ import annotations

import hashlib
import struct
import time
from pathlib import Path
from typing import Optional

from gambletron.ai.abstraction import canonical_preflop
from gambletron.ai.strategy import Strategy, TrainerStrategy
from gambletron.poker.card import Card

# Try C++ engine, fall back to pure Python
try:
    import gambletron_engine as engine

    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False


def make_infoset_key(
    player: int,
    betting_round: int,
    hole_cards: tuple,
    board: tuple,
    board_len: int,
    action_seq: tuple,
    action_seq_len: int,
) -> int:
    """Compute an infoset key from game state.

    Combines the player's abstract card bucket with the action sequence.
    """
    # Card abstraction: preflop uses canonical 169 buckets
    if betting_round == 0:
        card_bucket = canonical_preflop(Card(hole_cards[0]), Card(hole_cards[1]))
    else:
        # Simplified postflop: hash hole cards + board
        # (In production, this would use the trained k-means buckets)
        cards_data = struct.pack(
            f"<{2 + board_len}i",
            *sorted(hole_cards[:2]),
            *sorted(board[:board_len]),
        )
        card_bucket = int(hashlib.md5(cards_data).hexdigest(), 16) % 50

    # Hash action sequence
    if action_seq_len > 0:
        action_data = struct.pack(f"<{action_seq_len}i", *action_seq[:action_seq_len])
        action_hash = int(hashlib.md5(action_data).hexdigest(), 16) % (1 << 32)
    else:
        action_hash = 0

    # Combine: player, round, card bucket, action hash
    key = (
        (player << 56)
        | (betting_round << 48)
        | (card_bucket << 32)
        | action_hash
    )
    return key & 0xFFFFFFFFFFFFFFFF


def _cpp_infoset_key(
    player: int,
    betting_round: int,
    hole_cards: list,
    board: list,
    board_len: int,
    action_seq: list,
    action_seq_len: int,
) -> int:
    """C++ compatible infoset key function. Receives lists from C++."""
    return make_infoset_key(
        player,
        betting_round,
        tuple(hole_cards),
        tuple(board),
        board_len,
        tuple(action_seq),
        action_seq_len,
    )


class BlueprintTrainer:
    """Trains the blueprint strategy using C++ MCCFR engine."""

    def __init__(
        self,
        num_players: int = 6,
        num_threads: int = 1,
        discount_interval: int = 100,
        lcfr_threshold: int = 4000,
        prune_threshold: int = 2000,
        strategy_interval: int = 10000,
    ) -> None:
        if not HAS_CPP_ENGINE:
            raise RuntimeError(
                "C++ engine not available. Build it first with CMake."
            )

        self.config = engine.MCCFRConfig()
        self.config.num_players = num_players
        self.config.num_threads = num_threads
        self.config.discount_interval = discount_interval
        self.config.lcfr_threshold = lcfr_threshold
        self.config.prune_threshold = prune_threshold
        self.config.strategy_interval = strategy_interval

        if num_threads > 1:
            # Use built-in C++ key function for multi-threaded training
            self.trainer = engine.MCCFRTrainer(self.config)
        else:
            self.trainer = engine.MCCFRTrainer(self.config, _cpp_infoset_key)
        self._snapshots: list[dict] = []

    def save_checkpoint(self, path: str) -> None:
        """Save full training state to a binary checkpoint file."""
        self.trainer.save_checkpoint(str(path))

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a binary checkpoint file."""
        self.trainer.load_checkpoint(str(path))

    def train(
        self,
        num_iterations: int,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 1000,
        snapshot_points: Optional[dict[str, int]] = None,
        verbose: bool = True,
    ) -> TrainerStrategy:
        """Run blueprint training.

        Args:
            num_iterations: total MCCFR iterations to run
            checkpoint_dir: directory to save periodic checkpoints
            checkpoint_interval: iterations between checkpoints
            snapshot_points: dict of {name: iteration} for named snapshots
            verbose: print progress

        Returns:
            A TrainerStrategy that queries the C++ store on demand (no bulk copy).
        """
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        saved_snapshots: set[str] = set()
        start_iter = self.trainer.iterations_done()
        target_total = start_iter + num_iterations

        # Mark snapshots that are already past
        if snapshot_points:
            for name, target_iter in snapshot_points.items():
                if start_iter >= target_iter:
                    saved_snapshots.add(name)

        start_time = time.time()
        remaining = num_iterations

        while remaining > 0:
            batch = min(remaining, checkpoint_interval)
            self.trainer.train(batch)
            remaining -= batch

            done = self.trainer.iterations_done()
            new_iters = done - start_iter
            elapsed = time.time() - start_time
            iters_per_sec = new_iters / elapsed if elapsed > 0 else 0

            if verbose:
                threads_str = f", {self.config.num_threads} threads" if self.config.num_threads > 1 else ""
                print(
                    f"  Iteration {done}/{target_total} "
                    f"({iters_per_sec:.1f} it/s, "
                    f"{self.trainer.num_infosets()} infosets, "
                    f"{elapsed:.1f}s elapsed{threads_str})"
                )

            # Save binary checkpoint (for resuming)
            if checkpoint_dir:
                self.trainer.save_checkpoint(
                    f"{checkpoint_dir}/checkpoint.bin"
                )

            # Save named snapshots as binary checkpoints
            if snapshot_points and checkpoint_dir:
                for name, target_iter in snapshot_points.items():
                    if name not in saved_snapshots and done >= target_iter:
                        snapshot_path = f"{checkpoint_dir}/{name}.bin"
                        self.trainer.save_checkpoint(snapshot_path)
                        saved_snapshots.add(name)
                        if verbose:
                            print(f"  >> Saved '{name}' snapshot at iteration {done}")

        return TrainerStrategy(self.trainer)

    def _extract_strategy(self) -> Strategy:
        """Extract current strategy from the trainer."""
        strategy = Strategy()
        raw = self.trainer.get_all_strategies()
        for key, probs in raw.items():
            strategy.set(int(key), list(probs))
        return strategy


class PurePythonMCCFR:
    """Pure Python MCCFR for small games (testing/Kuhn poker)."""

    def __init__(self, num_players: int = 2) -> None:
        self.num_players = num_players
        self.regrets: dict[int, list[float]] = {}
        self.strategy_sum: dict[int, list[float]] = {}
        self.iterations = 0

    def get_strategy(self, key: int, num_actions: int) -> list[float]:
        regrets = self.regrets.get(key)
        if regrets is None:
            return [1.0 / num_actions] * num_actions

        positive = [max(r, 0) for r in regrets]
        total = sum(positive)
        if total > 0:
            return [p / total for p in positive]
        return [1.0 / num_actions] * num_actions

    def update_regrets(
        self, key: int, action_values: list[float], strategy: list[float]
    ) -> None:
        node_value = sum(s * v for s, v in zip(strategy, action_values))
        if key not in self.regrets:
            self.regrets[key] = [0.0] * len(action_values)
        for a in range(len(action_values)):
            self.regrets[key][a] += action_values[a] - node_value

    def update_strategy_sum(self, key: int, strategy: list[float]) -> None:
        if key not in self.strategy_sum:
            self.strategy_sum[key] = [0.0] * len(strategy)
        for a in range(len(strategy)):
            self.strategy_sum[key][a] += strategy[a]

    def get_average_strategy(self, key: int, num_actions: int) -> list[float]:
        sums = self.strategy_sum.get(key)
        if sums is None:
            return [1.0 / num_actions] * num_actions
        total = sum(sums)
        if total > 0:
            return [s / total for s in sums]
        return [1.0 / num_actions] * num_actions

    def extract_strategy(self) -> Strategy:
        strategy = Strategy()
        for key, sums in self.strategy_sum.items():
            total = sum(sums)
            if total > 0:
                probs = [s / total for s in sums]
            else:
                probs = [1.0 / len(sums)] * len(sums)
            strategy.set(key, probs)
        return strategy
