"""Table manager: runs multiple hands, tracks stacks, rotates dealer."""

from __future__ import annotations

from typing import List, Optional

from gambletron.players.base import Player
from gambletron.poker.card import Deck
from gambletron.poker.game import Game


class Table:
    """Manages a multi-hand poker session with 1-6 players."""

    def __init__(
        self,
        players: List[Player],
        starting_stack: int = 10000,
        small_blind: int = 50,
        big_blind: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        if not 2 <= len(players) <= 6:
            raise ValueError(f"Need 2-6 players, got {len(players)}")

        self.players = players
        self.stacks = [starting_stack] * len(players)
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_pos = 0
        self.deck = Deck(seed)
        self.hand_count = 0
        self.hand_results: List[List[int]] = []

    def play_hand(self) -> List[int]:
        """Play a single hand and return chip changes."""
        # Check if enough players have chips to play
        active_count = sum(1 for s in self.stacks if s > 0)
        if active_count < 2:
            return [0] * len(self.players)

        # Auto-fold busted players by marking them with 0 stacks
        # (Game will fold them since they can't post blinds or act)
        game = Game(
            players=self.players,
            stacks=list(self.stacks),
            dealer_pos=self.dealer_pos,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            deck=self.deck,
        )
        changes = game.play_hand()

        for i, change in enumerate(changes):
            self.stacks[i] += change

        self.hand_results.append(changes)
        self.hand_count += 1
        self._rotate_dealer()

        return changes

    def play_hands(self, n: int) -> List[List[int]]:
        """Play n hands. Returns list of chip changes per hand."""
        results = []
        for _ in range(n):
            results.append(self.play_hand())
        return results

    def _rotate_dealer(self) -> None:
        self.dealer_pos = (self.dealer_pos + 1) % len(self.players)

    @property
    def total_results(self) -> List[int]:
        """Cumulative chip changes for each player across all hands."""
        totals = [0] * len(self.players)
        for result in self.hand_results:
            for i, change in enumerate(result):
                totals[i] += change
        return totals
