"""Belief tracking: maintain probability distributions over opponent hands.

Each player's range is a distribution over 1326 possible hole card pairs,
updated via Bayes' rule after each observed action.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

# Total number of 2-card combinations from 52 cards
NUM_HANDS = 1326

# Precompute all 1326 hand indices: (card1, card2) where card1 < card2
ALL_HANDS: List[Tuple[int, int]] = []
HAND_INDEX: Dict[Tuple[int, int], int] = {}

for _idx, (_c1, _c2) in enumerate(combinations(range(52), 2)):
    ALL_HANDS.append((_c1, _c2))
    HAND_INDEX[(_c1, _c2)] = _idx


def hand_to_index(card1: int, card2: int) -> int:
    """Convert two card integers to a hand index (0-1325)."""
    lo, hi = min(card1, card2), max(card1, card2)
    return HAND_INDEX[(lo, hi)]


def index_to_hand(idx: int) -> Tuple[int, int]:
    """Convert a hand index back to two card integers."""
    return ALL_HANDS[idx]


class BeliefState:
    """Tracks belief distributions over opponent hands for all players."""

    def __init__(self, num_players: int) -> None:
        self.num_players = num_players
        # beliefs[player] = array of shape (1326,) with probabilities
        self.beliefs: List[np.ndarray] = [
            np.ones(NUM_HANDS, dtype=np.float64) / NUM_HANDS
            for _ in range(num_players)
        ]

    def reset(self) -> None:
        for i in range(self.num_players):
            self.beliefs[i] = np.ones(NUM_HANDS, dtype=np.float64) / NUM_HANDS

    def remove_known_cards(self, known_cards: List[int]) -> None:
        """Zero out probabilities for hands containing known cards."""
        known_set = set(known_cards)
        for player_idx in range(self.num_players):
            for hand_idx in range(NUM_HANDS):
                c1, c2 = ALL_HANDS[hand_idx]
                if c1 in known_set or c2 in known_set:
                    self.beliefs[player_idx][hand_idx] = 0.0
            self._normalize(player_idx)

    def update_on_action(
        self,
        player: int,
        action_probs_by_hand: np.ndarray,
    ) -> None:
        """Update beliefs for a player after observing their action.

        Args:
            player: the player who acted
            action_probs_by_hand: array of shape (1326,) where entry i is
                the probability that player would take the observed action
                if holding hand i (according to the current strategy).
        """
        # Bayes' rule: P(hand | action) ∝ P(action | hand) * P(hand)
        self.beliefs[player] *= action_probs_by_hand
        self._normalize(player)

    def get_reach_probs(self, player: int) -> np.ndarray:
        """Get the current belief distribution for a player."""
        return self.beliefs[player].copy()

    def get_nonzero_hands(self, player: int) -> List[int]:
        """Get indices of hands with nonzero probability."""
        return list(np.nonzero(self.beliefs[player] > 1e-10)[0])

    def _normalize(self, player: int) -> None:
        total = self.beliefs[player].sum()
        if total > 0:
            self.beliefs[player] /= total
        else:
            # All hands eliminated (shouldn't happen in practice)
            self.beliefs[player] = np.ones(NUM_HANDS, dtype=np.float64) / NUM_HANDS
