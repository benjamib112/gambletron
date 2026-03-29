"""Information abstraction: maps poker hands to abstract buckets.

Preflop: lossless abstraction (169 canonical hands via suit isomorphism).
Postflop: lossy abstraction via k-means on hand-strength features.
"""

from __future__ import annotations

import os
import pickle
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from gambletron.poker.card import Card, Rank, Suit
from gambletron.poker.hand import evaluate_hand

# Number of buckets per round
PREFLOP_BUCKETS = 169  # Lossless
BLUEPRINT_POSTFLOP_BUCKETS = 200
SEARCH_POSTFLOP_BUCKETS = 500

# Number of Monte Carlo rollouts for hand strength estimation
DEFAULT_NUM_ROLLOUTS = 1000


def canonical_preflop(card1: Card, card2: Card) -> int:
    """Map a preflop hand to one of 169 canonical buckets (lossless).

    Canonical form: (high_rank, low_rank, suited) where suited hands and
    offsuit hands are distinct. Returns bucket index 0-168.

    Ordering: pairs first (13), then suited (78), then offsuit (78) = 169.
    """
    r1, r2 = card1.rank, card2.rank
    suited = card1.suit == card2.suit

    high = max(r1, r2)
    low = min(r1, r2)

    if high == low:
        # Pair: 0-12 (22 through AA)
        return int(high)
    elif suited:
        # Suited non-pair: use triangular index
        # 13 + index in upper triangle
        return 13 + _triangle_index(high, low)
    else:
        # Offsuit non-pair
        return 13 + 78 + _triangle_index(high, low)


def _triangle_index(high: int, low: int) -> int:
    """Index into upper triangle of 13x13 matrix (excluding diagonal)."""
    # high > low guaranteed
    # Count elements before row 'high': sum(i for i in range(high)) = high*(high-1)/2
    return high * (high - 1) // 2 + low


class HandStrengthCalculator:
    """Estimates hand strength via Monte Carlo rollouts."""

    def __init__(self, num_rollouts: int = DEFAULT_NUM_ROLLOUTS, seed: int = 42) -> None:
        self.num_rollouts = num_rollouts
        self.rng = np.random.RandomState(seed)

    def hand_strength(
        self,
        hole_cards: List[int],
        board_cards: List[int],
        num_opponents: int = 1,
    ) -> float:
        """Expected hand strength (win probability) via Monte Carlo."""
        known = set(hole_cards) | set(board_cards)
        remaining_deck = [c for c in range(52) if c not in known]
        cards_needed = 5 - len(board_cards) + 2 * num_opponents

        wins = 0
        ties = 0
        total = 0

        for _ in range(self.num_rollouts):
            sampled = self.rng.choice(remaining_deck, size=cards_needed, replace=False)
            idx = 0

            # Complete the board
            full_board = list(board_cards) + list(sampled[idx:idx + 5 - len(board_cards)])
            idx += 5 - len(board_cards)

            my_cards = [Card(c) for c in hole_cards] + [Card(c) for c in full_board]
            my_score = evaluate_hand(my_cards)

            won = True
            tied = False
            for _ in range(num_opponents):
                opp_hole = [int(sampled[idx]), int(sampled[idx + 1])]
                idx += 2
                opp_cards = [Card(c) for c in opp_hole] + [Card(c) for c in full_board]
                opp_score = evaluate_hand(opp_cards)
                if opp_score > my_score:
                    won = False
                    tied = False
                    break
                elif opp_score == my_score:
                    tied = True

            if won and not tied:
                wins += 1
            elif tied:
                ties += 1
            total += 1

        return (wins + 0.5 * ties) / total if total > 0 else 0.5

    def hand_potential(
        self,
        hole_cards: List[int],
        board_cards: List[int],
    ) -> Tuple[float, float]:
        """Positive and negative hand potential.

        Returns (ppot, npot):
        - ppot: probability of improving from behind/tied to ahead
        - npot: probability of falling from ahead/tied to behind
        """
        if len(board_cards) >= 5:
            return 0.0, 0.0

        known = set(hole_cards) | set(board_cards)
        remaining = [c for c in range(52) if c not in known]

        # We need: 2 opponent cards + remaining board cards for current + final
        ahead_behind_ahead = 0
        ahead_behind_total = 0
        behind_ahead_ahead = 0
        behind_ahead_total = 0

        for _ in range(self.num_rollouts):
            sampled = self.rng.choice(
                remaining, size=2 + (5 - len(board_cards)), replace=False
            )
            opp_hole = [int(sampled[0]), int(sampled[1])]
            future_board = list(board_cards) + [int(sampled[2 + i]) for i in range(5 - len(board_cards))]
            current_board = list(board_cards)

            # Current evaluation
            my_current = evaluate_hand([Card(c) for c in hole_cards + current_board])
            opp_current = evaluate_hand([Card(c) for c in opp_hole + current_board])

            # Final evaluation
            my_final = evaluate_hand([Card(c) for c in hole_cards + future_board])
            opp_final = evaluate_hand([Card(c) for c in opp_hole + future_board])

            if my_current < opp_current:
                # Currently behind
                behind_ahead_total += 1
                if my_final > opp_final:
                    behind_ahead_ahead += 1
            elif my_current > opp_current:
                # Currently ahead
                ahead_behind_total += 1
                if my_final < opp_final:
                    ahead_behind_ahead += 1

        ppot = behind_ahead_ahead / max(behind_ahead_total, 1)
        npot = ahead_behind_ahead / max(ahead_behind_total, 1)
        return ppot, npot

    def compute_features(
        self,
        hole_cards: List[int],
        board_cards: List[int],
    ) -> np.ndarray:
        """Compute feature vector for clustering: [hand_strength, ppot, npot]."""
        hs = self.hand_strength(hole_cards, board_cards)
        ppot, npot = self.hand_potential(hole_cards, board_cards)
        return np.array([hs, ppot, npot], dtype=np.float32)


class PostflopAbstraction:
    """Lossy postflop abstraction using k-means clustering on hand features."""

    def __init__(self, num_buckets: int = BLUEPRINT_POSTFLOP_BUCKETS) -> None:
        self.num_buckets = num_buckets
        self.centroids: Optional[np.ndarray] = None
        self._trained = False

    def train(
        self,
        features: np.ndarray,
        max_iter: int = 100,
    ) -> None:
        """Train k-means clustering on precomputed feature vectors.

        Args:
            features: (N, 3) array of [hand_strength, ppot, npot] features
            max_iter: maximum k-means iterations
        """
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=self.num_buckets,
            max_iter=max_iter,
            n_init=3,
            random_state=42,
        )
        kmeans.fit(features)
        self.centroids = kmeans.cluster_centers_
        self._trained = True

    def get_bucket(self, features: np.ndarray) -> int:
        """Assign a feature vector to its nearest bucket."""
        if not self._trained:
            raise RuntimeError("Abstraction not trained yet")
        dists = np.linalg.norm(self.centroids - features, axis=1)
        return int(np.argmin(dists))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"centroids": self.centroids, "num_buckets": self.num_buckets}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.centroids = data["centroids"]
        self.num_buckets = data["num_buckets"]
        self._trained = True
