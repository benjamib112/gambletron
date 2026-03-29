"""Action abstraction: defines which bet sizes to consider.

Blueprint uses fine-grained preflop, coarser postflop.
Search uses 1-6 raise sizes. Includes pseudo-harmonic mapping for off-tree actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from gambletron.poker.state import BettingRound


@dataclass
class ActionAbstraction:
    """Defines the raise sizes available at each decision point."""

    # Raise sizes as fractions of the pot, per round per raise number
    # raise_fractions[round][raise_index] = list of pot fractions
    raise_fractions: dict  # BettingRound -> list of list of float

    @staticmethod
    def blueprint() -> ActionAbstraction:
        """Blueprint action abstraction matching Pluribus paper."""
        return ActionAbstraction(
            raise_fractions={
                BettingRound.PREFLOP: [
                    # First raise (open): many sizes
                    [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
                    # Second raise (3-bet): fewer sizes
                    [0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 8.0],
                    # Third raise (4-bet)
                    [0.5, 1.0, 2.0, 4.0],
                    # Further raises
                    [1.0, 2.0],
                ],
                BettingRound.FLOP: [
                    [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                    [0.5, 1.0, 2.0],
                    [1.0],
                ],
                BettingRound.TURN: [
                    [0.5, 1.0],  # First raise: 0.5x, 1x pot (+ all-in always)
                    [1.0],       # Subsequent: 1x pot (+ all-in)
                ],
                BettingRound.RIVER: [
                    [0.5, 1.0],
                    [1.0],
                ],
            }
        )

    @staticmethod
    def search() -> ActionAbstraction:
        """Search action abstraction (finer for current round)."""
        return ActionAbstraction(
            raise_fractions={
                BettingRound.PREFLOP: [
                    [0.5, 0.75, 1.0, 1.5, 2.5, 4.0],
                    [0.5, 1.0, 2.0, 4.0],
                    [1.0, 2.0],
                ],
                BettingRound.FLOP: [
                    [0.25, 0.5, 0.75, 1.0, 1.5],
                    [0.5, 1.0, 2.0],
                    [1.0],
                ],
                BettingRound.TURN: [
                    [0.5, 0.75, 1.0],
                    [1.0],
                ],
                BettingRound.RIVER: [
                    [0.5, 0.75, 1.0],
                    [1.0],
                ],
            }
        )

    def get_raise_sizes(
        self,
        betting_round: BettingRound,
        raise_num: int,
        pot: int,
        current_bet: int,
        player_stack: int,
        player_bet: int,
        min_raise: int,
    ) -> List[int]:
        """Get concrete raise-to amounts for the given situation.

        Always includes all-in as an option. Fold and call are always
        available separately (not returned here).
        """
        round_fracs = self.raise_fractions.get(betting_round, [[1.0]])
        idx = min(raise_num, len(round_fracs) - 1)
        fractions = round_fracs[idx]

        max_raise_to = player_bet + player_stack
        min_raise_to = current_bet + min_raise

        sizes = set()
        for frac in fractions:
            amount = int(current_bet + frac * pot)
            # Clamp to legal range
            amount = max(amount, min_raise_to)
            amount = min(amount, max_raise_to)
            if amount > current_bet and amount <= max_raise_to:
                sizes.add(amount)

        # Always include all-in
        if max_raise_to > current_bet:
            sizes.add(max_raise_to)

        return sorted(sizes)


def pseudo_harmonic_mapping(
    actual_bet: int,
    lower_abstract: int,
    upper_abstract: int,
) -> float:
    """Pseudo-harmonic action translation for off-tree bet sizes.

    Returns the probability of mapping to the lower abstract action.
    (1 - result) is the probability of mapping to the upper abstract action.

    Based on Ganzfried & Sandholm (IJCAI 2013).
    """
    if lower_abstract == upper_abstract:
        return 1.0
    if actual_bet <= lower_abstract:
        return 1.0
    if actual_bet >= upper_abstract:
        return 0.0

    # Pseudo-harmonic mapping: weight inversely proportional to distance
    # p(lower) = (upper - actual) / (upper - lower) adjusted harmonically
    dist_low = actual_bet - lower_abstract
    dist_high = upper_abstract - actual_bet

    # Harmonic: use reciprocals
    if dist_low == 0:
        return 1.0
    if dist_high == 0:
        return 0.0

    w_low = 1.0 / dist_low
    w_high = 1.0 / dist_high
    return w_low / (w_low + w_high)
