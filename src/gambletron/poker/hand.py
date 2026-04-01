"""Poker hand evaluation. Pure Python implementation with optional C++ acceleration."""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

from gambletron.poker.card import Card, Rank


# Hand rank categories (higher is better)
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8

_HAND_NAMES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush",
]


def hand_rank_name(category: int) -> str:
    return _HAND_NAMES[category]


# ── Human-readable hand descriptions ─────────────────────────────────────────

_RANK_LONG = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "10", Rank.JACK: "Jack", Rank.QUEEN: "Queen",
    Rank.KING: "King", Rank.ACE: "Ace",
}
_RANK_PLURAL = {
    Rank.TWO: "2s", Rank.THREE: "3s", Rank.FOUR: "4s", Rank.FIVE: "5s",
    Rank.SIX: "6s", Rank.SEVEN: "7s", Rank.EIGHT: "8s", Rank.NINE: "9s",
    Rank.TEN: "10s", Rank.JACK: "Jacks", Rank.QUEEN: "Queens",
    Rank.KING: "Kings", Rank.ACE: "Aces",
}


def _rn(r: int) -> str:
    return _RANK_LONG.get(Rank(r), str(r))


def _rp(r: int) -> str:
    return _RANK_PLURAL.get(Rank(r), str(r) + "s")


def describe_hand(score: Tuple) -> str:
    """Convert an evaluate_hand() score tuple to a readable string.

    Examples: "Flush, Ace-high", "Full House, Kings full of Aces", "Pair of Jacks"
    """
    cat = score[0]
    if cat == STRAIGHT_FLUSH:
        return "Royal Flush" if score[1] == int(Rank.ACE) else f"Straight Flush, {_rn(score[1])}-high"
    if cat == FOUR_OF_A_KIND:
        return f"Four {_rp(score[1])}"
    if cat == FULL_HOUSE:
        return f"Full House, {_rp(score[1])} full of {_rp(score[2])}"
    if cat == FLUSH:
        return f"Flush, {_rn(score[1])}-high"
    if cat == STRAIGHT:
        return f"Straight, {_rn(score[1])}-high"
    if cat == THREE_OF_A_KIND:
        return f"Three {_rp(score[1])}"
    if cat == TWO_PAIR:
        return f"Two Pair, {_rp(score[1])} and {_rp(score[2])}"
    if cat == ONE_PAIR:
        return f"Pair of {_rp(score[1])}"
    return f"{_rn(score[1])}-high"


def evaluate_hand(cards: List[Card]) -> Tuple[int, ...]:
    """Evaluate the best 5-card hand from a list of cards (typically 7).

    Returns a tuple that can be compared directly: higher tuple = better hand.
    Format: (category, *tiebreakers)
    """
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")
    if len(cards) == 5:
        return _eval5(cards)
    best = None
    for combo in combinations(cards, 5):
        score = _eval5(list(combo))
        if best is None or score > best:
            best = score
    return best


def _eval5(cards: List[Card]) -> Tuple[int, ...]:
    """Evaluate exactly 5 cards."""
    ranks = sorted((c.rank for c in cards), reverse=True)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1

    # Check for straight
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    high_card = None

    if len(unique_ranks) >= 5:
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i + 4] == 4:
                is_straight = True
                high_card = unique_ranks[i]
                break
        # Ace-low straight (A-2-3-4-5)
        if not is_straight and Rank.ACE in unique_ranks:
            low = sorted(unique_ranks)
            if low[:4] == [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE]:
                is_straight = True
                high_card = Rank.FIVE  # 5-high straight

    if is_straight and is_flush:
        return (STRAIGHT_FLUSH, high_card)

    # Count rank occurrences
    rank_counts: dict[Rank, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    groups = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    counts = [g[1] for g in groups]
    group_ranks = [g[0] for g in groups]

    if counts[0] == 4:
        return (FOUR_OF_A_KIND, group_ranks[0], group_ranks[1])

    if counts[0] == 3 and counts[1] == 2:
        return (FULL_HOUSE, group_ranks[0], group_ranks[1])

    if is_flush:
        return (FLUSH, *ranks)

    if is_straight:
        return (STRAIGHT, high_card)

    if counts[0] == 3:
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (THREE_OF_A_KIND, group_ranks[0], *kickers)

    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kicker = [r for r, c in rank_counts.items() if c == 1][0]
        return (TWO_PAIR, pairs[0], pairs[1], kicker)

    if counts[0] == 2:
        pair_rank = group_ranks[0]
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (ONE_PAIR, pair_rank, *kickers)

    return (HIGH_CARD, *ranks)


# Try to import C++ accelerated evaluator
try:
    from gambletron_engine import fast_evaluate_hand as _cpp_eval

    def evaluate_hand_fast(card_ints: List[int]) -> int:
        """C++ accelerated hand evaluation. Takes card integers, returns comparable score."""
        return _cpp_eval(card_ints)

    HAS_CPP_EVAL = True
except ImportError:
    HAS_CPP_EVAL = False

    def evaluate_hand_fast(card_ints: List[int]) -> int:
        cards = [Card(i) for i in card_ints]
        score = evaluate_hand(cards)
        # Pack into single int for fast comparison
        result = 0
        for i, v in enumerate(reversed(score)):
            result |= int(v) << (i * 4)
        return result
