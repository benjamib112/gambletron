from __future__ import annotations

import random
from enum import IntEnum
from typing import List


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    def __str__(self) -> str:
        return _SUIT_SYMBOLS[self.value]


class Rank(IntEnum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12

    def __str__(self) -> str:
        return _RANK_SYMBOLS[self.value]


_SUIT_SYMBOLS = ["c", "d", "h", "s"]
_RANK_SYMBOLS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

_RANK_FROM_CHAR = {c: Rank(i) for i, c in enumerate(_RANK_SYMBOLS)}
_SUIT_FROM_CHAR = {c: Suit(i) for i, c in enumerate(_SUIT_SYMBOLS)}


class Card:
    """A playing card. Internally stored as integer 0-51 for fast C++ interop.

    Encoding: card_int = rank * 4 + suit
    """

    __slots__ = ("_int",)

    def __init__(self, card_int: int) -> None:
        self._int = card_int

    @classmethod
    def from_rank_suit(cls, rank: Rank, suit: Suit) -> Card:
        return cls(int(rank) * 4 + int(suit))

    @classmethod
    def from_str(cls, s: str) -> Card:
        """Parse a card string like 'As', 'Th', '2c'."""
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s!r}")
        rank = _RANK_FROM_CHAR.get(s[0])
        suit = _SUIT_FROM_CHAR.get(s[1])
        if rank is None or suit is None:
            raise ValueError(f"Invalid card string: {s!r}")
        return cls.from_rank_suit(rank, suit)

    @property
    def rank(self) -> Rank:
        return Rank(self._int // 4)

    @property
    def suit(self) -> Suit:
        return Suit(self._int % 4)

    @property
    def int_value(self) -> int:
        return self._int

    def __repr__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Card):
            return self._int == other._int
        return NotImplemented

    def __hash__(self) -> int:
        return self._int

    def __lt__(self, other: Card) -> bool:
        return self._int < other._int


class Deck:
    """Standard 52-card deck with shuffle and deal operations."""

    def __init__(self, seed: int | None = None) -> None:
        self._cards = list(range(52))
        self._rng = random.Random(seed)
        self._index = 0

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)
        self._index = 0

    def deal(self, n: int = 1) -> List[Card]:
        if self._index + n > 52:
            raise RuntimeError("Not enough cards in deck")
        cards = [Card(self._cards[self._index + i]) for i in range(n)]
        self._index += n
        return cards

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    @property
    def remaining(self) -> int:
        return 52 - self._index

    def reset(self, seed: int | None = None) -> None:
        self._cards = list(range(52))
        if seed is not None:
            self._rng = random.Random(seed)
        self._index = 0
