"""Game state representation for Texas Hold'em."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from gambletron.poker.card import Card


class BettingRound(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class ActionType(IntEnum):
    FOLD = 0
    CALL = 1  # Also covers check (call of 0)
    RAISE = 2


@dataclass(frozen=True)
class Action:
    type: ActionType
    amount: int = 0  # Only used for RAISE; total chips put in pot this round

    def __repr__(self) -> str:
        if self.type == ActionType.FOLD:
            return "Fold"
        elif self.type == ActionType.CALL:
            return "Call"
        else:
            return f"Raise({self.amount})"

    @classmethod
    def fold(cls) -> Action:
        return cls(ActionType.FOLD)

    @classmethod
    def call(cls) -> Action:
        return cls(ActionType.CALL)

    @classmethod
    def raise_to(cls, amount: int) -> Action:
        return cls(ActionType.RAISE, amount)


@dataclass
class PlayerState:
    seat: int
    stack: int
    hole_cards: List[Card] = field(default_factory=list)
    is_folded: bool = False
    is_all_in: bool = False
    bet_this_round: int = 0
    bet_total: int = 0  # Total chips invested in pot this hand

    @property
    def is_active(self) -> bool:
        """Can this player still act (not folded and not all-in)?"""
        return not self.is_folded and not self.is_all_in

    @property
    def is_in_hand(self) -> bool:
        """Is this player still competing for the pot?"""
        return not self.is_folded


@dataclass
class GameState:
    """Complete state of a single hand of Texas Hold'em."""

    num_players: int
    dealer_pos: int
    small_blind: int = 50
    big_blind: int = 100

    players: List[PlayerState] = field(default_factory=list)
    community_cards: List[Card] = field(default_factory=list)
    pot: int = 0
    betting_round: BettingRound = BettingRound.PREFLOP
    current_player: int = 0
    last_raiser: Optional[int] = None
    min_raise: int = 100  # Minimum raise increment
    num_actions_this_round: int = 0
    is_hand_over: bool = False
    action_history: List[List[tuple]] = field(default_factory=list)
    # action_history[round] = [(player_seat, Action), ...]

    def __post_init__(self) -> None:
        if not self.action_history:
            self.action_history = [[] for _ in range(4)]

    @property
    def current_bet(self) -> int:
        """The highest bet any player has made this round."""
        if not self.players:
            return 0
        return max(p.bet_this_round for p in self.players)

    @property
    def players_in_hand(self) -> List[PlayerState]:
        return [p for p in self.players if p.is_in_hand]

    @property
    def active_players(self) -> List[PlayerState]:
        """Players who can still act (not folded, not all-in)."""
        return [p for p in self.players if p.is_active]

    @property
    def num_in_hand(self) -> int:
        return sum(1 for p in self.players if p.is_in_hand)

    def visible_to(self, seat: int) -> VisibleGameState:
        """Return the game state visible to a specific player."""
        return VisibleGameState(
            num_players=self.num_players,
            dealer_pos=self.dealer_pos,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            my_seat=seat,
            my_cards=list(self.players[seat].hole_cards),
            community_cards=list(self.community_cards),
            pot=self.pot,
            betting_round=self.betting_round,
            current_player=self.current_player,
            min_raise=self.min_raise,
            player_stacks=[p.stack for p in self.players],
            player_bets=[p.bet_this_round for p in self.players],
            player_folded=[p.is_folded for p in self.players],
            player_all_in=[p.is_all_in for p in self.players],
            action_history=[list(rnd) for rnd in self.action_history],
        )


@dataclass(frozen=True)
class VisibleGameState:
    """Game state from a specific player's perspective (no opponent hole cards)."""

    num_players: int
    dealer_pos: int
    small_blind: int
    big_blind: int
    my_seat: int
    my_cards: List[Card]
    community_cards: List[Card]
    pot: int
    betting_round: BettingRound
    current_player: int
    min_raise: int
    player_stacks: List[int]
    player_bets: List[int]
    player_folded: List[bool]
    player_all_in: List[bool]
    action_history: List[List[tuple]]

    @property
    def to_call(self) -> int:
        """How much the current player needs to add to call."""
        current_bet = max(self.player_bets)
        my_bet = self.player_bets[self.my_seat]
        return current_bet - my_bet

    @property
    def my_stack(self) -> int:
        return self.player_stacks[self.my_seat]
