"""Events sent from the game process to the display process via a Queue."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HandStartEvent:
    """Clears the display and sets dealer for a new hand."""
    hand_num: int
    dealer_pos: int
    num_players: int
    player_stacks: List[int]


@dataclass
class PreflopEvent:
    """Initial state after blinds are posted."""
    pot: int
    current_player: int
    dealer_pos: int
    player_stacks: List[int]


@dataclass
class ActionEvent:
    """Fired after each player action is applied."""
    seat: int
    description: str          # e.g. "folds", "calls $100", "raises to $300", "checks"
    pot: int
    current_player: Optional[int]   # seat index of next player, or None
    player_folded: List[bool]
    betting_round: str        # 'PREFLOP', 'FLOP', 'TURN', 'RIVER'
    player_stacks: List[int]


@dataclass
class CommunityCardsEvent:
    """Fired when new community cards are dealt at a street transition."""
    betting_round: str        # 'FLOP', 'TURN', 'RIVER'
    community_cards: List[int]    # complete list of card ints so far
    pot: int
    current_player: Optional[int]
    player_stacks: List[int]


@dataclass
class ShowdownEvent:
    """Fired at showdown to reveal hole cards of all remaining players."""
    hole_cards: Dict[int, List[int]]  # seat -> [card_int, card_int]
    community_cards: List[int]
    pot: int
    player_stacks: List[int]


@dataclass
class WinnerEvent:
    """Fired after the pot is awarded. Persists until the next HandStartEvent."""
    seats: List[int]
    pot_won: int
    hand_desc: Dict[int, str]     # seat -> hand name, e.g. "Flush, Ace-high"


@dataclass
class HandEndEvent:
    """Fired at the very end of a hand (after winner). Currently a no-op on display."""
    pass


@dataclass
class ShowReadyButtonEvent:
    """Show a 'Ready to Deal' button on the display for the physical table."""
    pass


@dataclass
class HideReadyButtonEvent:
    """Hide the 'Ready to Deal' button."""
    pass
