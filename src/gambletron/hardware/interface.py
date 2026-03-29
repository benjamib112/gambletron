"""Abstract hardware interfaces for the physical poker table.

These ABCs decouple the game engine from specific hardware implementations.
SimulatedHardware is used for training and digital demo.
SerialHardware connects to the physical Pi+Arduino table.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from gambletron.poker.card import Card


class CardInput(ABC):
    """Interface for receiving card identifications (e.g., from RFID readers)."""

    @abstractmethod
    def wait_for_card(self, seat: int, timeout: float = 30.0) -> Optional[Card]:
        """Wait for a card to be detected at a seat. Returns None on timeout."""
        ...

    @abstractmethod
    def wait_for_community_cards(
        self, count: int, timeout: float = 30.0
    ) -> List[Card]:
        """Wait for community cards to be detected on the board."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset card detection state for a new hand."""
        ...


class ChipInterface(ABC):
    """Interface for chip management (counting, dispensing, collecting)."""

    @abstractmethod
    def get_player_stack(self, seat: int) -> int:
        """Read the current chip count at a seat."""
        ...

    @abstractmethod
    def dispense_chips(self, seat: int, amount: int) -> bool:
        """Dispense chips to a player. Returns True on success."""
        ...

    @abstractmethod
    def collect_bet(self, seat: int, amount: int) -> bool:
        """Collect a bet from a player into the pot. Returns True on success."""
        ...

    @abstractmethod
    def collect_pot(self) -> int:
        """Collect all bets into the central pot. Returns total collected."""
        ...

    @abstractmethod
    def award_pot(self, seat: int, amount: int) -> bool:
        """Award pot winnings to a player."""
        ...


class SeatSensor(ABC):
    """Interface for detecting which seats are occupied."""

    @abstractmethod
    def get_occupied_seats(self) -> List[int]:
        """Return list of occupied seat numbers."""
        ...

    @abstractmethod
    def is_seat_occupied(self, seat: int) -> bool:
        """Check if a specific seat is occupied."""
        ...


class TableController(ABC):
    """High-level table controller combining all hardware interfaces."""

    @abstractmethod
    def get_card_input(self) -> CardInput:
        ...

    @abstractmethod
    def get_chip_interface(self) -> ChipInterface:
        ...

    @abstractmethod
    def get_seat_sensor(self) -> SeatSensor:
        ...

    @abstractmethod
    def deal_card_to(self, seat: int) -> None:
        """Signal the card dealer to deal a card to a seat."""
        ...

    @abstractmethod
    def deal_community(self, count: int) -> None:
        """Signal the dealer to deal community cards."""
        ...

    @abstractmethod
    def signal_player_turn(self, seat: int) -> None:
        """Indicate which player's turn it is (e.g., light an LED)."""
        ...

    @abstractmethod
    def signal_hand_over(self) -> None:
        """Signal that the hand is over."""
        ...
