"""Simulated hardware: software implementation for training and digital demo."""

from __future__ import annotations

from typing import Dict, List, Optional

from gambletron.hardware.interface import (
    CardInput,
    ChipInterface,
    SeatSensor,
    TableController,
)
from gambletron.poker.card import Card, Deck


class SimulatedCardInput(CardInput):
    """Simulates card detection using a software deck."""

    def __init__(self, deck: Optional[Deck] = None) -> None:
        self.deck = deck or Deck()
        self._detected: Dict[int, List[Card]] = {}

    def wait_for_card(self, seat: int, timeout: float = 30.0) -> Optional[Card]:
        card = self.deck.deal_one()
        self._detected.setdefault(seat, []).append(card)
        return card

    def wait_for_community_cards(
        self, count: int, timeout: float = 30.0
    ) -> List[Card]:
        return self.deck.deal(count)

    def reset(self) -> None:
        self._detected.clear()

    def inject_card(self, seat: int, card: Card) -> None:
        """Manually inject a specific card (for testing)."""
        self._detected.setdefault(seat, []).append(card)


class SimulatedChipInterface(ChipInterface):
    """Simulates chip management with in-memory tracking."""

    def __init__(self, num_seats: int = 6, starting_stack: int = 10000) -> None:
        self.stacks: Dict[int, int] = {
            i: starting_stack for i in range(num_seats)
        }
        self.pot = 0

    def get_player_stack(self, seat: int) -> int:
        return self.stacks.get(seat, 0)

    def dispense_chips(self, seat: int, amount: int) -> bool:
        self.stacks[seat] = self.stacks.get(seat, 0) + amount
        return True

    def collect_bet(self, seat: int, amount: int) -> bool:
        current = self.stacks.get(seat, 0)
        actual = min(amount, current)
        self.stacks[seat] = current - actual
        self.pot += actual
        return True

    def collect_pot(self) -> int:
        total = self.pot
        self.pot = 0
        return total

    def award_pot(self, seat: int, amount: int) -> bool:
        self.stacks[seat] = self.stacks.get(seat, 0) + amount
        return True


class SimulatedSeatSensor(SeatSensor):
    """Simulates seat occupancy detection."""

    def __init__(self, occupied: Optional[List[int]] = None) -> None:
        self._occupied = set(occupied or [])

    def get_occupied_seats(self) -> List[int]:
        return sorted(self._occupied)

    def is_seat_occupied(self, seat: int) -> bool:
        return seat in self._occupied

    def set_occupied(self, seats: List[int]) -> None:
        self._occupied = set(seats)


class SimulatedTableController(TableController):
    """Simulated table controller for digital demo and training."""

    def __init__(
        self,
        num_seats: int = 6,
        starting_stack: int = 10000,
        deck: Optional[Deck] = None,
    ) -> None:
        self._card_input = SimulatedCardInput(deck)
        self._chip_interface = SimulatedChipInterface(num_seats, starting_stack)
        self._seat_sensor = SimulatedSeatSensor(list(range(num_seats)))

    def get_card_input(self) -> CardInput:
        return self._card_input

    def get_chip_interface(self) -> ChipInterface:
        return self._chip_interface

    def get_seat_sensor(self) -> SeatSensor:
        return self._seat_sensor

    def deal_card_to(self, seat: int) -> None:
        pass  # No physical action needed

    def deal_community(self, count: int) -> None:
        pass  # No physical action needed

    def signal_player_turn(self, seat: int) -> None:
        pass  # No physical indicator

    def signal_hand_over(self) -> None:
        pass  # No physical indicator
