"""Physical table controller for RPi 5 with GPIO dealer and HID RFID readers.

Composes:
  - GPIODealer: triggers the physical card dealer via GPIO pins
  - HIDCardReaderPool: reads RFID-tagged cards from per-seat HID readers
  - SerialChipInterface: manages chips via serial Arduino
  - SerialSeatSensor: detects seat occupancy via serial Arduino

Community cards are virtual: randomly selected from cards not dealt physically.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

from gambletron.hardware.gpio_dealer import GPIODealer
from gambletron.hardware.hid_reader import HIDCardReaderPool
from gambletron.hardware.interface import (
    CardInput,
    ChipInterface,
    SeatSensor,
    TableController,
)
from gambletron.hardware.serial_comm import (
    SerialChipInterface,
    SerialConnection,
    SerialSeatSensor,
)
from gambletron.poker.card import Card

DEAL_WAIT_SECONDS = 8.0
CARDS_PER_SEAT = 2
NUM_SEATS = 6


class PhysicalCardInput(CardInput):
    """CardInput backed by HID RFID readers and virtual community cards.

    Hole cards come from the physical RFID readers (pre-buffered after
    the dealer fires). Community cards are randomly generated from the
    remaining 40 cards in the deck.
    """

    def __init__(self, reader_pool: HIDCardReaderPool, num_seats: int = NUM_SEATS) -> None:
        self._readers = reader_pool
        self._num_seats = num_seats
        self._hole_cards: Dict[int, List[Card]] = {}
        self._dealt_card_ints: set = set()
        self._community_generated: List[Card] = []

    def load_dealt_cards(self, cards_by_seat: Dict[int, List[Card]]) -> None:
        """Store hole cards read from RFID after the dealer fires."""
        self._hole_cards = cards_by_seat
        self._dealt_card_ints = set()
        for cards in cards_by_seat.values():
            for c in cards:
                self._dealt_card_ints.add(c.int_value)

    def wait_for_card(self, seat: int, timeout: float = 30.0) -> Optional[Card]:
        """Return the next pre-buffered hole card for this seat."""
        cards = self._hole_cards.get(seat, [])
        if cards:
            return cards.pop(0)
        return None

    def wait_for_community_cards(self, count: int, timeout: float = 30.0) -> List[Card]:
        """Generate virtual community cards from the remaining deck."""
        excluded = self._dealt_card_ints | {c.int_value for c in self._community_generated}
        available = [i for i in range(52) if i not in excluded]
        chosen = random.sample(available, count)
        cards = [Card(c) for c in chosen]
        self._community_generated.extend(cards)
        return cards

    def reset(self) -> None:
        self._hole_cards.clear()
        self._dealt_card_ints.clear()
        self._community_generated.clear()


class PhysicalTableController(TableController):
    """Full physical table: GPIO dealer, HID RFID readers, serial chip controller."""

    def __init__(
        self,
        chip_port: str,
        device_paths: Optional[List[str]] = None,
        num_seats: int = NUM_SEATS,
        baudrate: int = 115200,
    ) -> None:
        self._num_seats = num_seats
        self._gpio_dealer = GPIODealer()
        self._reader_pool = HIDCardReaderPool(device_paths, num_seats)
        self._card_input = PhysicalCardInput(self._reader_pool, num_seats)

        self._chip_conn = SerialConnection(chip_port, baudrate)
        self._chip_interface = SerialChipInterface(self._chip_conn)
        self._seat_sensor = SerialSeatSensor(self._chip_conn)

    def connect(self) -> None:
        self._gpio_dealer.open()
        self._reader_pool.open()
        self._chip_conn.connect()

    def disconnect(self) -> None:
        self._gpio_dealer.close()
        self._reader_pool.close()
        self._chip_conn.disconnect()

    def trigger_deal(self) -> Dict[int, List[Card]]:
        """Trigger the physical dealer and wait for all RFID reads.

        Returns a dict of {seat: [card, card]} for all seats.
        Raises RuntimeError if not all cards are detected in time.
        """
        self._reader_pool.clear_all()
        self._card_input.reset()

        self._gpio_dealer.trigger()

        cards = self._reader_pool.wait_all_cards(
            cards_per_seat=CARDS_PER_SEAT,
            num_seats=self._num_seats,
            timeout=DEAL_WAIT_SECONDS,
        )

        for seat in range(self._num_seats):
            if len(cards.get(seat, [])) < CARDS_PER_SEAT:
                raise RuntimeError(
                    f"Seat {seat}: expected {CARDS_PER_SEAT} cards, "
                    f"got {len(cards.get(seat, []))}"
                )

        self._card_input.load_dealt_cards(cards)
        return cards

    # ── TableController interface ─────────────────────────────────────────────

    def get_card_input(self) -> CardInput:
        return self._card_input

    def get_chip_interface(self) -> ChipInterface:
        return self._chip_interface

    def get_seat_sensor(self) -> SeatSensor:
        return self._seat_sensor

    def deal_card_to(self, seat: int) -> None:
        pass  # cards already dealt physically

    def deal_community(self, count: int) -> None:
        pass  # community cards are virtual

    def signal_player_turn(self, seat: int) -> None:
        pass

    def signal_hand_over(self) -> None:
        pass
