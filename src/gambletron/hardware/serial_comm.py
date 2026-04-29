"""Serial hardware implementation: connects to Arduino controllers via USB.

Requires pyserial: pip install pyserial
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from gambletron.hardware.interface import (
    CardInput,
    ChipInterface,
    SeatSensor,
    TableController,
)
from gambletron.hardware.protocol import Message, RFIDCardMap
from gambletron.poker.card import Card


class SerialConnection:
    """Manages a serial connection to an Arduino."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 1.0,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None
        self._lock = threading.Lock()
        self._callbacks: Dict[str, List[Callable]] = {}
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False

    def connect(self) -> None:
        """Open the serial connection."""
        try:
            import serial
        except ImportError:
            raise RuntimeError(
                "pyserial not installed. Install with: pip install pyserial"
            )

        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen, daemon=True
        )
        self._listener_thread.start()

    def disconnect(self) -> None:
        """Close the serial connection."""
        self._running = False
        if self._listener_thread:
            self._listener_thread.join(timeout=2.0)
        if self._serial:
            self._serial.close()
            self._serial = None

    def send(self, msg: Message) -> None:
        """Send a message to the Arduino."""
        if not self._serial:
            raise RuntimeError("Not connected")
        with self._lock:
            self._serial.write(msg.serialize())

    def on(self, msg_type: str, callback: Callable[[Message], None]) -> None:
        """Register a callback for a message type."""
        self._callbacks.setdefault(msg_type, []).append(callback)

    def _listen(self) -> None:
        """Background thread that reads incoming messages."""
        buffer = b""
        while self._running and self._serial:
            try:
                data = self._serial.read(self._serial.in_waiting or 1)
                if not data:
                    continue
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        try:
                            msg = Message.deserialize(line)
                            for cb in self._callbacks.get(msg.type, []):
                                cb(msg)
                        except (json.JSONDecodeError, KeyError):
                            pass  # Ignore malformed messages
            except Exception:
                if self._running:
                    time.sleep(0.1)


class SerialCardInput(CardInput):
    """Reads cards from RFID readers via serial connection."""

    def __init__(
        self, connection: SerialConnection, rfid_map: RFIDCardMap
    ) -> None:
        self._conn = connection
        self._rfid_map = rfid_map
        self._pending: Dict[int, Card] = {}
        self._board_cards: List[Card] = []
        self._event = threading.Event()

        self._conn.on("card", self._on_card_detected)

    def _on_card_detected(self, msg: Message) -> None:
        seat = msg.data.get("seat", -1)
        rfid_id = msg.data.get("rfid", "")
        card_int = self._rfid_map.lookup(rfid_id)
        if card_int is not None:
            card = Card(card_int)
            if seat >= 0:
                self._pending[seat] = card
            else:
                self._board_cards.append(card)
            self._event.set()

    def wait_for_card(self, seat: int, timeout: float = 30.0) -> Optional[Card]:
        self._event.clear()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if seat in self._pending:
                return self._pending.pop(seat)
            self._event.wait(timeout=0.5)
            self._event.clear()
        return None

    def wait_for_community_cards(
        self, count: int, timeout: float = 30.0
    ) -> List[Card]:
        deadline = time.time() + timeout
        while len(self._board_cards) < count and time.time() < deadline:
            self._event.wait(timeout=0.5)
            self._event.clear()
        result = self._board_cards[:count]
        self._board_cards = self._board_cards[count:]
        return result

    def reset(self) -> None:
        self._pending.clear()
        self._board_cards.clear()


class SerialChipInterface(ChipInterface):
    """Manages chips via serial commands to the chip controller Arduino."""

    def __init__(self, connection: SerialConnection) -> None:
        self._conn = connection
        self._stacks: Dict[int, int] = {}
        self._conn.on("chips", self._on_chips_count)

    def _on_chips_count(self, msg: Message) -> None:
        seat = msg.data.get("seat", -1)
        amount = msg.data.get("amount", 0)
        if seat >= 0:
            self._stacks[seat] = amount

    def get_player_stack(self, seat: int) -> int:
        return self._stacks.get(seat, 0)

    def dispense_chips(self, seat: int, amount: int) -> bool:
        from gambletron.hardware.protocol import msg_dispense
        self._conn.send(msg_dispense(seat, amount))
        self._stacks[seat] = self._stacks.get(seat, 0) + amount
        return True

    def collect_bet(self, seat: int, amount: int) -> bool:
        from gambletron.hardware.protocol import msg_collect
        self._conn.send(msg_collect(seat, amount))
        current = self._stacks.get(seat, 0)
        self._stacks[seat] = max(0, current - amount)
        return True

    def collect_pot(self) -> int:
        from gambletron.hardware.protocol import msg_collect_pot
        self._conn.send(msg_collect_pot())
        return 0  # Actual amount comes from game state

    def award_pot(self, seat: int, amount: int) -> bool:
        return self.dispense_chips(seat, amount)


class SerialSeatSensor(SeatSensor):
    """Detects seat occupancy via serial sensor data."""

    def __init__(self, connection: SerialConnection) -> None:
        self._conn = connection
        self._occupied: Dict[int, bool] = {}
        self._conn.on("seat", self._on_seat_status)

    def _on_seat_status(self, msg: Message) -> None:
        seat = msg.data.get("seat", -1)
        occupied = msg.data.get("occupied", False)
        if seat >= 0:
            self._occupied[seat] = occupied

    def get_occupied_seats(self) -> List[int]:
        return sorted(s for s, occ in self._occupied.items() if occ)

    def is_seat_occupied(self, seat: int) -> bool:
        return self._occupied.get(seat, False)


class SerialTableController(TableController):
    """Physical table controller using serial connections to Arduinos."""

    def __init__(
        self,
        dealer_port: str,
        chip_port: str,
        baudrate: int = 115200,
    ) -> None:
        self._dealer_conn = SerialConnection(dealer_port, baudrate)
        self._chip_conn = SerialConnection(chip_port, baudrate)
        self._rfid_map = RFIDCardMap()

        self._card_input = SerialCardInput(self._dealer_conn, self._rfid_map)
        self._chip_interface = SerialChipInterface(self._chip_conn)
        self._seat_sensor = SerialSeatSensor(self._dealer_conn)

    def connect(self) -> None:
        self._dealer_conn.connect()
        self._chip_conn.connect()

    def disconnect(self) -> None:
        self._dealer_conn.disconnect()
        self._chip_conn.disconnect()

    def get_card_input(self) -> CardInput:
        return self._card_input

    def get_chip_interface(self) -> ChipInterface:
        return self._chip_interface

    def get_seat_sensor(self) -> SeatSensor:
        return self._seat_sensor

    def deal_card_to(self, seat: int) -> None:
        from gambletron.hardware.protocol import msg_deal_card
        self._dealer_conn.send(msg_deal_card(seat))

    def deal_community(self, count: int) -> None:
        from gambletron.hardware.protocol import msg_deal_board
        self._dealer_conn.send(msg_deal_board(count))

    def signal_player_turn(self, seat: int) -> None:
        from gambletron.hardware.protocol import msg_led
        # Turn off all LEDs, turn on the active one
        for s in range(6):
            self._dealer_conn.send(msg_led(s, "off"))
        self._dealer_conn.send(msg_led(seat, "on"))

    def signal_hand_over(self) -> None:
        for s in range(6):
            from gambletron.hardware.protocol import msg_led
            self._dealer_conn.send(msg_led(s, "off"))
