"""HID-based RFID card readers for physical table.

Each seat has a USB RFID reader that presents as an HID keyboard device.
When a card is scanned, the reader types the UID digits followed by Enter.
We use evdev to read these events directly, grabbing each device exclusively
so the UIDs don't leak into the terminal or pygame.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from gambletron.hardware.protocol import RFIDCardMap
from gambletron.poker.card import Card

# Evdev key code to character mapping for digits + Enter
_KEY_MAP = {
    2: "1", 3: "2", 4: "3", 5: "4", 6: "5",
    7: "6", 8: "7", 9: "8", 10: "9", 11: "0",
    28: "\n",
}

SEAT_DEVICE_PATHS = [
    "/dev/input/event0",  # seat 0
    "/dev/input/event1",  # seat 1
    "/dev/input/event2",  # seat 2
    "/dev/input/event3",  # seat 3
    "/dev/input/event4",  # seat 4
    "/dev/input/event5",  # seat 5
]


class HIDCardReader:
    """Reads RFID UIDs from a single HID input device."""

    def __init__(self, device_path: str, seat: int, rfid_map: RFIDCardMap) -> None:
        self._path = device_path
        self._seat = seat
        self._rfid_map = rfid_map
        self._device = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._buffer = ""
        self._cards: List[Card] = []
        self._event = threading.Event()
        self._lock = threading.Lock()

    def open(self) -> None:
        import evdev
        self._device = evdev.InputDevice(self._path)
        self._device.grab()
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._device:
            try:
                self._device.ungrab()
            except Exception:
                pass
            self._device.close()
            self._device = None
        if self._thread:
            self._thread.join(timeout=2.0)

    def _listen(self) -> None:
        import evdev
        while self._running and self._device:
            try:
                for event in self._device.read_loop():
                    if not self._running:
                        break
                    if event.type != evdev.ecodes.EV_KEY:
                        continue
                    # Only process key-down events (value == 1)
                    if event.value != 1:
                        continue
                    char = _KEY_MAP.get(event.code)
                    if char is None:
                        continue
                    if char == "\n":
                        self._process_uid(self._buffer)
                        self._buffer = ""
                    else:
                        self._buffer += char
            except OSError:
                if self._running:
                    import time
                    time.sleep(0.1)

    def _process_uid(self, uid: str) -> None:
        if not uid:
            return
        card_int = self._rfid_map.lookup(uid)
        if card_int is not None:
            with self._lock:
                self._cards.append(Card(card_int))
                self._event.set()

    def pop_card(self, timeout: float = 30.0) -> Optional[Card]:
        """Wait for and return the next card detected at this seat."""
        deadline = __import__("time").time() + timeout
        while __import__("time").time() < deadline:
            with self._lock:
                if self._cards:
                    return self._cards.pop(0)
            self._event.wait(timeout=0.5)
            self._event.clear()
        return None

    @property
    def cards_buffered(self) -> int:
        with self._lock:
            return len(self._cards)

    def clear(self) -> None:
        with self._lock:
            self._cards.clear()
            self._event.clear()


class HIDCardReaderPool:
    """Manages one HID RFID reader per seat."""

    def __init__(
        self,
        device_paths: Optional[List[str]] = None,
        num_seats: int = 6,
    ) -> None:
        paths = device_paths or SEAT_DEVICE_PATHS[:num_seats]
        self._rfid_map = RFIDCardMap()
        self._readers: List[HIDCardReader] = [
            HIDCardReader(path, seat, self._rfid_map)
            for seat, path in enumerate(paths)
        ]

    def open(self) -> None:
        for reader in self._readers:
            reader.open()

    def close(self) -> None:
        for reader in self._readers:
            reader.close()

    def wait_for_card(self, seat: int, timeout: float = 30.0) -> Optional[Card]:
        return self._readers[seat].pop_card(timeout)

    def wait_all_cards(self, cards_per_seat: int, num_seats: int, timeout: float = 8.0) -> Dict[int, List[Card]]:
        """Wait until every seat has the expected number of cards. Returns {seat: [cards]}."""
        import time
        deadline = time.time() + timeout
        result: Dict[int, List[Card]] = {s: [] for s in range(num_seats)}

        while time.time() < deadline:
            done = True
            for seat in range(num_seats):
                while len(result[seat]) < cards_per_seat:
                    card = self._readers[seat].pop_card(timeout=0.1)
                    if card is None:
                        break
                    result[seat].append(card)
                if len(result[seat]) < cards_per_seat:
                    done = False
            if done:
                break
            time.sleep(0.1)

        return result

    def clear_all(self) -> None:
        for reader in self._readers:
            reader.clear()

    def reader(self, seat: int) -> HIDCardReader:
        return self._readers[seat]
