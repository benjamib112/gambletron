"""Serial message protocol for Pi <-> Arduino communication.

Messages are newline-delimited JSON for simplicity and debuggability.
Each message has a "type" field and type-specific data fields.

Outbound (Pi -> Arduino):
  DEAL_CARD     {"type": "deal", "seat": 0}
  DEAL_BOARD    {"type": "deal_board", "count": 3}
  DISPENSE      {"type": "dispense", "seat": 0, "amount": 500}
  COLLECT       {"type": "collect", "seat": 0, "amount": 100}
  COLLECT_POT   {"type": "collect_pot"}
  LED_ON        {"type": "led", "seat": 0, "state": "on"}
  LED_OFF       {"type": "led", "seat": 0, "state": "off"}
  RESET         {"type": "reset"}

Inbound (Arduino -> Pi):
  CARD_DETECTED {"type": "card", "seat": 0, "rfid": "A1B2C3D4", "card_id": 51}
  CHIPS_COUNT   {"type": "chips", "seat": 0, "amount": 9900}
  SEAT_STATUS   {"type": "seat", "seat": 0, "occupied": true}
  ACK           {"type": "ack", "ref": "deal"}
  ERROR         {"type": "error", "message": "dealer jam"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Message:
    type: str
    data: Dict[str, Any]

    def serialize(self) -> bytes:
        payload = {"type": self.type, **self.data}
        return (json.dumps(payload) + "\n").encode("utf-8")

    @classmethod
    def deserialize(cls, raw: bytes) -> Message:
        payload = json.loads(raw.decode("utf-8").strip())
        msg_type = payload.pop("type")
        return cls(type=msg_type, data=payload)


# Outbound message constructors
def msg_deal_card(seat: int) -> Message:
    return Message("deal", {"seat": seat})


def msg_deal_board(count: int) -> Message:
    return Message("deal_board", {"count": count})


def msg_dispense(seat: int, amount: int) -> Message:
    return Message("dispense", {"seat": seat, "amount": amount})


def msg_collect(seat: int, amount: int) -> Message:
    return Message("collect", {"seat": seat, "amount": amount})


def msg_collect_pot() -> Message:
    return Message("collect_pot", {})


def msg_led(seat: int, state: str = "on") -> Message:
    return Message("led", {"seat": seat, "state": state})


def msg_reset() -> Message:
    return Message("reset", {})


# RFID card mapping: maps RFID tag IDs to card integers (0-51)
class RFIDCardMap:
    """Maps RFID tag identifiers to card integers."""

    def __init__(self) -> None:
        self._rfid_to_card: Dict[str, int] = {}
        self._card_to_rfid: Dict[int, str] = {}

    def register(self, rfid_id: str, card_int: int) -> None:
        """Register an RFID tag to a specific card."""
        self._rfid_to_card[rfid_id] = card_int
        self._card_to_rfid[card_int] = rfid_id

    def lookup(self, rfid_id: str) -> Optional[int]:
        """Look up a card integer from an RFID tag ID."""
        return self._rfid_to_card.get(rfid_id)

    def reverse_lookup(self, card_int: int) -> Optional[str]:
        """Look up an RFID tag ID from a card integer."""
        return self._card_to_rfid.get(card_int)

    def load_from_dict(self, mapping: Dict[str, int]) -> None:
        """Load a complete mapping from a dictionary."""
        for rfid_id, card_int in mapping.items():
            self.register(rfid_id, card_int)

    @property
    def is_complete(self) -> bool:
        """Check if all 52 cards are mapped."""
        return len(self._rfid_to_card) == 52
