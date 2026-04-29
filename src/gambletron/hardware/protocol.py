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
_RFID_MAPPING = {
    "2737139972": 3,
    "1512403204": 3,
    "2791915780": 7,
    "2657698052": 7,
    "2657685764": 11,
    "2573799684": 11,
    "0438399236": 15,
    "2686546180": 15,
    "2657689860": 19,
    "2791907588": 19,
    "2686550276": 23,
    "2250342660": 23,
    "2686513412": 27,
    "2183196932": 27,
    "2519056644": 31,
    "1512423684": 31,
    "2657706244": 35,
    "2573820164": 35,
    "2737135876": 39,
    "0000017924": 39,
    "2737189124": 43,
    "2586194180": 43,
    "2737168644": 47,
    "2519064836": 47,
    "2737185028": 51,
    "0000000064": 51,
    "060625668": 0,
    "2708304132": 0,
    "3664601348": 4,
    "2708300036": 4,
    "2737144068": 8,
    "2519040260": 8,
    "0438382852": 12,
    "2686529796": 12,
    "2737152260": 16,
    "1512415492": 16,
    "2737148164": 20,
    "2519044356": 20,
    "0169922820": 24,
    "2686505220": 24,
    "2755150084": 28,
    "2318942468": 28,
    "3664613636": 32,
    "2708312324": 32,
    "3060658436": 36,
    "2708336900": 36,
    "2791936260": 40,
    "2657718532": 40,
    "2737164548": 44,
    "2519060740": 44,
    "2793287940": 48,
    "4202574084": 48,
    "2686554372": 1,
    "0438407428": 1,
    "2686501124": 5,
    "2183184644": 5,
    "2686533892": 9,
    "2250326276": 9,
    "2657681668": 13,
    "2573795588": 13,
    "2657669380": 17,
    "2305347844": 17,
    "2708328708": 21,
    "3664630020": 21,
    "2657702148": 25,
    "2791919876": 25,
    "2708320516": 29,
    "3060642052": 29,
    "2708291844": 33,
    "3060613380": 33,
    "3060617476": 37,
    "2708295940": 37,
    "2573836548": 41,
    "2791940356": 41,
    "2708345092": 45,
    "3127775492": 45,
    "2250313988": 49,
    "2686521604": 49,
    "2708340996": 2,
    "3127771396": 2,
    "2737172740": 6,
    "2519068932": 6,
    "2183192836": 10,
    "0169926916": 10,
    "0000017156": 14,
    "2791911684": 14,
    "1780879620": 18,
    "2737180932": 18,
    "2724786436": 22,
    "2657677572": 22,
    "2686517508": 26,
    "0438370564": 26,
    "2657710340": 30,
    "2573824260": 30,
    "0438378756": 34,
    "2686525700": 34,
    "2686537988": 38,
    "2250330372": 38,
    "2737156356": 42,
    "2519052548": 42,
    "3664625924": 46,
    "3060646148": 46,
    "3060637956": 50,
    "2708316420": 50,
}


class RFIDCardMap:
    """Maps RFID tag identifiers to card integers using hardcoded UID mapping."""

    def __init__(self) -> None:
        self._rfid_to_card: Dict[str, int] = dict(_RFID_MAPPING)

    def lookup(self, rfid_id: str) -> Optional[int]:
        return self._rfid_to_card.get(rfid_id)
