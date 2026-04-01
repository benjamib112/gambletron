"""Display event sinks: route game events to the display process."""

from __future__ import annotations

import multiprocessing
import queue
from typing import Dict, List, Optional

from gambletron.display.events import (
    ActionEvent,
    CommunityCardsEvent,
    HandEndEvent,
    HandStartEvent,
    PreflopEvent,
    ShowdownEvent,
    WinnerEvent,
)


class NullDisplaySink:
    """No-op sink used when no display is attached."""

    def hand_start(self, hand_num: int, dealer_pos: int, num_players: int) -> None: pass
    def preflop(self, pot: int, current_player: int, dealer_pos: int) -> None: pass
    def action(self, seat: int, description: str, pot: int,
               current_player: Optional[int], player_folded: List[bool],
               betting_round: str) -> None: pass
    def community_cards(self, betting_round: str, cards: List[int],
                        pot: int, current_player: Optional[int]) -> None: pass
    def showdown(self, hole_cards: Dict[int, List[int]],
                 community_cards: List[int], pot: int) -> None: pass
    def winner(self, seats: List[int], pot_won: int,
               hand_desc: Dict[int, str]) -> None: pass
    def hand_end(self) -> None: pass


class QueueDisplaySink:
    """Sends display events into a multiprocessing.Queue consumed by the display process.

    All methods are non-blocking and silently drop events if the queue is full,
    so a slow display can never stall the game loop.
    """

    def __init__(self, q: multiprocessing.Queue) -> None:
        self._q = q

    def _send(self, event: object) -> None:
        try:
            self._q.put_nowait(event)
        except (queue.Full, Exception):
            pass  # never let display issues affect the game

    def hand_start(self, hand_num: int, dealer_pos: int, num_players: int) -> None:
        self._send(HandStartEvent(
            hand_num=hand_num, dealer_pos=dealer_pos, num_players=num_players,
        ))

    def preflop(self, pot: int, current_player: int, dealer_pos: int) -> None:
        self._send(PreflopEvent(
            pot=pot, current_player=current_player, dealer_pos=dealer_pos,
        ))

    def action(self, seat: int, description: str, pot: int,
               current_player: Optional[int], player_folded: List[bool],
               betting_round: str) -> None:
        self._send(ActionEvent(
            seat=seat, description=description, pot=pot,
            current_player=current_player,
            player_folded=list(player_folded),
            betting_round=betting_round,
        ))

    def community_cards(self, betting_round: str, cards: List[int],
                        pot: int, current_player: Optional[int]) -> None:
        self._send(CommunityCardsEvent(
            betting_round=betting_round,
            community_cards=list(cards),
            pot=pot,
            current_player=current_player,
        ))

    def showdown(self, hole_cards: Dict[int, List[int]],
                 community_cards: List[int], pot: int) -> None:
        self._send(ShowdownEvent(
            hole_cards={k: list(v) for k, v in hole_cards.items()},
            community_cards=list(community_cards),
            pot=pot,
        ))

    def winner(self, seats: List[int], pot_won: int,
               hand_desc: Dict[int, str]) -> None:
        self._send(WinnerEvent(
            seats=list(seats), pot_won=pot_won, hand_desc=dict(hand_desc),
        ))

    def hand_end(self) -> None:
        self._send(HandEndEvent())
