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
    HideReadyButtonEvent,
    PreflopEvent,
    ShowdownEvent,
    ShowReadyButtonEvent,
    WinnerEvent,
)


class NullDisplaySink:
    """No-op sink used when no display is attached."""

    def hand_start(self, hand_num: int, dealer_pos: int, num_players: int,
                   player_stacks: List[int]) -> None: pass
    def preflop(self, pot: int, current_player: int, dealer_pos: int,
                player_stacks: List[int]) -> None: pass
    def action(self, seat: int, description: str, pot: int,
               current_player: Optional[int], player_folded: List[bool],
               betting_round: str, player_stacks: List[int]) -> None: pass
    def community_cards(self, betting_round: str, cards: List[int],
                        pot: int, current_player: Optional[int],
                        player_stacks: List[int]) -> None: pass
    def showdown(self, hole_cards: Dict[int, List[int]],
                 community_cards: List[int], pot: int,
                 player_stacks: List[int]) -> None: pass
    def winner(self, seats: List[int], pot_won: int,
               hand_desc: Dict[int, str]) -> None: pass
    def hand_end(self) -> None: pass
    def show_ready_button(self) -> None: pass
    def hide_ready_button(self) -> None: pass


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

    def hand_start(self, hand_num: int, dealer_pos: int, num_players: int,
                   player_stacks: List[int]) -> None:
        self._send(HandStartEvent(
            hand_num=hand_num, dealer_pos=dealer_pos, num_players=num_players,
            player_stacks=list(player_stacks),
        ))

    def preflop(self, pot: int, current_player: int, dealer_pos: int,
                player_stacks: List[int]) -> None:
        self._send(PreflopEvent(
            pot=pot, current_player=current_player, dealer_pos=dealer_pos,
            player_stacks=list(player_stacks),
        ))

    def action(self, seat: int, description: str, pot: int,
               current_player: Optional[int], player_folded: List[bool],
               betting_round: str, player_stacks: List[int]) -> None:
        self._send(ActionEvent(
            seat=seat, description=description, pot=pot,
            current_player=current_player,
            player_folded=list(player_folded),
            betting_round=betting_round,
            player_stacks=list(player_stacks),
        ))

    def community_cards(self, betting_round: str, cards: List[int],
                        pot: int, current_player: Optional[int],
                        player_stacks: List[int]) -> None:
        self._send(CommunityCardsEvent(
            betting_round=betting_round,
            community_cards=list(cards),
            pot=pot,
            current_player=current_player,
            player_stacks=list(player_stacks),
        ))

    def showdown(self, hole_cards: Dict[int, List[int]],
                 community_cards: List[int], pot: int,
                 player_stacks: List[int]) -> None:
        self._send(ShowdownEvent(
            hole_cards={k: list(v) for k, v in hole_cards.items()},
            community_cards=list(community_cards),
            pot=pot,
            player_stacks=list(player_stacks),
        ))

    def winner(self, seats: List[int], pot_won: int,
               hand_desc: Dict[int, str]) -> None:
        self._send(WinnerEvent(
            seats=list(seats), pot_won=pot_won, hand_desc=dict(hand_desc),
        ))

    def hand_end(self) -> None:
        self._send(HandEndEvent())

    def show_ready_button(self) -> None:
        self._send(ShowReadyButtonEvent())

    def hide_ready_button(self) -> None:
        self._send(HideReadyButtonEvent())
