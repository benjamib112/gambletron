"""Random player that picks uniformly from legal actions."""

from __future__ import annotations

import random

from gambletron.players.base import Player
from gambletron.poker.state import Action, ActionType, VisibleGameState


class RandomPlayer(Player):
    def __init__(self, name: str = "Random", seed: int | None = None) -> None:
        super().__init__(name)
        self._rng = random.Random(seed)

    def get_action(self, state: VisibleGameState) -> Action:
        actions = _get_simple_legal_actions(state)
        return self._rng.choice(actions)


def _get_simple_legal_actions(state: VisibleGameState) -> list[Action]:
    """Compute legal actions from visible state (without full GameState)."""
    current_bet = max(state.player_bets)
    my_bet = state.player_bets[state.my_seat]
    my_stack = state.player_stacks[state.my_seat]
    to_call = current_bet - my_bet

    actions = []

    if to_call > 0:
        actions.append(Action.fold())

    actions.append(Action.call())

    chips_after_call = my_stack - to_call
    if chips_after_call > 0:
        min_raise_to = current_bet + state.min_raise
        max_raise_to = my_bet + my_stack
        if max_raise_to > current_bet:
            actual_min = min(min_raise_to, max_raise_to)
            actions.append(Action.raise_to(actual_min))
            if max_raise_to > actual_min:
                actions.append(Action.raise_to(max_raise_to))

    return actions
