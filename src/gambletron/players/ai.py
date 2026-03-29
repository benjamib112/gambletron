"""Pluribus-style AI player: blueprint strategy + real-time search."""

from __future__ import annotations

import enum
import random
from pathlib import Path
from typing import Optional

from gambletron.ai.belief import BeliefState
from gambletron.ai.blueprint import make_infoset_key
from gambletron.ai.search import RealTimeSearch, SubgameState
from gambletron.ai.strategy import Strategy
from gambletron.players.base import Player
from gambletron.poker.state import Action, ActionType, BettingRound, VisibleGameState


class Difficulty(enum.Enum):
    """AI difficulty levels.

    Easy:       Early snapshot, 20% action noise, no search
    Medium:     Mid snapshot, 5% action noise, no search
    Hard:       Later snapshot, no noise, no search
    Expert:     Full training, no noise, real-time search
    Superhuman: Full training (12.4B iterations), no noise, real-time search
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
    SUPERHUMAN = "superhuman"


# Default iteration targets for each difficulty snapshot during training
DIFFICULTY_SNAPSHOT_TARGETS = {
    Difficulty.EASY: 100_000,
    Difficulty.MEDIUM: 1_000_000,
    Difficulty.HARD: 10_000_000,
    Difficulty.EXPERT: 50_000_000,
    Difficulty.SUPERHUMAN: 12_400_000_000,  # Pluribus paper: ~12.4B iterations
}


# Noise levels per difficulty (probability of choosing a random legal action)
DIFFICULTY_NOISE = {
    Difficulty.EASY: 0.20,
    Difficulty.MEDIUM: 0.05,
    Difficulty.HARD: 0.0,
    Difficulty.EXPERT: 0.0,
    Difficulty.SUPERHUMAN: 0.0,
}


class AIPlayer(Player):
    """Pluribus-style AI that uses blueprint + real-time search.

    - Preflop: uses blueprint strategy directly (unless off-tree action)
    - Postflop: uses depth-limited real-time search (Superhuman only)
    """

    def __init__(
        self,
        name: str = "Pluribus",
        blueprint: Optional[Strategy] = None,
        blueprint_path: Optional[str] = None,
        difficulty: Difficulty = Difficulty.EXPERT,
        num_search_iters: int = 200,
        seed: int = 42,
    ) -> None:
        super().__init__(name)

        if blueprint is not None:
            self.blueprint = blueprint
        elif blueprint_path is not None:
            self.blueprint = Strategy.from_file(blueprint_path)
        else:
            self.blueprint = Strategy()  # Empty strategy = uniform random

        self.difficulty = difficulty
        self.use_search = difficulty in (Difficulty.EXPERT, Difficulty.SUPERHUMAN)
        self.noise = DIFFICULTY_NOISE.get(difficulty, 0.0)
        self.num_search_iters = num_search_iters
        self.rng = random.Random(seed)
        self.seed = seed

        self._beliefs: Optional[BeliefState] = None
        self._search: Optional[RealTimeSearch] = None
        self._current_round = -1
        self._action_history: list = []

    def notify_hand_start(self, state: VisibleGameState) -> None:
        self._beliefs = BeliefState(state.num_players)
        known = list(state.my_cards)
        self._beliefs.remove_known_cards([c.int_value for c in known])
        self._current_round = -1
        self._action_history = []
        self._search = None

    def notify_action(self, seat: int, action: Action) -> None:
        self._action_history.append((seat, action))

    def get_action(self, state: VisibleGameState) -> Action:
        """Decide on an action using blueprint or search."""
        actions = self._get_legal_actions(state)
        if not actions:
            return Action.call()

        if len(actions) == 1:
            return actions[0]

        # Apply noise: with some probability, pick a random legal action
        if self.noise > 0 and self.rng.random() < self.noise:
            return self.rng.choice(actions)

        # Preflop: use blueprint directly
        if state.betting_round == BettingRound.PREFLOP and not self._needs_search(state):
            return self._blueprint_action(state, actions)

        # Postflop or off-tree: use real-time search (Superhuman only)
        if self.use_search:
            return self._search_action(state, actions)

        # Fallback to blueprint
        return self._blueprint_action(state, actions)

    def _needs_search(self, state: VisibleGameState) -> bool:
        """Check if search is needed (e.g., off-tree opponent bet)."""
        return False

    def _blueprint_action(
        self, state: VisibleGameState, actions: list[Action]
    ) -> Action:
        """Select action from blueprint strategy."""
        key = self._make_key(state)
        probs = self.blueprint.get_or_uniform(key, len(actions))

        # Sample action
        r = self.rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return actions[i]
        return actions[-1]

    def _search_action(
        self, state: VisibleGameState, actions: list[Action]
    ) -> Action:
        """Use real-time search to find a better action."""
        if self._search is None or state.betting_round != self._current_round:
            self._search = RealTimeSearch(
                blueprint=self.blueprint,
                num_players=state.num_players,
                num_search_iters=self.num_search_iters,
                seed=self.seed + state.betting_round,
            )
            self._current_round = state.betting_round

        subgame = SubgameState(
            num_players=state.num_players,
            pot=state.pot,
            betting_round=state.betting_round,
            community_cards=[c.int_value for c in state.community_cards],
            player_stacks=list(state.player_stacks),
            player_bets=list(state.player_bets),
            player_folded=list(state.player_folded),
            player_all_in=list(state.player_all_in),
            current_player=state.my_seat,
            hole_cards={
                state.my_seat: (
                    state.my_cards[0].int_value,
                    state.my_cards[1].int_value,
                )
            },
            action_history=[],
        )

        if self._beliefs is None:
            self._beliefs = BeliefState(state.num_players)
        known_cards = [c.int_value for c in state.my_cards + state.community_cards]
        self._beliefs.remove_known_cards(known_cards)

        probs = self._search.search(subgame, state.my_seat, self._beliefs)

        if len(probs) != len(actions):
            probs = [1.0 / len(actions)] * len(actions)

        r = self.rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return actions[i]
        return actions[-1]

    def _make_key(self, state: VisibleGameState) -> int:
        """Make an infoset key from visible state."""
        action_seq = []
        for seat, action in self._action_history:
            action_seq.append(action.type.value * 10000 + action.amount)

        return make_infoset_key(
            state.my_seat,
            state.betting_round,
            (state.my_cards[0].int_value, state.my_cards[1].int_value),
            tuple(c.int_value for c in state.community_cards),
            len(state.community_cards),
            tuple(action_seq),
            len(action_seq),
        )

    def _get_legal_actions(self, state: VisibleGameState) -> list[Action]:
        """Compute legal actions from visible state."""
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
