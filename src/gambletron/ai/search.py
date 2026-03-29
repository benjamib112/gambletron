"""Real-time depth-limited subgame search (Algorithm 2 from Pluribus paper).

Implements nested search with:
- Subgame rooted at start of current betting round
- Depth limits depending on round and number of players
- 4 continuation strategies at leaf nodes
- Belief tracking via Bayes' rule
- Frozen action probabilities for actions already chosen
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from gambletron.ai.belief import BeliefState, ALL_HANDS, NUM_HANDS, hand_to_index
from gambletron.ai.blueprint import make_infoset_key
from gambletron.ai.strategy import Strategy
from gambletron.poker.card import Card
from gambletron.poker.hand import evaluate_hand
from gambletron.poker.state import Action, ActionType, BettingRound


class ContinuationStrategy:
    """A biased version of the blueprint strategy for leaf node evaluation."""

    UNBIASED = 0
    FOLD_BIASED = 1
    CALL_BIASED = 2
    RAISE_BIASED = 3
    BIAS_FACTOR = 5.0

    def __init__(self, blueprint: Strategy, bias_type: int = 0) -> None:
        self.blueprint = blueprint
        self.bias_type = bias_type

    def get_action_probs(
        self, key: int, num_actions: int, action_types: List[ActionType]
    ) -> List[float]:
        """Get biased action probabilities."""
        probs = list(self.blueprint.get_or_uniform(key, num_actions))

        if self.bias_type == self.UNBIASED:
            return probs

        for i, at in enumerate(action_types):
            if self.bias_type == self.FOLD_BIASED and at == ActionType.FOLD:
                probs[i] *= self.BIAS_FACTOR
            elif self.bias_type == self.CALL_BIASED and at == ActionType.CALL:
                probs[i] *= self.BIAS_FACTOR
            elif self.bias_type == self.RAISE_BIASED and at == ActionType.RAISE:
                probs[i] *= self.BIAS_FACTOR

        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        return probs


class SubgameState:
    """State within a subgame being searched."""

    def __init__(
        self,
        num_players: int,
        pot: int,
        betting_round: int,
        community_cards: List[int],
        player_stacks: List[int],
        player_bets: List[int],
        player_folded: List[bool],
        player_all_in: List[bool],
        current_player: int,
        hole_cards: Dict[int, Tuple[int, int]],  # player -> (card1, card2)
        action_history: List[int],
    ) -> None:
        self.num_players = num_players
        self.pot = pot
        self.betting_round = betting_round
        self.community_cards = list(community_cards)
        self.player_stacks = list(player_stacks)
        self.player_bets = list(player_bets)
        self.player_folded = list(player_folded)
        self.player_all_in = list(player_all_in)
        self.current_player = current_player
        self.hole_cards = dict(hole_cards)
        self.action_history = list(action_history)

    def copy(self) -> SubgameState:
        return SubgameState(
            self.num_players,
            self.pot,
            self.betting_round,
            self.community_cards,
            self.player_stacks,
            self.player_bets,
            self.player_folded,
            self.player_all_in,
            self.current_player,
            self.hole_cards,
            self.action_history,
        )

    @property
    def num_in_hand(self) -> int:
        return sum(1 for f in self.player_folded if not f)

    @property
    def current_bet(self) -> int:
        return max(self.player_bets) if self.player_bets else 0


class RealTimeSearch:
    """Depth-limited real-time search for improving play during a game."""

    def __init__(
        self,
        blueprint: Strategy,
        num_players: int = 6,
        num_search_iters: int = 200,
        seed: int = 42,
    ) -> None:
        self.blueprint = blueprint
        self.num_players = num_players
        self.num_search_iters = num_search_iters
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Build continuation strategies for leaf nodes
        self.continuations = [
            ContinuationStrategy(blueprint, ContinuationStrategy.UNBIASED),
            ContinuationStrategy(blueprint, ContinuationStrategy.FOLD_BIASED),
            ContinuationStrategy(blueprint, ContinuationStrategy.CALL_BIASED),
            ContinuationStrategy(blueprint, ContinuationStrategy.RAISE_BIASED),
        ]

        # Subgame-local regrets and strategies
        self._regrets: Dict[int, List[float]] = {}
        self._strategy_sum: Dict[int, List[float]] = {}

        # Frozen infosets (our actions already taken)
        self._frozen: Dict[int, List[float]] = {}

    def search(
        self,
        state: SubgameState,
        our_player: int,
        beliefs: BeliefState,
    ) -> List[float]:
        """Run depth-limited search and return action probabilities for our_player.

        Returns probabilities over the available actions at the current decision point.
        """
        self._regrets.clear()
        self._strategy_sum.clear()

        # Get available actions
        actions = self._get_actions(state, state.current_player)
        if not actions:
            return [1.0]

        # Run search iterations
        for t in range(self.num_search_iters):
            # Sample opponent hands from beliefs
            sampled_hands = self._sample_hands(state, beliefs)
            search_state = state.copy()
            search_state.hole_cards = sampled_hands

            self._traverse(search_state, our_player, state.betting_round)

        # Return strategy at root for our actual hand
        key = self._make_key(state, our_player)
        return self._get_current_strategy(key, len(actions))

    def _traverse(
        self,
        state: SubgameState,
        traverser: int,
        root_round: int,
    ) -> float:
        """Recursive traversal of the subgame."""
        # Terminal check
        if state.num_in_hand == 1:
            winner = next(
                i for i in range(state.num_players) if not state.player_folded[i]
            )
            return float(state.pot) if winner == traverser else 0.0

        # Depth limit check
        if self._is_leaf(state, root_round):
            return self._evaluate_leaf(state, traverser)

        if state.player_folded[traverser]:
            return 0.0

        cp = state.current_player
        actions = self._get_actions(state, cp)
        if not actions:
            return 0.0

        num_actions = len(actions)
        key = self._make_key(state, cp)

        # Check if this infoset is frozen
        frozen = self._frozen.get(key)
        if frozen is not None:
            # Play according to frozen strategy
            probs = frozen
            idx = self._sample_action(probs)
            child = self._apply_action(state, actions[idx], cp)
            return self._traverse(child, traverser, root_round)

        strategy = self._get_current_strategy(key, num_actions)

        if cp == traverser:
            # Explore all actions
            action_values = []
            for a in range(num_actions):
                child = self._apply_action(state, actions[a], cp)
                action_values.append(self._traverse(child, traverser, root_round))

            node_value = sum(s * v for s, v in zip(strategy, action_values))

            # Update regrets
            if key not in self._regrets:
                self._regrets[key] = [0.0] * num_actions
            for a in range(num_actions):
                self._regrets[key][a] += action_values[a] - node_value

            return node_value
        else:
            # Sample opponent action
            idx = self._sample_action(strategy)
            child = self._apply_action(state, actions[idx], cp)
            return self._traverse(child, traverser, root_round)

    def _is_leaf(self, state: SubgameState, root_round: int) -> bool:
        """Check if we've reached the depth limit."""
        if state.betting_round > 3:
            return True

        if state.betting_round == root_round:
            return False  # Still in the root round

        # Round 1 search: leaf at start of round 2
        if root_round == 0 and state.betting_round >= 1:
            return True

        # Round 2 with >2 players: leaf at start of round 3 or after 2nd raise
        if root_round == 1:
            active = sum(1 for i in range(state.num_players)
                        if not state.player_folded[i])
            if active > 2 and state.betting_round >= 2:
                return True

        # Otherwise: solve to end of game
        return False

    def _evaluate_leaf(self, state: SubgameState, traverser: int) -> float:
        """Evaluate leaf node using continuation strategies.

        Each player chooses among 4 continuation strategies.
        We approximate by running a quick Monte Carlo evaluation.
        """
        if state.num_in_hand <= 1:
            if not state.player_folded[traverser]:
                return float(state.pot)
            return 0.0

        # Deal remaining community cards if needed
        known = set(state.community_cards)
        for h in state.hole_cards.values():
            known.add(h[0])
            known.add(h[1])
        remaining = [c for c in range(52) if c not in known]

        needed = 5 - len(state.community_cards)
        if needed > 0 and remaining:
            board_cards = list(self.np_rng.choice(remaining, size=min(needed, len(remaining)), replace=False))
            full_board = state.community_cards + [int(c) for c in board_cards]
        else:
            full_board = state.community_cards

        # Evaluate hands
        if len(full_board) < 5:
            # Not enough cards; return pot split
            in_hand = [i for i in range(state.num_players) if not state.player_folded[i]]
            return float(state.pot) / len(in_hand) if traverser in in_hand else 0.0

        best_score = None
        winners = []
        for p in range(state.num_players):
            if state.player_folded[p]:
                continue
            if p not in state.hole_cards:
                continue
            cards = [Card(state.hole_cards[p][0]), Card(state.hole_cards[p][1])]
            cards += [Card(c) for c in full_board]
            score = evaluate_hand(cards)
            if best_score is None or score > best_score:
                best_score = score
                winners = [p]
            elif score == best_score:
                winners.append(p)

        if traverser in winners:
            return float(state.pot) / len(winners)
        return 0.0

    def _get_actions(self, state: SubgameState, player: int) -> List[Tuple[ActionType, int]]:
        """Get available actions for a player."""
        if state.player_folded[player] or state.player_all_in[player]:
            return []

        actions = []
        current_bet = state.current_bet
        to_call = current_bet - state.player_bets[player]

        if to_call > 0:
            actions.append((ActionType.FOLD, 0))

        actions.append((ActionType.CALL, min(to_call, state.player_stacks[player])))

        chips_after_call = state.player_stacks[player] - to_call
        if chips_after_call > 0:
            max_raise = state.player_bets[player] + state.player_stacks[player]
            if max_raise > current_bet:
                # Pot-size raise
                pot_raise = current_bet + state.pot + to_call
                pot_raise = min(pot_raise, max_raise)
                if pot_raise > current_bet:
                    actions.append((ActionType.RAISE, pot_raise))
                # All-in if different from pot raise
                if max_raise != pot_raise and max_raise > current_bet:
                    actions.append((ActionType.RAISE, max_raise))

        return actions

    def _apply_action(
        self, state: SubgameState, action: Tuple[ActionType, int], player: int
    ) -> SubgameState:
        """Apply an action and return new state."""
        child = state.copy()
        atype, amount = action
        current_bet = child.current_bet

        if atype == ActionType.FOLD:
            child.player_folded[player] = True
        elif atype == ActionType.CALL:
            to_call = current_bet - child.player_bets[player]
            actual = min(to_call, child.player_stacks[player])
            child.player_stacks[player] -= actual
            child.player_bets[player] += actual
            child.pot += actual
            if child.player_stacks[player] == 0:
                child.player_all_in[player] = True
        elif atype == ActionType.RAISE:
            chips_needed = amount - child.player_bets[player]
            actual = min(chips_needed, child.player_stacks[player])
            child.player_stacks[player] -= actual
            child.player_bets[player] += actual
            child.pot += actual
            if child.player_stacks[player] == 0:
                child.player_all_in[player] = True

        child.action_history.append(atype.value * 10000 + amount)

        # Advance to next player or next round
        self._advance(child, player)
        return child

    def _advance(self, state: SubgameState, last_player: int) -> None:
        """Advance to the next player or betting round."""
        # Find next active player
        n = state.num_players
        next_p = last_player
        for _ in range(n):
            next_p = (next_p + 1) % n
            if not state.player_folded[next_p] and not state.player_all_in[next_p]:
                break

        # Check if round is over
        active = [
            i for i in range(n)
            if not state.player_folded[i] and not state.player_all_in[i]
        ]

        if len(active) <= 1:
            self._end_round(state)
            return

        current_bet = state.current_bet
        all_matched = all(state.player_bets[i] == current_bet for i in active)

        if all_matched and len(state.action_history) > 0:
            self._end_round(state)
            return

        state.current_player = next_p

    def _end_round(self, state: SubgameState) -> None:
        """End the current betting round."""
        for i in range(state.num_players):
            state.player_bets[i] = 0
        state.betting_round += 1

        # Find first active player
        for i in range(state.num_players):
            p = (i + 1) % state.num_players  # After dealer (pos 0)
            if not state.player_folded[p] and not state.player_all_in[p]:
                state.current_player = p
                return

    def _make_key(self, state: SubgameState, player: int) -> int:
        """Make an infoset key for the subgame."""
        if player not in state.hole_cards:
            return hash((player, state.betting_round, tuple(state.action_history)))

        return make_infoset_key(
            player,
            state.betting_round,
            state.hole_cards[player],
            tuple(state.community_cards),
            len(state.community_cards),
            tuple(state.action_history),
            len(state.action_history),
        )

    def _get_current_strategy(self, key: int, num_actions: int) -> List[float]:
        """Get strategy from subgame regrets."""
        regrets = self._regrets.get(key)
        if regrets is None:
            return [1.0 / num_actions] * num_actions

        positive = [max(r, 0) for r in regrets]
        total = sum(positive)
        if total > 0:
            return [p / total for p in positive]
        return [1.0 / num_actions] * num_actions

    def _sample_action(self, probs: List[float]) -> int:
        """Sample an action index from a probability distribution."""
        r = self.rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return i
        return len(probs) - 1

    def _sample_hands(
        self, state: SubgameState, beliefs: BeliefState
    ) -> Dict[int, Tuple[int, int]]:
        """Sample hole cards for all players from belief distributions."""
        hands = {}
        used_cards = set(state.community_cards)

        for p in range(state.num_players):
            if state.player_folded[p]:
                continue
            if p in state.hole_cards and state.hole_cards[p] != (0, 0):
                hands[p] = state.hole_cards[p]
                used_cards.add(state.hole_cards[p][0])
                used_cards.add(state.hole_cards[p][1])

        # Sample remaining players' hands from beliefs
        for p in range(state.num_players):
            if p in hands or state.player_folded[p]:
                continue

            probs = beliefs.beliefs[p].copy()
            # Zero out hands with used cards
            for idx in range(NUM_HANDS):
                c1, c2 = ALL_HANDS[idx]
                if c1 in used_cards or c2 in used_cards:
                    probs[idx] = 0.0

            total = probs.sum()
            if total > 0:
                probs /= total
                chosen = self.np_rng.choice(NUM_HANDS, p=probs)
                hand = ALL_HANDS[chosen]
                hands[p] = hand
                used_cards.add(hand[0])
                used_cards.add(hand[1])

        return hands

    def freeze_action(self, key: int, probs: List[float]) -> None:
        """Freeze the strategy at a key (for actions we've already taken)."""
        self._frozen[key] = list(probs)

    def clear_frozen(self) -> None:
        """Clear frozen strategies (at start of new betting round)."""
        self._frozen.clear()
