"""Game runner for a single hand of Texas Hold'em."""

from __future__ import annotations

from typing import List, Optional, Tuple

from gambletron.players.base import Player
from gambletron.poker.card import Card, Deck
from gambletron.poker.hand import describe_hand, evaluate_hand
from gambletron.poker.rules import apply_action, get_legal_actions, is_legal_action
from gambletron.poker.state import (
    Action,
    ActionType,
    BettingRound,
    GameState,
    PlayerState,
)
from gambletron.display.sink import NullDisplaySink


class Game:
    """Runs a single hand of no-limit Texas Hold'em."""

    def __init__(
        self,
        players: List[Player],
        stacks: List[int],
        dealer_pos: int = 0,
        small_blind: int = 50,
        big_blind: int = 100,
        deck: Optional[Deck] = None,
        display_sink=None,
    ) -> None:
        if not 2 <= len(players) <= 6:
            raise ValueError(f"Need 2-6 players, got {len(players)}")
        if len(stacks) != len(players):
            raise ValueError("Stacks list must match players list")

        self.players = players
        self.deck = deck or Deck()
        self.display = display_sink if display_sink is not None else NullDisplaySink()
        self.state = GameState(
            num_players=len(players),
            dealer_pos=dealer_pos,
            small_blind=small_blind,
            big_blind=big_blind,
        )
        # Initialize player states
        for i, stack in enumerate(stacks):
            self.state.players.append(PlayerState(seat=i, stack=stack))

    def play_hand(self) -> List[int]:
        """Play a complete hand. Returns chip changes for each player."""
        initial_stacks = [p.stack for p in self.state.players]

        # Auto-fold busted players (0 stack)
        for p in self.state.players:
            if p.stack == 0:
                p.is_folded = True

        self.deck.shuffle()
        self._post_blinds()
        self._deal_hole_cards()

        # Notify players of hand start
        for i, player in enumerate(self.players):
            player.notify_hand_start(self.state.visible_to(i))

        # Tell display: initial preflop state
        self.display.preflop(
            pot=self.state.pot,
            current_player=self.state.current_player,
            dealer_pos=self.state.dealer_pos,
        )

        # Main game loop
        while not self.state.is_hand_over:
            if self.state.betting_round == BettingRound.FLOP and len(self.state.community_cards) == 0:
                self._deal_community(3)
            elif self.state.betting_round == BettingRound.TURN and len(self.state.community_cards) == 3:
                self._deal_community(1)
            elif self.state.betting_round == BettingRound.RIVER and len(self.state.community_cards) == 4:
                self._deal_community(1)

            if self.state.is_hand_over:
                break

            self._play_betting_round()

        # Deal remaining community cards if needed for showdown
        while len(self.state.community_cards) < 5 and self.state.num_in_hand > 1:
            needed = 5 - len(self.state.community_cards)
            self._deal_community(needed)

        # Determine winner(s) and award pot
        self._showdown()

        # Notify players of hand end
        for i, player in enumerate(self.players):
            player.notify_hand_end(self.state.visible_to(i))

        return [p.stack - initial_stacks[i] for i, p in enumerate(self.state.players)]

    def _post_blinds(self) -> None:
        n = self.state.num_players
        if n == 2:
            sb_pos = self.state.dealer_pos
            bb_pos = (self.state.dealer_pos + 1) % n
        else:
            sb_pos = (self.state.dealer_pos + 1) % n
            bb_pos = (self.state.dealer_pos + 2) % n

        sb = self.state.players[sb_pos]
        bb = self.state.players[bb_pos]

        sb_amount = min(self.state.small_blind, sb.stack)
        sb.stack -= sb_amount
        sb.bet_this_round = sb_amount
        sb.bet_total = sb_amount
        self.state.pot += sb_amount
        if sb.stack == 0:
            sb.is_all_in = True

        bb_amount = min(self.state.big_blind, bb.stack)
        bb.stack -= bb_amount
        bb.bet_this_round = bb_amount
        bb.bet_total = bb_amount
        self.state.pot += bb_amount
        if bb.stack == 0:
            bb.is_all_in = True

        # First to act preflop is player after big blind
        if n == 2:
            self.state.current_player = sb_pos  # Dealer/SB acts first preflop in heads-up
        else:
            first = (bb_pos + 1) % n
            while not self.state.players[first].is_active:
                first = (first + 1) % n
            self.state.current_player = first

        self.state.last_raiser = bb_pos  # BB is considered the "raiser" for round-end logic

    def _deal_hole_cards(self) -> None:
        for p in self.state.players:
            p.hole_cards = self.deck.deal(2)

    def _deal_community(self, n: int) -> None:
        cards = self.deck.deal(n)
        self.state.community_cards.extend(cards)
        # Notify display of the new community cards
        next_cp = self.state.current_player if self.state.current_player >= 0 else None
        self.display.community_cards(
            betting_round=self.state.betting_round.name,
            cards=[c.int_value for c in self.state.community_cards],
            pot=self.state.pot,
            current_player=next_cp,
        )

    def _play_betting_round(self) -> None:
        """Run the current betting round until complete."""
        current_round = self.state.betting_round
        while not self.state.is_hand_over:
            # Stop if _end_betting_round has transitioned to the next round;
            # the outer play_hand loop will deal community cards first.
            if self.state.betting_round != current_round:
                break

            cp = self.state.current_player
            if cp < 0:
                break

            player = self.players[cp]
            pstate = self.state.players[cp]

            if not pstate.is_active:
                break

            visible = self.state.visible_to(cp)
            action = player.get_action(visible)

            # Validate action; if invalid, default to call (or fold if can't call)
            if not is_legal_action(self.state, action):
                legal = get_legal_actions(self.state)
                # Default to call if available, else fold
                action = next(
                    (a for a in legal if a.type == ActionType.CALL),
                    next((a for a in legal if a.type == ActionType.FOLD), legal[0]),
                )

            # Capture pre-action state before apply_action mutates it
            to_call       = self.state.current_bet - pstate.bet_this_round
            round_name    = self.state.betting_round.name
            stack_before  = pstate.stack

            # Notify all players
            for i, p in enumerate(self.players):
                p.notify_action(cp, action)

            apply_action(self.state, action)

            # Push action event to display (use pre-action round name)
            next_cp = self.state.current_player if self.state.current_player >= 0 else None
            self.display.action(
                seat=cp,
                description=_describe_action(action, to_call, stack_before),
                pot=self.state.pot,
                current_player=next_cp,
                player_folded=[p.is_folded for p in self.state.players],
                betting_round=round_name,
            )

    def _showdown(self) -> None:
        """Determine winner(s) and distribute the pot."""
        in_hand = self.state.players_in_hand

        if len(in_hand) == 1:
            pot = self.state.pot
            in_hand[0].stack += pot
            self.state.pot = 0
            self.display.winner(
                seats=[in_hand[0].seat],
                pot_won=pot,
                hand_desc={},
            )
            return

        # Evaluate hands
        hand_scores = {}
        for p in in_hand:
            all_cards = p.hole_cards + self.state.community_cards
            if len(all_cards) >= 5:
                hand_scores[p.seat] = evaluate_hand(all_cards)

        if not hand_scores:
            return

        # Push showdown event: reveal hole cards
        self.display.showdown(
            hole_cards={p.seat: [c.int_value for c in p.hole_cards] for p in in_hand},
            community_cards=[c.int_value for c in self.state.community_cards],
            pot=self.state.pot,
        )

        # Determine winner(s) for display (best hand among those evaluated)
        best = max(hand_scores.values())
        winner_seats = [s for s, sc in hand_scores.items() if sc == best]
        hand_desc = {s: describe_hand(hand_scores[s]) for s in winner_seats}
        pot_before = self.state.pot

        # Handle side pots
        _distribute_pot(self.state, hand_scores)

        self.display.winner(
            seats=winner_seats,
            pot_won=pot_before,
            hand_desc=hand_desc,
        )


def _describe_action(action: Action, to_call: int, stack_before: int) -> str:
    """Return a short display string for an action, e.g. 'raises to $300'."""
    if action.type == ActionType.FOLD:
        return "folds"
    if action.type == ActionType.CALL:
        if to_call == 0:
            return "checks"
        actual = min(to_call, stack_before)
        if stack_before <= to_call:
            return f"calls all-in  (${actual:,})"
        return f"calls  ${actual:,}"
    if action.type == ActionType.RAISE:
        return f"raises to  ${action.amount:,}"
    return str(action)


def _distribute_pot(state: GameState, hand_scores: dict) -> None:
    """Distribute pot including side pots."""
    players_in = [p for p in state.players if p.is_in_hand]

    # Sort by total bet amount to handle side pots
    sorted_players = sorted(players_in, key=lambda p: p.bet_total)

    remaining_players = list(players_in)
    prev_bet = 0

    for i, p in enumerate(sorted_players):
        if p.bet_total <= prev_bet:
            continue

        # This player's contribution level
        level = p.bet_total
        side_pot = 0

        for other in state.players:
            contribution = min(other.bet_total, level) - min(other.bet_total, prev_bet)
            side_pot += contribution

        if side_pot <= 0:
            continue

        # Find winner(s) among remaining players eligible for this side pot
        eligible = [rp for rp in remaining_players if rp.seat in hand_scores]
        if not eligible:
            continue

        best_score = max(hand_scores[rp.seat] for rp in eligible)
        winners = [rp for rp in eligible if hand_scores[rp.seat] == best_score]

        share = side_pot // len(winners)
        remainder = side_pot % len(winners)
        for j, w in enumerate(winners):
            w.stack += share + (1 if j < remainder else 0)

        state.pot -= side_pot
        prev_bet = level

        # Remove players who are maxed out at this level
        remaining_players = [rp for rp in remaining_players if rp.bet_total > level]

    # Any remaining pot (shouldn't happen normally)
    if state.pot > 0 and players_in:
        best_score = max(hand_scores.get(p.seat, ()) for p in players_in)
        winners = [p for p in players_in if hand_scores.get(p.seat) == best_score]
        if winners:
            share = state.pot // len(winners)
            remainder = state.pot % len(winners)
            for j, w in enumerate(winners):
                w.stack += share + (1 if j < remainder else 0)
            state.pot = 0
