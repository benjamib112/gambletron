"""Texas Hold'em rules: legal actions, betting logic, raise validation."""

from __future__ import annotations

from typing import List

from gambletron.poker.state import Action, ActionType, GameState


def get_legal_actions(state: GameState) -> List[Action]:
    """Return all legal actions for the current player."""
    player = state.players[state.current_player]
    if not player.is_active:
        return []

    actions = []
    current_bet = state.current_bet
    to_call = current_bet - player.bet_this_round

    # Fold is always legal if there's a bet to call
    if to_call > 0:
        actions.append(Action.fold())

    # Call/Check
    if to_call == 0:
        actions.append(Action.call())  # Check
    elif to_call >= player.stack:
        # All-in call
        actions.append(Action.call())
    else:
        actions.append(Action.call())

    # Raise (only if player has chips beyond calling)
    chips_after_call = player.stack - to_call
    if chips_after_call > 0:
        min_raise_to = current_bet + state.min_raise
        # If min raise would be all-in or more, the only raise is all-in
        max_raise_to = player.bet_this_round + player.stack
        if max_raise_to > current_bet:
            actual_min = min(min_raise_to, max_raise_to)
            # Player can raise to any amount between min and max
            actions.append(Action.raise_to(actual_min))  # Minimum raise
            if max_raise_to > actual_min:
                actions.append(Action.raise_to(max_raise_to))  # All-in

    return actions


def is_legal_action(state: GameState, action: Action) -> bool:
    """Check if a specific action is legal for the current player."""
    player = state.players[state.current_player]
    if not player.is_active:
        return False

    current_bet = state.current_bet
    to_call = current_bet - player.bet_this_round

    if action.type == ActionType.FOLD:
        return to_call > 0

    if action.type == ActionType.CALL:
        return True

    if action.type == ActionType.RAISE:
        chips_after_call = player.stack - to_call
        if chips_after_call <= 0:
            return False  # Can't raise if calling uses all chips

        max_raise_to = player.bet_this_round + player.stack
        if action.amount > max_raise_to:
            return False  # Can't bet more than stack

        min_raise_to = current_bet + state.min_raise
        # Allow all-in even if below min raise
        if action.amount == max_raise_to:
            return action.amount > current_bet
        # Otherwise must meet minimum raise
        return action.amount >= min_raise_to

    return False


def apply_action(state: GameState, action: Action) -> None:
    """Apply an action to the game state. Mutates state in place."""
    player = state.players[state.current_player]
    current_bet = state.current_bet

    state.action_history[state.betting_round].append(
        (state.current_player, action)
    )
    state.num_actions_this_round += 1

    if action.type == ActionType.FOLD:
        player.is_folded = True

    elif action.type == ActionType.CALL:
        to_call = current_bet - player.bet_this_round
        actual_call = min(to_call, player.stack)
        player.stack -= actual_call
        player.bet_this_round += actual_call
        player.bet_total += actual_call
        state.pot += actual_call
        if player.stack == 0:
            player.is_all_in = True

    elif action.type == ActionType.RAISE:
        raise_to = action.amount
        chips_needed = raise_to - player.bet_this_round
        actual_chips = min(chips_needed, player.stack)
        raise_increment = raise_to - current_bet
        if raise_increment > state.min_raise:
            state.min_raise = raise_increment
        player.stack -= actual_chips
        player.bet_this_round += actual_chips
        player.bet_total += actual_chips
        state.pot += actual_chips
        state.last_raiser = state.current_player
        if player.stack == 0:
            player.is_all_in = True

    # Check if hand is over (only one player left)
    if state.num_in_hand == 1:
        state.is_hand_over = True
        return

    # Advance to next active player
    _advance_player(state)

    # Check if betting round is over
    if _is_round_over(state):
        _end_betting_round(state)


def _advance_player(state: GameState) -> None:
    """Move to the next player who can act."""
    n = state.num_players
    seat = state.current_player
    for _ in range(n):
        seat = (seat + 1) % n
        if state.players[seat].is_active:
            state.current_player = seat
            return
    # No active players (everyone folded or all-in)
    state.current_player = -1


def _is_round_over(state: GameState) -> bool:
    """Check if the current betting round is complete."""
    active = state.active_players
    if not active:
        return True

    # All active players must have matching bets
    current_bet = state.current_bet
    for p in active:
        if p.bet_this_round != current_bet:
            return False

    # All active players must have acted at least once this round.
    # This correctly handles the BB option preflop: even though BB's blind
    # is tracked as last_raiser, BB still needs to act (check or raise) before
    # the round can end.
    acted = {seat for seat, _ in state.action_history[state.betting_round]}
    for p in active:
        if p.seat not in acted:
            return False

    return True


def _end_betting_round(state: GameState) -> None:
    """Transition to the next betting round or end the hand."""
    # Reset per-round state
    for p in state.players:
        p.bet_this_round = 0
    state.last_raiser = None
    state.num_actions_this_round = 0

    if state.betting_round == BettingRound.RIVER:
        state.is_hand_over = True
        return

    # Check if only one player can still act (rest are all-in or folded)
    active = state.active_players
    if len(active) <= 1:
        # Run out remaining community cards but no more betting
        state.is_hand_over = True
        return

    state.betting_round = BettingRound(state.betting_round + 1)
    state.min_raise = state.big_blind

    # First player to act postflop is first active player after dealer
    n = state.num_players
    seat = state.dealer_pos
    for _ in range(n):
        seat = (seat + 1) % n
        if state.players[seat].is_active:
            state.current_player = seat
            return


# Import here to avoid circular dependency at module level
from gambletron.poker.state import BettingRound  # noqa: E402
