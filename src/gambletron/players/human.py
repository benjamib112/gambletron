"""Human player: CLI-based input for the digital demo."""

from __future__ import annotations

from gambletron.players.base import Player
from gambletron.poker.state import Action, ActionType, BettingRound, VisibleGameState


_ROUND_NAMES = {
    BettingRound.PREFLOP: "Preflop",
    BettingRound.FLOP: "Flop",
    BettingRound.TURN: "Turn",
    BettingRound.RIVER: "River",
}


class HumanPlayer(Player):
    """Interactive CLI player."""

    def get_action(self, state: VisibleGameState) -> Action:
        self._display_state(state)
        return self._prompt_action(state)

    def _display_state(self, state: VisibleGameState) -> None:
        print()
        print(f"=== {_ROUND_NAMES.get(state.betting_round, '?')} ===")
        print(f"Your cards: {' '.join(str(c) for c in state.my_cards)}")
        if state.community_cards:
            print(f"Board: {' '.join(str(c) for c in state.community_cards)}")
        print(f"Pot: ${state.pot}")
        print(f"Your stack: ${state.my_stack}")

        # Show other players
        for i in range(state.num_players):
            if i == state.my_seat:
                continue
            status = ""
            if state.player_folded[i]:
                status = " (folded)"
            elif state.player_all_in[i]:
                status = " (all-in)"
            print(
                f"  Seat {i}: ${state.player_stacks[i]} "
                f"(bet ${state.player_bets[i]}){status}"
            )

        current_bet = max(state.player_bets)
        to_call = current_bet - state.player_bets[state.my_seat]
        if to_call > 0:
            print(f"To call: ${to_call}")

    def _prompt_action(self, state: VisibleGameState) -> Action:
        current_bet = max(state.player_bets)
        my_bet = state.player_bets[state.my_seat]
        to_call = current_bet - my_bet
        my_stack = state.my_stack

        while True:
            options = {}

            if to_call > 0:
                options["f"] = Action.fold()
                print(f"  [f] Fold")

            if to_call == 0:
                options["c"] = Action.call()
                print(f"  [c] Check")
            else:
                call_amount = min(to_call, my_stack)
                options["c"] = Action.call()
                print(f"  [c] Call ${call_amount}")

            chips_after_call = my_stack - to_call
            if chips_after_call > 0:
                min_raise_to = current_bet + state.min_raise
                max_raise_to = my_bet + my_stack
                if max_raise_to > current_bet:
                    if min_raise_to > max_raise_to:
                        min_raise_to = max_raise_to

                    if state.betting_round == BettingRound.PREFLOP:
                        # Standard open (2.5x BB)
                        if 250 >= min_raise_to and 250 < max_raise_to:
                            options["r"] = Action.raise_to(250)
                            print(f"  [r] Raise to $250")
                        # Pot-sized raise
                        pot_raise = state.pot + 2 * to_call + current_bet
                        if (pot_raise >= min_raise_to and pot_raise < max_raise_to
                                and pot_raise != 250):
                            options["p"] = Action.raise_to(pot_raise)
                            print(f"  [p] Pot raise to ${pot_raise}")
                        # All-in
                        if max_raise_to > min_raise_to:
                            options["a"] = Action.raise_to(max_raise_to)
                            print(f"  [a] All-in (${max_raise_to})")
                    else:
                        # Postflop: pot-sized raise
                        pot_raise = current_bet + state.pot + to_call
                        if pot_raise >= min_raise_to and pot_raise <= max_raise_to:
                            options["r"] = Action.raise_to(pot_raise)
                            print(f"  [r] Pot raise to ${pot_raise}")
                        elif max_raise_to >= min_raise_to:
                            options["a"] = Action.raise_to(max_raise_to)
                            print(f"  [a] All-in (${max_raise_to})")

            choice = input("Your action: ").strip().lower()

            if choice in options:
                return options[choice]
            else:
                print("Invalid choice. Try again.")

    def notify_hand_start(self, state: VisibleGameState) -> None:
        print(f"\n{'='*50}")
        print(f"New hand! You are seat {state.my_seat} (dealer at seat {state.dealer_pos})")

    def notify_hand_end(self, state: VisibleGameState) -> None:
        print(f"\nHand over. Your stack: ${state.my_stack}")

    def notify_action(self, seat: int, action: Action) -> None:
        print(f"  Seat {seat}: {action}")
