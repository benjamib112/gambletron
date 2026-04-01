"""Tests for the game runner and rules."""

from gambletron.players.base import Player
from gambletron.players.random_player import RandomPlayer
from gambletron.poker.card import Deck
from gambletron.poker.game import Game
from gambletron.poker.state import Action, ActionType, BettingRound, GameState, PlayerState, VisibleGameState
from gambletron.poker.rules import apply_action, get_legal_actions, is_legal_action
from gambletron.poker.table import Table


class FoldPlayer(Player):
    """Always folds (or checks if no bet to call)."""

    def get_action(self, state: VisibleGameState) -> Action:
        to_call = max(state.player_bets) - state.player_bets[state.my_seat]
        if to_call > 0:
            return Action.fold()
        return Action.call()


class CallPlayer(Player):
    """Always calls."""

    def get_action(self, state: VisibleGameState) -> Action:
        return Action.call()


def test_game_runs_to_completion():
    players = [RandomPlayer(f"P{i}", seed=i) for i in range(6)]
    game = Game(players, stacks=[10000] * 6, dealer_pos=0)
    changes = game.play_hand()
    assert len(changes) == 6
    assert sum(changes) == 0  # Zero-sum


def test_game_heads_up():
    players = [RandomPlayer("P0", seed=0), RandomPlayer("P1", seed=1)]
    game = Game(players, stacks=[10000, 10000], dealer_pos=0)
    changes = game.play_hand()
    assert sum(changes) == 0


def test_all_fold_bb_wins():
    players = [FoldPlayer("P0"), FoldPlayer("P1"), FoldPlayer("P2")]
    game = Game(players, stacks=[10000] * 3, dealer_pos=0)
    changes = game.play_hand()
    # P1=SB(-50), P2=BB wins(+50+50-100=... all folders give BB the pot)
    assert sum(changes) == 0
    # BB should win
    bb_pos = 2  # dealer=0, sb=1, bb=2
    assert changes[bb_pos] >= 0


def test_all_call_showdown():
    players = [CallPlayer(f"P{i}") for i in range(4)]
    game = Game(players, stacks=[10000] * 4, dealer_pos=0, deck=Deck(seed=42))
    changes = game.play_hand()
    assert sum(changes) == 0


def test_table_plays_multiple_hands():
    players = [RandomPlayer(f"P{i}", seed=i) for i in range(6)]
    table = Table(players, starting_stack=10000, seed=42)
    table.play_hands(10)
    assert table.hand_count <= 10
    assert sum(table.total_results) == 0


def test_legal_actions_preflop():
    state = GameState(num_players=3, dealer_pos=0, small_blind=50, big_blind=100)
    for i in range(3):
        state.players.append(PlayerState(seat=i, stack=10000))

    # Post blinds manually
    state.players[1].stack -= 50
    state.players[1].bet_this_round = 50
    state.players[1].bet_total = 50
    state.players[2].stack -= 100
    state.players[2].bet_this_round = 100
    state.players[2].bet_total = 100
    state.pot = 150
    state.current_player = 0
    state.last_raiser = 2

    actions = get_legal_actions(state)
    types = {a.type for a in actions}
    assert ActionType.FOLD in types
    assert ActionType.CALL in types
    assert ActionType.RAISE in types


def test_fold_is_not_legal_when_no_bet():
    state = GameState(num_players=2, dealer_pos=0)
    for i in range(2):
        state.players.append(PlayerState(seat=i, stack=10000))
    state.current_player = 0

    actions = get_legal_actions(state)
    types = {a.type for a in actions}
    assert ActionType.FOLD not in types
    assert ActionType.CALL in types  # Check


def test_chips_conserved_many_hands():
    """Verify chips are perfectly conserved across many random hands."""
    players = [RandomPlayer(f"P{i}", seed=i * 7) for i in range(6)]
    table = Table(players, starting_stack=10000, seed=99)
    table.play_hands(50)

    total_chips = sum(table.stacks)
    assert total_chips == 60000  # 6 * 10000
