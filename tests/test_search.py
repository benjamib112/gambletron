"""Tests for real-time search and belief tracking."""

import numpy as np

from gambletron.ai.belief import BeliefState, hand_to_index, index_to_hand, NUM_HANDS
from gambletron.ai.search import RealTimeSearch, SubgameState
from gambletron.ai.strategy import Strategy
from gambletron.poker.state import ActionType


def test_belief_initialization():
    beliefs = BeliefState(num_players=6)
    assert beliefs.beliefs[0].shape == (NUM_HANDS,)
    assert abs(beliefs.beliefs[0].sum() - 1.0) < 1e-6


def test_belief_remove_known_cards():
    beliefs = BeliefState(num_players=2)
    # Remove ace of spades (card 51) and king of spades (card 47)
    beliefs.remove_known_cards([51, 47])

    # All hands containing card 51 or 47 should be zero
    for idx in range(NUM_HANDS):
        c1, c2 = index_to_hand(idx)
        if c1 == 51 or c2 == 51 or c1 == 47 or c2 == 47:
            assert beliefs.beliefs[0][idx] == 0.0

    assert abs(beliefs.beliefs[0].sum() - 1.0) < 1e-6


def test_belief_update():
    beliefs = BeliefState(num_players=2)

    # Simulate: player 0 took an action. Hands with high cards are more
    # likely to take this action.
    action_probs = np.ones(NUM_HANDS, dtype=np.float64) * 0.1
    # Hands with aces get higher probability
    for idx in range(NUM_HANDS):
        c1, c2 = index_to_hand(idx)
        r1, r2 = c1 // 4, c2 // 4  # ranks
        if r1 == 12 or r2 == 12:  # Ace
            action_probs[idx] = 0.9

    beliefs.update_on_action(0, action_probs)

    # Ace-containing hands should now have higher belief
    ace_belief = 0.0
    non_ace_belief = 0.0
    for idx in range(NUM_HANDS):
        c1, c2 = index_to_hand(idx)
        r1, r2 = c1 // 4, c2 // 4
        if r1 == 12 or r2 == 12:
            ace_belief += beliefs.beliefs[0][idx]
        else:
            non_ace_belief += beliefs.beliefs[0][idx]

    assert ace_belief > 0.5  # Aces should dominate after update


def test_hand_index_roundtrip():
    for c1 in range(52):
        for c2 in range(c1 + 1, 52):
            idx = hand_to_index(c1, c2)
            assert 0 <= idx < NUM_HANDS
            rc1, rc2 = index_to_hand(idx)
            assert (rc1, rc2) == (c1, c2)


def test_search_basic():
    """Test that search produces valid action probabilities."""
    blueprint = Strategy()
    # Seed some dummy strategies
    for key in range(100):
        blueprint.set(key, [0.4, 0.4, 0.2])

    search = RealTimeSearch(
        blueprint=blueprint,
        num_players=2,
        num_search_iters=50,
        seed=42,
    )

    state = SubgameState(
        num_players=2,
        pot=300,
        betting_round=1,  # Flop
        community_cards=[0, 5, 10],  # 2c, 3d, 4h
        player_stacks=[9700, 9700],
        player_bets=[0, 0],
        player_folded=[False, False],
        player_all_in=[False, False],
        current_player=0,
        hole_cards={0: (48, 49), 1: (20, 21)},  # Known for testing
        action_history=[],
    )

    beliefs = BeliefState(num_players=2)
    beliefs.remove_known_cards([0, 5, 10, 48, 49])

    probs = search.search(state, our_player=0, beliefs=beliefs)
    assert len(probs) > 0
    assert abs(sum(probs) - 1.0) < 0.1  # Should approximately sum to 1
    assert all(p >= 0 for p in probs)
