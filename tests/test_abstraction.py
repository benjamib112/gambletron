"""Tests for information and action abstraction."""

from gambletron.ai.abstraction import (
    canonical_preflop,
    HandStrengthCalculator,
    PREFLOP_BUCKETS,
)
from gambletron.ai.action_abstraction import (
    ActionAbstraction,
    pseudo_harmonic_mapping,
)
from gambletron.poker.card import Card
from gambletron.poker.state import BettingRound


def test_preflop_buckets_count():
    """All 1326 preflop combos map to exactly 169 canonical buckets."""
    buckets = set()
    for i in range(52):
        for j in range(i + 1, 52):
            b = canonical_preflop(Card(i), Card(j))
            assert 0 <= b < PREFLOP_BUCKETS
            buckets.add(b)
    assert len(buckets) == 169


def test_suited_vs_offsuit():
    # AKs and AKo should be different buckets
    aks = canonical_preflop(Card.from_str("As"), Card.from_str("Ks"))
    ako = canonical_preflop(Card.from_str("Ah"), Card.from_str("Kd"))
    assert aks != ako


def test_pairs_same_bucket():
    # AA of different suits -> same bucket
    b1 = canonical_preflop(Card.from_str("Ac"), Card.from_str("Ad"))
    b2 = canonical_preflop(Card.from_str("Ah"), Card.from_str("As"))
    assert b1 == b2


def test_hand_strength_reasonable():
    calc = HandStrengthCalculator(num_rollouts=500, seed=42)
    # AA preflop should be strong
    aa_strength = calc.hand_strength([12 * 4, 12 * 4 + 1], [])
    assert aa_strength > 0.7

    # 72o preflop should be weak
    low_strength = calc.hand_strength([5 * 4, 0 * 4 + 1], [])
    assert low_strength < 0.5

    assert aa_strength > low_strength


def test_action_abstraction_blueprint():
    aa = ActionAbstraction.blueprint()
    sizes = aa.get_raise_sizes(
        betting_round=BettingRound.PREFLOP,
        raise_num=0,
        pot=150,  # SB + BB
        current_bet=100,
        player_stack=10000,
        player_bet=0,
        min_raise=100,
    )
    assert len(sizes) > 0
    assert all(s > 100 for s in sizes)  # All above current bet
    assert sizes[-1] == 10000  # All-in included


def test_action_abstraction_search():
    aa = ActionAbstraction.search()
    sizes = aa.get_raise_sizes(
        betting_round=BettingRound.FLOP,
        raise_num=0,
        pot=400,
        current_bet=0,
        player_stack=9600,
        player_bet=0,
        min_raise=100,
    )
    assert len(sizes) > 0
    assert sizes[-1] == 9600  # All-in


def test_pseudo_harmonic_mapping():
    # Exact match on lower
    assert pseudo_harmonic_mapping(100, 100, 200) == 1.0
    # Exact match on upper
    assert pseudo_harmonic_mapping(200, 100, 200) == 0.0
    # Midpoint should be close to 0.5
    p = pseudo_harmonic_mapping(150, 100, 200)
    assert 0.4 <= p <= 0.6
    # Closer to lower -> higher probability for lower
    p_near_low = pseudo_harmonic_mapping(110, 100, 200)
    p_near_high = pseudo_harmonic_mapping(190, 100, 200)
    assert p_near_low > p_near_high
