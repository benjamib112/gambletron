"""Tests for MCCFR training."""

import pytest

from gambletron.ai.blueprint import BlueprintTrainer, PurePythonMCCFR, HAS_CPP_ENGINE
from gambletron.ai.strategy import Strategy


def test_pure_python_mccfr_kuhn():
    """Test pure Python MCCFR on Kuhn poker (3-card, 2-player).

    Kuhn poker Nash equilibrium:
    - Player 0 with Jack: always pass (check)
    - Player 0 with Queen: bet ~1/3 of the time
    - Player 0 with King: bet ~3x as often as with Queen (as a bluff catcher)
    - etc.

    We just verify it converges to reasonable strategies.
    """
    mccfr = PurePythonMCCFR(num_players=2)

    # Simplified Kuhn: cards 0(J), 1(Q), 2(K)
    # Actions: 0=pass/fold, 1=bet/call
    import random

    rng = random.Random(42)

    for iteration in range(10000):
        for traverser in range(2):
            # Deal cards
            cards = rng.sample([0, 1, 2], 2)
            _kuhn_traverse(mccfr, cards, traverser, [], 1.0)
        mccfr.iterations += 1

    # Check that strategies exist and are valid probability distributions
    strategy = mccfr.extract_strategy()
    assert len(strategy) > 0

    # Verify some strategies sum to 1
    for key in mccfr.strategy_sum:
        avg = mccfr.get_average_strategy(key, 2)
        assert abs(sum(avg) - 1.0) < 0.01


def _kuhn_traverse(
    mccfr: PurePythonMCCFR,
    cards: list,
    traverser: int,
    history: list,
    reach_prob: float,
) -> float:
    """Traverse Kuhn poker game tree for external-sampling MCCFR."""
    import random

    rng = random.Random(42 + len(history))

    # Terminal conditions
    if len(history) >= 2:
        if history[-1] == 0 and history[-2] == 1:
            # Fold after bet
            winner = 1 - (len(history) % 2)
            return 1.0 if winner == traverser else -1.0
        if history[-1] == 0 and history[-2] == 0:
            # Both pass: showdown
            winner = 0 if cards[0] > cards[1] else 1
            return 1.0 if winner == traverser else -1.0
        if history[-1] == 1 and history[-2] == 1:
            # Both bet: showdown with pot=2 each
            winner = 0 if cards[0] > cards[1] else 1
            return 2.0 if winner == traverser else -2.0

    if len(history) >= 3:
        # Should not reach here in Kuhn
        return 0.0

    player = len(history) % 2
    card = cards[player]
    key = card * 100 + sum(a * (10 ** i) for i, a in enumerate(history))
    num_actions = 2

    strategy = mccfr.get_strategy(key, num_actions)

    if player == traverser:
        action_values = []
        for a in range(num_actions):
            action_values.append(
                _kuhn_traverse(mccfr, cards, traverser, history + [a], reach_prob)
            )
        mccfr.update_regrets(key, action_values, strategy)
        mccfr.update_strategy_sum(key, strategy)
        return sum(s * v for s, v in zip(strategy, action_values))
    else:
        # Sample opponent action
        a = 0 if rng.random() < strategy[0] else 1
        return _kuhn_traverse(
            mccfr, cards, traverser, history + [a], reach_prob * strategy[a]
        )


@pytest.mark.skipif(not HAS_CPP_ENGINE, reason="C++ engine not built")
def test_cpp_mccfr_small():
    """Test C++ MCCFR trainer with a small number of iterations."""
    trainer = BlueprintTrainer(
        num_players=2,
        discount_interval=50,
        lcfr_threshold=200,
        prune_threshold=100,
        strategy_interval=100,
    )

    strategy = trainer.train(num_iterations=100, verbose=False)
    assert len(strategy) > 0


@pytest.mark.skipif(not HAS_CPP_ENGINE, reason="C++ engine not built")
def test_cpp_mccfr_6player():
    """Test C++ MCCFR with 6 players for a small number of iterations."""
    trainer = BlueprintTrainer(
        num_players=6,
        discount_interval=50,
        lcfr_threshold=200,
        prune_threshold=100,
        strategy_interval=100,
    )

    strategy = trainer.train(num_iterations=50, verbose=False)
    assert len(strategy) > 0


def test_strategy_serialization(tmp_path):
    """Test saving and loading strategies."""
    strategy = Strategy()
    strategy.set(12345, [0.3, 0.5, 0.2])
    strategy.set(67890, [0.7, 0.3])

    path = tmp_path / "test_strategy.pkl"
    strategy.save(path)

    loaded = Strategy.from_file(path)
    assert loaded.get(12345) == [0.3, 0.5, 0.2]
    assert loaded.get(67890) == [0.7, 0.3]
    assert loaded.get(99999) is None
