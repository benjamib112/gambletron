"""Tests for hand evaluation."""

from gambletron.poker.card import Card
from gambletron.poker.hand import (
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    ONE_PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    evaluate_hand,
)


def cards(s: str) -> list:
    return [Card.from_str(c) for c in s.split()]


def test_high_card():
    score = evaluate_hand(cards("2c 4d 7h 9s Kc"))
    assert score[0] == HIGH_CARD


def test_one_pair():
    score = evaluate_hand(cards("2c 2d 7h 9s Kc"))
    assert score[0] == ONE_PAIR


def test_two_pair():
    score = evaluate_hand(cards("2c 2d 7h 7s Kc"))
    assert score[0] == TWO_PAIR


def test_three_of_a_kind():
    score = evaluate_hand(cards("2c 2d 2h 9s Kc"))
    assert score[0] == THREE_OF_A_KIND


def test_straight():
    score = evaluate_hand(cards("3c 4d 5h 6s 7c"))
    assert score[0] == STRAIGHT


def test_ace_low_straight():
    score = evaluate_hand(cards("Ac 2d 3h 4s 5c"))
    assert score[0] == STRAIGHT


def test_flush():
    score = evaluate_hand(cards("2c 4c 7c 9c Kc"))
    assert score[0] == FLUSH


def test_full_house():
    score = evaluate_hand(cards("2c 2d 2h Ks Kc"))
    assert score[0] == FULL_HOUSE


def test_four_of_a_kind():
    score = evaluate_hand(cards("2c 2d 2h 2s Kc"))
    assert score[0] == FOUR_OF_A_KIND


def test_straight_flush():
    score = evaluate_hand(cards("3c 4c 5c 6c 7c"))
    assert score[0] == STRAIGHT_FLUSH


def test_hand_ordering():
    high_card = evaluate_hand(cards("2c 4d 7h 9s Kc"))
    pair = evaluate_hand(cards("2c 2d 7h 9s Kc"))
    two_pair = evaluate_hand(cards("2c 2d 7h 7s Kc"))
    trips = evaluate_hand(cards("2c 2d 2h 9s Kc"))
    straight = evaluate_hand(cards("3c 4d 5h 6s 7c"))
    flush = evaluate_hand(cards("2c 4c 7c 9c Kc"))
    full_house = evaluate_hand(cards("2c 2d 2h Ks Kc"))
    quads = evaluate_hand(cards("2c 2d 2h 2s Kc"))
    sf = evaluate_hand(cards("3c 4c 5c 6c 7c"))

    assert high_card < pair < two_pair < trips < straight < flush < full_house < quads < sf


def test_7_card_evaluation():
    # Best 5 from 7: should find the flush
    score = evaluate_hand(cards("2c 4c 7c 9c Kc 3d 5h"))
    assert score[0] == FLUSH


def test_pair_kickers():
    # Higher kicker wins
    low_kicker = evaluate_hand(cards("Ac Ad 3h 4s 5c"))
    high_kicker = evaluate_hand(cards("Ac Ad 3h 4s Kc"))
    assert high_kicker > low_kicker


def test_higher_pair_wins():
    low_pair = evaluate_hand(cards("2c 2d 7h 9s Kc"))
    high_pair = evaluate_hand(cards("Ac Ad 3h 4s 5c"))
    assert high_pair > low_pair
