"""Tests for card representation."""

from gambletron.poker.card import Card, Deck, Rank, Suit


def test_card_from_rank_suit():
    c = Card.from_rank_suit(Rank.ACE, Suit.SPADES)
    assert c.rank == Rank.ACE
    assert c.suit == Suit.SPADES


def test_card_from_str():
    c = Card.from_str("As")
    assert c.rank == Rank.ACE
    assert c.suit == Suit.SPADES
    assert repr(c) == "As"


def test_card_from_str_all():
    c = Card.from_str("2c")
    assert c.rank == Rank.TWO
    assert c.suit == Suit.CLUBS
    assert c.int_value == 0

    c = Card.from_str("Th")
    assert c.rank == Rank.TEN
    assert c.suit == Suit.HEARTS


def test_card_int_roundtrip():
    for i in range(52):
        c = Card(i)
        assert c.int_value == i
        c2 = Card.from_rank_suit(c.rank, c.suit)
        assert c == c2


def test_card_equality():
    assert Card.from_str("As") == Card.from_str("As")
    assert Card.from_str("As") != Card.from_str("Ah")


def test_deck_deal():
    deck = Deck(seed=42)
    deck.shuffle()
    cards = deck.deal(5)
    assert len(cards) == 5
    assert deck.remaining == 47
    assert len(set(c.int_value for c in cards)) == 5  # All unique


def test_deck_deterministic():
    d1 = Deck(seed=42)
    d1.shuffle()
    cards1 = d1.deal(10)

    d2 = Deck(seed=42)
    d2.shuffle()
    cards2 = d2.deal(10)

    assert [c.int_value for c in cards1] == [c.int_value for c in cards2]


def test_deck_full():
    deck = Deck(seed=0)
    deck.shuffle()
    all_cards = deck.deal(52)
    assert len(set(c.int_value for c in all_cards)) == 52
    assert deck.remaining == 0
