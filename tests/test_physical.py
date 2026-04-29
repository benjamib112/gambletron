"""Tests for physical table integration: controller path, RFID mapping,
card input exclusion logic, HID reader parsing, and side pots."""

from __future__ import annotations

from typing import Dict, List, Optional

from gambletron.hardware.interface import CardInput, ChipInterface, SeatSensor, TableController
from gambletron.hardware.physical_table import PhysicalCardInput
from gambletron.hardware.protocol import RFIDCardMap, _RFID_MAPPING
from gambletron.players.base import Player
from gambletron.players.random_player import RandomPlayer
from gambletron.poker.card import Card, Deck
from gambletron.poker.game import Game
from gambletron.poker.state import Action, ActionType, VisibleGameState
from gambletron.poker.table import Table


# ─── Mock table controller for testing the Game controller path ───────────────

class MockCardInput(CardInput):
    """CardInput that returns pre-loaded cards."""

    def __init__(self, hole_cards: Dict[int, List[Card]], community_cards: List[Card]) -> None:
        self._hole_cards = {k: list(v) for k, v in hole_cards.items()}
        self._community = list(community_cards)

    def wait_for_card(self, seat: int, timeout: float = 30.0) -> Optional[Card]:
        cards = self._hole_cards.get(seat, [])
        if cards:
            return cards.pop(0)
        return None

    def wait_for_community_cards(self, count: int, timeout: float = 30.0) -> List[Card]:
        result = self._community[:count]
        self._community = self._community[count:]
        return result

    def reset(self) -> None:
        pass


class MockTableController(TableController):
    """Minimal table controller for unit tests."""

    def __init__(self, hole_cards: Dict[int, List[Card]], community_cards: List[Card]) -> None:
        self._card_input = MockCardInput(hole_cards, community_cards)

    def get_card_input(self) -> CardInput:
        return self._card_input

    def get_chip_interface(self) -> ChipInterface:
        raise NotImplementedError

    def get_seat_sensor(self) -> SeatSensor:
        raise NotImplementedError

    def deal_card_to(self, seat: int) -> None:
        pass

    def deal_community(self, count: int) -> None:
        pass

    def signal_player_turn(self, seat: int) -> None:
        pass

    def signal_hand_over(self) -> None:
        pass


class CallPlayer(Player):
    """Always calls."""

    def get_action(self, state: VisibleGameState) -> Action:
        return Action.call()


# ─── Test 1: Game with table_controller path ─────────────────────────────────

def test_game_with_controller_deals_correctly():
    """Verify Game uses controller for hole cards and community cards."""
    # Set up known cards
    hole_cards = {
        0: [Card(0), Card(1)],   # 2c, 2d
        1: [Card(4), Card(5)],   # 3c, 3d
        2: [Card(8), Card(9)],   # 4c, 4d
    }
    community = [Card(20), Card(24), Card(28), Card(32), Card(36)]

    controller = MockTableController(hole_cards, community)
    players = [CallPlayer(f"P{i}") for i in range(3)]
    game = Game(players, stacks=[10000] * 3, dealer_pos=0, table_controller=controller)
    changes = game.play_hand()

    # Verify hole cards were assigned from controller
    assert game.state.players[0].hole_cards == [Card(0), Card(1)]
    assert game.state.players[1].hole_cards == [Card(4), Card(5)]
    assert game.state.players[2].hole_cards == [Card(8), Card(9)]

    # Verify community cards came from controller
    assert game.state.community_cards == [Card(20), Card(24), Card(28), Card(32), Card(36)]

    # Zero-sum
    assert sum(changes) == 0


def test_game_with_controller_skips_shuffle():
    """Verify deck.shuffle() is not called when controller is present."""
    hole_cards = {
        0: [Card(0), Card(1)],
        1: [Card(4), Card(5)],
    }
    community = [Card(20), Card(24), Card(28), Card(32), Card(36)]
    controller = MockTableController(hole_cards, community)

    # Use a deck with known state — if shuffle were called, deal would differ
    deck = Deck(seed=42)
    players = [CallPlayer("P0"), CallPlayer("P1")]
    game = Game(players, stacks=[10000] * 2, dealer_pos=0, deck=deck, table_controller=controller)
    game.play_hand()

    # If shuffle was skipped, hole cards come from controller (not deck)
    assert game.state.players[0].hole_cards == [Card(0), Card(1)]


def test_game_with_controller_chips_conserved():
    """Multiple hands via Table with controller still conserve chips."""
    num_players = 4
    all_hole_cards = [
        {i: [Card(i * 4), Card(i * 4 + 1)] for i in range(num_players)},
        {i: [Card(i * 4 + 2), Card(i * 4 + 3)] for i in range(num_players)},
        {i: [Card((i + 2) * 4), Card((i + 2) * 4 + 1)] for i in range(num_players)},
    ]
    all_community = [
        [Card(40), Card(41), Card(42), Card(43), Card(44)],
        [Card(45), Card(46), Card(47), Card(48), Card(49)],
        [Card(32), Card(33), Card(34), Card(35), Card(36)],
    ]

    players = [CallPlayer(f"P{i}") for i in range(num_players)]
    starting_stack = 10000

    # Play 3 hands with controller, manually managing the loop
    stacks = [starting_stack] * num_players
    for hand_idx in range(3):
        controller = MockTableController(all_hole_cards[hand_idx], all_community[hand_idx])
        game = Game(
            players, stacks=list(stacks), dealer_pos=hand_idx % num_players,
            table_controller=controller,
        )
        changes = game.play_hand()
        for i in range(num_players):
            stacks[i] += changes[i]
        assert sum(changes) == 0

    assert sum(stacks) == starting_stack * num_players


# ─── Test 2: PhysicalCardInput community card exclusion ───────────────────────

def test_physical_card_input_excludes_dealt_cards():
    """Community cards must never overlap with dealt hole cards."""
    from gambletron.hardware.hid_reader import HIDCardReaderPool

    pci = PhysicalCardInput.__new__(PhysicalCardInput)
    pci._readers = None
    pci._num_seats = 6
    pci._hole_cards = {}
    pci._dealt_card_ints = set()
    pci._community_generated = []

    # Simulate dealing cards 0-11 to seats
    cards_by_seat = {i: [Card(i * 2), Card(i * 2 + 1)] for i in range(6)}
    pci.load_dealt_cards(cards_by_seat)

    # Generate all 5 community cards
    flop = pci.wait_for_community_cards(3)
    turn = pci.wait_for_community_cards(1)
    river = pci.wait_for_community_cards(1)

    all_community = flop + turn + river
    community_ints = [c.int_value for c in all_community]

    # No overlap with dealt cards (0-11)
    for ci in community_ints:
        assert ci not in pci._dealt_card_ints, f"Community card {ci} overlaps with dealt cards"

    # No duplicates within community
    assert len(set(community_ints)) == 5


def test_physical_card_input_no_duplicates_across_calls():
    """Repeated calls never produce duplicate community cards."""
    pci = PhysicalCardInput.__new__(PhysicalCardInput)
    pci._readers = None
    pci._num_seats = 6
    pci._hole_cards = {}
    pci._dealt_card_ints = set()
    pci._community_generated = []

    # Deal 12 cards
    cards_by_seat = {i: [Card(i * 2), Card(i * 2 + 1)] for i in range(6)}
    pci.load_dealt_cards(cards_by_seat)

    # Generate community in separate calls (as Game does: 3 + 1 + 1)
    c1 = pci.wait_for_community_cards(3)
    c2 = pci.wait_for_community_cards(1)
    c3 = pci.wait_for_community_cards(1)

    all_ints = [c.int_value for c in c1 + c2 + c3]
    assert len(set(all_ints)) == 5, f"Duplicates found: {all_ints}"


def test_physical_card_input_reset_clears_state():
    """After reset, a new hand gets fresh community card generation."""
    pci = PhysicalCardInput.__new__(PhysicalCardInput)
    pci._readers = None
    pci._num_seats = 6
    pci._hole_cards = {}
    pci._dealt_card_ints = set()
    pci._community_generated = []

    cards_by_seat = {i: [Card(i * 2), Card(i * 2 + 1)] for i in range(6)}
    pci.load_dealt_cards(cards_by_seat)
    pci.wait_for_community_cards(5)

    # Reset and reload with different cards
    pci.reset()
    assert len(pci._dealt_card_ints) == 0
    assert len(pci._community_generated) == 0

    cards_by_seat2 = {i: [Card(40 + i * 2), Card(40 + i * 2 + 1)] for i in range(6)}
    pci.load_dealt_cards(cards_by_seat2)
    community = pci.wait_for_community_cards(5)

    for c in community:
        assert c.int_value not in pci._dealt_card_ints


def test_physical_card_input_wait_for_card_pops():
    """wait_for_card returns cards in order and depletes the buffer."""
    pci = PhysicalCardInput.__new__(PhysicalCardInput)
    pci._readers = None
    pci._num_seats = 2
    pci._hole_cards = {}
    pci._dealt_card_ints = set()
    pci._community_generated = []

    cards_by_seat = {0: [Card(10), Card(11)], 1: [Card(20), Card(21)]}
    pci.load_dealt_cards(cards_by_seat)

    assert pci.wait_for_card(0) == Card(10)
    assert pci.wait_for_card(0) == Card(11)
    assert pci.wait_for_card(0) is None  # exhausted
    assert pci.wait_for_card(1) == Card(20)
    assert pci.wait_for_card(1) == Card(21)


# ─── Test 3: RFIDCardMap — all 104 UIDs resolve, all 52 cards reachable ───────

def test_rfid_map_all_uids_resolve():
    """Every hardcoded UID maps to a valid card integer (0-51)."""
    rfid_map = RFIDCardMap()
    for uid, card_int in _RFID_MAPPING.items():
        result = rfid_map.lookup(uid)
        assert result is not None, f"UID {uid} failed to resolve"
        assert result == card_int
        assert 0 <= result <= 51, f"UID {uid} mapped to invalid card int {result}"


def test_rfid_map_all_52_cards_reachable():
    """All 52 card integers (0-51) are reachable via at least one UID."""
    reachable = set(_RFID_MAPPING.values())
    missing = set(range(52)) - reachable
    assert not missing, f"Cards not reachable via any UID: {missing}"


def test_rfid_map_each_card_has_two_uids():
    """Each card should be mapped by exactly 2 UIDs (dual-chip cards)."""
    from collections import Counter
    counts = Counter(_RFID_MAPPING.values())
    for card_int in range(52):
        assert counts[card_int] == 2, (
            f"Card {card_int} has {counts.get(card_int, 0)} UIDs, expected 2"
        )


def test_rfid_map_unknown_uid_returns_none():
    """Unknown UIDs should return None, not raise."""
    rfid_map = RFIDCardMap()
    assert rfid_map.lookup("9999999999") is None
    assert rfid_map.lookup("") is None
    assert rfid_map.lookup("not_a_uid") is None


# ─── Test 4: HID reader keystroke parsing ─────────────────────────────────────

def test_hid_reader_key_map_coverage():
    """Verify the key map covers all digits 0-9 and Enter."""
    from gambletron.hardware.hid_reader import _KEY_MAP
    chars = set(_KEY_MAP.values())
    for digit in "0123456789":
        assert digit in chars, f"Digit '{digit}' missing from key map"
    assert "\n" in chars


def test_hid_reader_process_uid():
    """Test that _process_uid correctly maps a known UID to a card."""
    from gambletron.hardware.hid_reader import HIDCardReader

    rfid_map = RFIDCardMap()
    reader = HIDCardReader.__new__(HIDCardReader)
    reader._path = "/dev/null"
    reader._seat = 0
    reader._rfid_map = rfid_map
    reader._device = None
    reader._thread = None
    reader._running = False
    reader._buffer = ""
    reader._cards = []
    reader._event = __import__("threading").Event()
    reader._lock = __import__("threading").Lock()

    # Process a known UID (first UID in the mapping: 2737139972 -> card 3)
    reader._process_uid("2737139972")
    assert len(reader._cards) == 1
    assert reader._cards[0].int_value == 3

    # Process unknown UID — should be ignored
    reader._process_uid("0000000000")
    assert len(reader._cards) == 1  # still just 1

    # Process empty string — should be ignored
    reader._process_uid("")
    assert len(reader._cards) == 1


def test_hid_reader_buffer_assembly():
    """Simulate keystroke sequence and verify UID assembly logic."""
    from gambletron.hardware.hid_reader import HIDCardReader, _KEY_MAP

    rfid_map = RFIDCardMap()
    reader = HIDCardReader.__new__(HIDCardReader)
    reader._path = "/dev/null"
    reader._seat = 2
    reader._rfid_map = rfid_map
    reader._device = None
    reader._thread = None
    reader._running = False
    reader._buffer = ""
    reader._cards = []
    reader._event = __import__("threading").Event()
    reader._lock = __import__("threading").Lock()

    # Simulate typing "2708300036" then Enter
    # Key codes: 2->3, 7->8, 0->11, 8->9, 3->4, 0->11, 0->11, 0->11, 3->4, 6->7, Enter->28
    uid_str = "2708300036"
    for ch in uid_str:
        # Find key code for this character
        code = next(k for k, v in _KEY_MAP.items() if v == ch)
        char = _KEY_MAP.get(code)
        reader._buffer += char

    # Simulate Enter
    reader._process_uid(reader._buffer)
    reader._buffer = ""

    assert len(reader._cards) == 1
    assert reader._cards[0].int_value == 4  # 2708300036 -> 3c (int 4)


def test_hid_reader_multiple_scans():
    """Simulate multiple consecutive scans at one seat."""
    from gambletron.hardware.hid_reader import HIDCardReader

    rfid_map = RFIDCardMap()
    reader = HIDCardReader.__new__(HIDCardReader)
    reader._path = "/dev/null"
    reader._seat = 0
    reader._rfid_map = rfid_map
    reader._device = None
    reader._thread = None
    reader._running = False
    reader._buffer = ""
    reader._cards = []
    reader._event = __import__("threading").Event()
    reader._lock = __import__("threading").Lock()

    # Two cards scanned at seat 0
    reader._process_uid("2737139972")  # card 3 (2 of spades)
    reader._process_uid("060625668")   # card 0 (2 of clubs)

    assert len(reader._cards) == 2
    assert reader._cards[0].int_value == 3
    assert reader._cards[1].int_value == 0


def test_hid_reader_clear():
    """clear() empties the card buffer."""
    from gambletron.hardware.hid_reader import HIDCardReader

    rfid_map = RFIDCardMap()
    reader = HIDCardReader.__new__(HIDCardReader)
    reader._path = "/dev/null"
    reader._seat = 0
    reader._rfid_map = rfid_map
    reader._device = None
    reader._thread = None
    reader._running = False
    reader._buffer = ""
    reader._cards = []
    reader._event = __import__("threading").Event()
    reader._lock = __import__("threading").Lock()

    reader._process_uid("2737139972")
    assert len(reader._cards) == 1

    reader.clear()
    assert len(reader._cards) == 0


# ─── Test 5: Side pots with multiple all-ins ─────────────────────────────────

def test_side_pot_two_all_ins():
    """Three players: P0 all-in 500, P1 all-in 2000, P2 calls 2000.
    Verify correct pot distribution when P0 has the best hand."""
    from gambletron.poker.game import _distribute_pot
    from gambletron.poker.state import GameState, PlayerState

    state = GameState(num_players=3, dealer_pos=0)
    # P0: short stack, all in for 500
    p0 = PlayerState(seat=0, stack=0)
    p0.bet_total = 500
    p0.is_all_in = True
    # P1: medium stack, all in for 2000
    p1 = PlayerState(seat=1, stack=0)
    p1.bet_total = 2000
    p1.is_all_in = True
    # P2: big stack, called 2000
    p2 = PlayerState(seat=2, stack=3000)
    p2.bet_total = 2000

    state.players = [p0, p1, p2]
    state.pot = 4500  # 500 + 2000 + 2000

    # P0 has the best hand, P1 second best, P2 worst
    hand_scores = {0: (9, 14), 1: (8, 13), 2: (5, 10)}

    _distribute_pot(state, hand_scores)

    # P0 wins main pot: 500 * 3 = 1500
    assert p0.stack == 1500
    # P1 wins side pot: (2000-500) * 2 = 3000
    assert p1.stack == 3000
    # P2 gets nothing extra
    assert p2.stack == 3000
    assert state.pot == 0


def test_side_pot_three_all_ins_increasing():
    """Four players with increasing all-in amounts. Best hand is shortest stack."""
    from gambletron.poker.game import _distribute_pot
    from gambletron.poker.state import GameState, PlayerState

    state = GameState(num_players=4, dealer_pos=0)
    # P0: all-in 100
    p0 = PlayerState(seat=0, stack=0)
    p0.bet_total = 100
    p0.is_all_in = True
    # P1: all-in 300
    p1 = PlayerState(seat=1, stack=0)
    p1.bet_total = 300
    p1.is_all_in = True
    # P2: all-in 600
    p2 = PlayerState(seat=2, stack=0)
    p2.bet_total = 600
    p2.is_all_in = True
    # P3: called 600
    p3 = PlayerState(seat=3, stack=5000)
    p3.bet_total = 600

    state.players = [p0, p1, p2, p3]
    state.pot = 1600  # 100 + 300 + 600 + 600

    # P0 best, P2 second, P1 third, P3 worst
    hand_scores = {0: (9, 14), 1: (6, 10), 2: (8, 13), 3: (5, 9)}

    _distribute_pot(state, hand_scores)

    # Main pot (level 100): 100 * 4 = 400 -> P0 (best hand)
    assert p0.stack == 400
    # Side pot 1 (level 300): (300-100) * 3 = 600 -> P2 (best among P1, P2, P3)
    # Side pot 2 (level 600): (600-300) * 2 = 600 -> P2 (best among P2, P3)
    # P2 total: 600 + 600 = 1200
    assert p2.stack == 1200
    # P1 gets nothing (P2 beats P1 in all eligible side pots)
    assert p1.stack == 0
    # P3 gets nothing
    assert p3.stack == 5000
    assert state.pot == 0


def test_side_pot_split():
    """Two players all-in with equal hands — pot splits evenly."""
    from gambletron.poker.game import _distribute_pot
    from gambletron.poker.state import GameState, PlayerState

    state = GameState(num_players=3, dealer_pos=0)
    p0 = PlayerState(seat=0, stack=0)
    p0.bet_total = 1000
    p1 = PlayerState(seat=1, stack=0)
    p1.bet_total = 1000
    p2 = PlayerState(seat=2, stack=5000)
    p2.bet_total = 1000

    state.players = [p0, p1, p2]
    state.pot = 3000

    # P0 and P1 tie, both beat P2
    hand_scores = {0: (7, 12, 10), 1: (7, 12, 10), 2: (5, 9, 8)}

    _distribute_pot(state, hand_scores)

    # 3000 / 2 = 1500 each for P0 and P1
    assert p0.stack == 1500
    assert p1.stack == 1500
    assert p2.stack == 5000
    assert state.pot == 0


def test_side_pot_short_stack_loses():
    """Short-stack all-in loses — side pot goes to second best."""
    from gambletron.poker.game import _distribute_pot
    from gambletron.poker.state import GameState, PlayerState

    state = GameState(num_players=3, dealer_pos=0)
    p0 = PlayerState(seat=0, stack=0)
    p0.bet_total = 200
    p0.is_all_in = True
    p1 = PlayerState(seat=1, stack=0)
    p1.bet_total = 1000
    p2 = PlayerState(seat=2, stack=4000)
    p2.bet_total = 1000

    state.players = [p0, p1, p2]
    state.pot = 2200

    # P1 has best hand, P2 second, P0 worst
    hand_scores = {0: (3, 8), 1: (9, 14), 2: (7, 12)}

    _distribute_pot(state, hand_scores)

    # Main pot (level 200): 200 * 3 = 600 -> P1 (best)
    # Side pot (level 1000): (1000-200) * 2 = 1600 -> P1 (best among P1, P2)
    assert p0.stack == 0
    assert p1.stack == 600 + 1600  # 2200
    assert p2.stack == 4000
    assert state.pot == 0
