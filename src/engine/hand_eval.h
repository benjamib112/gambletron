#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace gambletron {

// Hand rank categories
enum HandCategory : uint8_t {
    HIGH_CARD = 0,
    ONE_PAIR = 1,
    TWO_PAIR = 2,
    THREE_OF_A_KIND = 3,
    STRAIGHT = 4,
    FLUSH = 5,
    FULL_HOUSE = 6,
    FOUR_OF_A_KIND = 7,
    STRAIGHT_FLUSH = 8,
};

// Hand score packed into 32 bits for fast comparison.
// Bits [31:28] = category, remaining bits = tiebreakers
using HandScore = uint32_t;

// Evaluate the best 5-card hand from 7 cards.
// Cards are encoded as integers 0-51: card = rank * 4 + suit
// rank: 0=2, 1=3, ..., 12=Ace; suit: 0=C, 1=D, 2=H, 3=S
HandScore evaluate_7cards(const int* cards);

// Evaluate exactly 5 cards
HandScore evaluate_5cards(const int* cards);

// Evaluate best hand from any number of cards (5-7)
HandScore evaluate_hand(const std::vector<int>& cards);

// Get the category from a hand score
inline HandCategory get_category(HandScore score) {
    return static_cast<HandCategory>(score >> 28);
}

} // namespace gambletron
