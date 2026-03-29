#include "hand_eval.h"

#include <algorithm>
#include <array>

namespace gambletron {

namespace {

// Pack a hand score: category in top 4 bits, up to 5 tiebreaker nibbles below
inline HandScore pack_score(HandCategory cat, int t0 = 0, int t1 = 0,
                            int t2 = 0, int t3 = 0, int t4 = 0) {
    return (static_cast<uint32_t>(cat) << 28) |
           (static_cast<uint32_t>(t0 & 0xF) << 24) |
           (static_cast<uint32_t>(t1 & 0xF) << 20) |
           (static_cast<uint32_t>(t2 & 0xF) << 16) |
           (static_cast<uint32_t>(t3 & 0xF) << 12) |
           (static_cast<uint32_t>(t4 & 0xF) << 8);
}

struct EvalResult {
    HandCategory category;
    std::array<int, 5> tiebreakers; // Most significant first
};

HandScore eval5_impl(const int cards[5]) {
    int ranks[5], suits[5];
    for (int i = 0; i < 5; i++) {
        ranks[i] = cards[i] / 4;  // 0=2 .. 12=Ace
        suits[i] = cards[i] % 4;
    }

    // Sort ranks descending
    int sorted_ranks[5];
    std::copy(ranks, ranks + 5, sorted_ranks);
    std::sort(sorted_ranks, sorted_ranks + 5, std::greater<int>());

    // Check flush
    bool is_flush = (suits[0] == suits[1]) && (suits[1] == suits[2]) &&
                    (suits[2] == suits[3]) && (suits[3] == suits[4]);

    // Check straight
    bool is_straight = false;
    int straight_high = -1;

    // Normal straight check
    if (sorted_ranks[0] - sorted_ranks[4] == 4) {
        // Check all unique
        bool all_unique = true;
        for (int i = 0; i < 4; i++) {
            if (sorted_ranks[i] == sorted_ranks[i + 1]) {
                all_unique = false;
                break;
            }
        }
        if (all_unique) {
            is_straight = true;
            straight_high = sorted_ranks[0];
        }
    }

    // Ace-low straight: A-2-3-4-5
    if (!is_straight && sorted_ranks[0] == 12 && sorted_ranks[1] == 3 &&
        sorted_ranks[2] == 2 && sorted_ranks[3] == 1 &&
        sorted_ranks[4] == 0) {
        is_straight = true;
        straight_high = 3; // 5-high
    }

    if (is_straight && is_flush)
        return pack_score(STRAIGHT_FLUSH, straight_high);

    // Count rank occurrences
    int count[13] = {};
    for (int i = 0; i < 5; i++) count[ranks[i]]++;

    // Find groups: (count, rank) pairs sorted by count desc, then rank desc
    struct Group {
        int count;
        int rank;
    };
    Group groups[5];
    int ngroups = 0;
    for (int r = 12; r >= 0; r--) {
        if (count[r] > 0) {
            groups[ngroups++] = {count[r], r};
        }
    }
    // Stable sort by count descending (rank already descending within same count)
    std::sort(groups, groups + ngroups,
              [](const Group& a, const Group& b) {
                  return a.count > b.count ||
                         (a.count == b.count && a.rank > b.rank);
              });

    if (groups[0].count == 4)
        return pack_score(FOUR_OF_A_KIND, groups[0].rank, groups[1].rank);

    if (groups[0].count == 3 && groups[1].count == 2)
        return pack_score(FULL_HOUSE, groups[0].rank, groups[1].rank);

    if (is_flush)
        return pack_score(FLUSH, sorted_ranks[0], sorted_ranks[1],
                          sorted_ranks[2], sorted_ranks[3], sorted_ranks[4]);

    if (is_straight)
        return pack_score(STRAIGHT, straight_high);

    if (groups[0].count == 3)
        return pack_score(THREE_OF_A_KIND, groups[0].rank, groups[1].rank,
                          groups[2].rank);

    if (groups[0].count == 2 && groups[1].count == 2)
        return pack_score(TWO_PAIR, groups[0].rank, groups[1].rank,
                          groups[2].rank);

    if (groups[0].count == 2)
        return pack_score(ONE_PAIR, groups[0].rank, groups[1].rank,
                          groups[2].rank, groups[3].rank);

    return pack_score(HIGH_CARD, sorted_ranks[0], sorted_ranks[1],
                      sorted_ranks[2], sorted_ranks[3], sorted_ranks[4]);
}

} // anonymous namespace

HandScore evaluate_5cards(const int* cards) { return eval5_impl(cards); }

HandScore evaluate_7cards(const int* cards) {
    // Try all C(7,5)=21 combinations
    HandScore best = 0;
    int combo[5];
    for (int i = 0; i < 7; i++) {
        for (int j = i + 1; j < 7; j++) {
            // Exclude cards i and j
            int k = 0;
            for (int c = 0; c < 7; c++) {
                if (c != i && c != j) {
                    combo[k++] = cards[c];
                }
            }
            HandScore score = eval5_impl(combo);
            if (score > best) best = score;
        }
    }
    return best;
}

HandScore evaluate_hand(const std::vector<int>& cards) {
    if (cards.size() == 5) return evaluate_5cards(cards.data());
    if (cards.size() == 7) return evaluate_7cards(cards.data());

    // General case: try all C(n,5) combinations
    int n = static_cast<int>(cards.size());
    HandScore best = 0;
    int combo[5];
    // Simple nested loop for small n (5-7)
    for (int a = 0; a < n - 4; a++)
        for (int b = a + 1; b < n - 3; b++)
            for (int c = b + 1; c < n - 2; c++)
                for (int d = c + 1; d < n - 1; d++)
                    for (int e = d + 1; e < n; e++) {
                        combo[0] = cards[a];
                        combo[1] = cards[b];
                        combo[2] = cards[c];
                        combo[3] = cards[d];
                        combo[4] = cards[e];
                        HandScore s = eval5_impl(combo);
                        if (s > best) best = s;
                    }
    return best;
}

} // namespace gambletron
