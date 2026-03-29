#include "mccfr.h"
#include "hand_eval.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace gambletron {

// Forward declarations
static void apply_action_to_state(MCCFRState& state, int action, int player);
static std::vector<int> get_available_actions(const MCCFRState& state, int player);

// ============================================================
// Built-in C++ infoset key function
// ============================================================

// Canonical preflop bucket (169 buckets: pairs, suited, offsuit)
static int canonical_preflop(int card0, int card1) {
    int rank0 = card0 / 4;  // 0=2, 1=3, ..., 12=A
    int suit0 = card0 % 4;
    int rank1 = card1 / 4;
    int suit1 = card1 % 4;

    int hi = std::max(rank0, rank1);
    int lo = std::min(rank0, rank1);
    bool suited = (suit0 == suit1);

    if (hi == lo) {
        // Pair: bucket 0-12
        return hi;
    } else if (suited) {
        // Suited: hi * 13 + lo, offset by 13 (skip pairs)
        // Map to unique index in 13 + 78 suited + 78 offsuit = 169
        return 13 + (hi * (hi - 1) / 2) + lo;
    } else {
        // Offsuit: offset by 13 + 78 = 91
        return 91 + (hi * (hi - 1) / 2) + lo;
    }
}

// Simple hash combine
static uint64_t hash_combine(uint64_t seed, uint64_t val) {
    seed ^= val * 0x9e3779b97f4a7c15ULL + 0x9e3779b9ULL + (seed << 6) + (seed >> 2);
    return seed;
}

InfosetKey builtin_infoset_key(
    int player, int betting_round,
    const int* hole_cards, int hole_len,
    const int* board, int board_len,
    const int* action_seq, int action_seq_len)
{
    uint64_t card_bucket;

    if (betting_round == 0) {
        card_bucket = canonical_preflop(hole_cards[0], hole_cards[1]);
    } else {
        // Postflop: hash sorted hole cards + sorted board
        int sorted_hole[2] = {
            std::min(hole_cards[0], hole_cards[1]),
            std::max(hole_cards[0], hole_cards[1])
        };
        uint64_t h = 0;
        h = hash_combine(h, sorted_hole[0]);
        h = hash_combine(h, sorted_hole[1]);

        // Sort board cards for canonical form
        int sorted_board[5];
        std::copy(board, board + board_len, sorted_board);
        std::sort(sorted_board, sorted_board + board_len);
        for (int i = 0; i < board_len; i++) {
            h = hash_combine(h, sorted_board[i]);
        }
        card_bucket = h % 50;
    }

    // Hash action sequence
    uint64_t action_hash = 0;
    for (int i = 0; i < action_seq_len; i++) {
        action_hash = hash_combine(action_hash, action_seq[i]);
    }
    action_hash &= 0xFFFFFFFFULL;

    // Combine: player, round, card bucket, action hash
    uint64_t key = (static_cast<uint64_t>(player) << 56)
                 | (static_cast<uint64_t>(betting_round) << 48)
                 | (card_bucket << 32)
                 | action_hash;
    return key;
}

// ============================================================
// MCCFRTrainer implementation
// ============================================================

MCCFRTrainer::MCCFRTrainer(const MCCFRConfig& config, InfosetKeyFn key_fn)
    : config_(config), key_fn_(std::move(key_fn)), use_builtin_key_(false) {}

MCCFRTrainer::MCCFRTrainer(const MCCFRConfig& config)
    : config_(config), use_builtin_key_(true) {}

InfosetKey MCCFRTrainer::compute_key(const MCCFRState& state, int player) {
    if (use_builtin_key_) {
        return builtin_infoset_key(
            player, state.betting_round,
            state.hole_cards[player], 2,
            state.board, state.board_len,
            state.action_history, state.action_history_len);
    } else {
        return key_fn_(
            player, state.betting_round,
            std::vector<int>(state.hole_cards[player], state.hole_cards[player] + 2),
            std::vector<int>(state.board, state.board + state.board_len),
            state.board_len,
            std::vector<int>(state.action_history, state.action_history + state.action_history_len),
            state.action_history_len);
    }
}

void MCCFRTrainer::run_iteration(int total_iter, ThreadContext& ctx) {
    for (int player = 0; player < config_.num_players; player++) {
        // Update average strategy periodically (preflop only)
        if (total_iter % config_.strategy_interval == 0) {
            MCCFRState state = create_initial_state(ctx);
            update_strategy(state, player, ctx);
        }

        // Choose traversal method
        if (total_iter > config_.prune_threshold) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            if (dist(ctx.rng) < 0.05f) {
                MCCFRState state = create_initial_state(ctx);
                traverse_mccfr(state, player, ctx);
            } else {
                MCCFRState state = create_initial_state(ctx);
                traverse_mccfr_pruned(state, player, ctx);
            }
        } else {
            MCCFRState state = create_initial_state(ctx);
            traverse_mccfr(state, player, ctx);
        }
    }
}

void MCCFRTrainer::train(int num_iterations) {
    int num_threads = std::max(1, config_.num_threads);

    if (num_threads == 1) {
        // Single-threaded path (also works with Python callback)
        ThreadContext ctx;
        ctx.rng.seed(42 + iterations_done_.load());

        for (int t = 0; t < num_iterations; t++) {
            int total_iter = iterations_done_.load() + 1;
            run_iteration(total_iter, ctx);
            iterations_done_++;

            // Linear CFR discounting
            if (iterations_done_ < config_.lcfr_threshold &&
                iterations_done_ % config_.discount_interval == 0) {
                int64_t done = iterations_done_.load();
                float d = static_cast<float>(done / config_.discount_interval) /
                          (done / config_.discount_interval + 1.0f);
                store_.discount(d);
            }
        }
    } else {
        // Multi-threaded path (requires built-in key function)
        // Run in batches: threads work in parallel, discounting between batches
        int batch_size = config_.discount_interval > 0 ? config_.discount_interval : num_iterations;
        int remaining = num_iterations;

        while (remaining > 0) {
            int this_batch = std::min(remaining, batch_size);
            std::atomic<int> batch_remaining(this_batch);

            auto worker = [&](int thread_id) {
                ThreadContext ctx;
                ctx.rng.seed(42 + thread_id * 1000000 + iterations_done_.load());

                while (true) {
                    int left = batch_remaining.fetch_sub(1);
                    if (left <= 0) {
                        batch_remaining.fetch_add(1);
                        break;
                    }

                    int total_iter = iterations_done_.load() + 1;
                    run_iteration(total_iter, ctx);
                    ++iterations_done_;
                }
            };

            std::vector<std::thread> threads;
            for (int i = 0; i < num_threads; i++) {
                threads.emplace_back(worker, i);
            }
            for (auto& t : threads) {
                t.join();
            }

            remaining -= this_batch;

            // Linear CFR discounting (main thread only, no concurrent access)
            int64_t done = iterations_done_.load();
            if (done < config_.lcfr_threshold &&
                done % config_.discount_interval == 0) {
                float d = static_cast<float>(done / config_.discount_interval) /
                          (done / config_.discount_interval + 1.0f);
                store_.discount(d);
            }
        }
    }
}

MCCFRState MCCFRTrainer::create_initial_state(ThreadContext& ctx) {
    MCCFRState state{};
    state.num_players = config_.num_players;
    state.betting_round = 0;
    state.board_len = 0;
    state.action_history_len = 0;

    // Deal cards
    std::vector<int> deck(52);
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), ctx.rng);

    int idx = 0;
    for (int i = 0; i < config_.num_players; i++) {
        state.hole_cards[i][0] = deck[idx++];
        state.hole_cards[i][1] = deck[idx++];
        state.player_folded[i] = false;
        state.player_all_in[i] = false;
        state.player_stacks[i] = 10000;
        state.player_bets[i] = 0;
    }

    // Store remaining deck for board cards
    for (int i = 0; i < 5; i++) {
        state.board[i] = deck[idx++];
    }

    // Post blinds (heads up: dealer=SB=0, BB=1; otherwise SB=1, BB=2)
    int sb_pos, bb_pos;
    if (config_.num_players == 2) {
        sb_pos = 0;
        bb_pos = 1;
    } else {
        sb_pos = 1;
        bb_pos = 2;
    }

    state.player_stacks[sb_pos] -= 50;
    state.player_bets[sb_pos] = 50;
    state.player_stacks[bb_pos] -= 100;
    state.player_bets[bb_pos] = 100;
    state.pot = 150;

    // First to act preflop
    if (config_.num_players == 2) {
        state.current_player = 0;
    } else {
        state.current_player = (bb_pos + 1) % config_.num_players;
    }

    state.actions_this_round = 0;
    state.raises_this_round = 1;  // BB counts as first raise
    state.last_raiser = bb_pos;   // BB counts as initial "raiser"
    state.round_starter = state.current_player;

    return state;
}

float MCCFRTrainer::traverse_mccfr(MCCFRState& state, int traverser, ThreadContext& ctx) {
    static constexpr int INITIAL_STACK = 10000;

    // Terminal node
    int active_count = 0;
    int last_active = -1;
    for (int i = 0; i < state.num_players; i++) {
        if (!state.player_folded[i]) {
            active_count++;
            last_active = i;
        }
    }

    if (active_count == 1) {
        // Net payoff: remaining stack (+ pot if winner) - initial stack
        if (last_active == traverser)
            return static_cast<float>(state.player_stacks[traverser] + state.pot - INITIAL_STACK);
        return static_cast<float>(state.player_stacks[traverser] - INITIAL_STACK);
    }

    if (state.betting_round > 3) {
        float payoffs[6] = {};
        get_payoffs(state, payoffs);
        return payoffs[traverser];
    }

    if (state.player_folded[traverser]) {
        // Traverser folded: lost whatever they put in
        return static_cast<float>(state.player_stacks[traverser] - INITIAL_STACK);
    }

    int cp = state.current_player;

    InfosetKey key = compute_key(state, cp);

    std::vector<int> actions = get_available_actions(state, cp);

    int num_actions = static_cast<int>(actions.size());
    InfosetData& data = store_.get_or_create(key, num_actions);
    data.calculate_strategy();

    if (cp == traverser) {
        std::vector<float> action_values(num_actions, 0.0f);
        float node_value = 0.0f;

        for (int a = 0; a < num_actions; a++) {
            MCCFRState child = state;
            apply_action_to_state(child, actions[a], cp);
            action_values[a] = traverse_mccfr(child, traverser, ctx);
            node_value += data.current_strategy[a] * action_values[a];
        }

        for (int a = 0; a < num_actions; a++) {
            int32_t regret_delta = static_cast<int32_t>(action_values[a] - node_value);
            data.regrets[a] += regret_delta;
            data.regrets[a] = std::max(data.regrets[a], config_.regret_floor);
        }

        return node_value;
    } else {
        std::discrete_distribution<int> dist(
            data.current_strategy, data.current_strategy + num_actions);
        int sampled = dist(ctx.rng);

        MCCFRState child = state;
        apply_action_to_state(child, actions[sampled], cp);
        return traverse_mccfr(child, traverser, ctx);
    }
}

float MCCFRTrainer::traverse_mccfr_pruned(MCCFRState& state, int traverser, ThreadContext& ctx) {
    static constexpr int INITIAL_STACK = 10000;

    int active_count = 0;
    int last_active = -1;
    for (int i = 0; i < state.num_players; i++) {
        if (!state.player_folded[i]) {
            active_count++;
            last_active = i;
        }
    }

    if (active_count == 1) {
        if (last_active == traverser)
            return static_cast<float>(state.player_stacks[traverser] + state.pot - INITIAL_STACK);
        return static_cast<float>(state.player_stacks[traverser] - INITIAL_STACK);
    }

    if (state.betting_round > 3) {
        float payoffs[6] = {};
        get_payoffs(state, payoffs);
        return payoffs[traverser];
    }

    if (state.player_folded[traverser]) {
        return static_cast<float>(state.player_stacks[traverser] - INITIAL_STACK);
    }

    int cp = state.current_player;

    InfosetKey key = compute_key(state, cp);

    std::vector<int> actions = get_available_actions(state, cp);

    int num_actions = static_cast<int>(actions.size());
    InfosetData& data = store_.get_or_create(key, num_actions);
    data.calculate_strategy();

    if (cp == traverser) {
        std::vector<float> action_values(num_actions, 0.0f);
        std::vector<bool> explored(num_actions, false);
        float node_value = 0.0f;

        for (int a = 0; a < num_actions; a++) {
            if (data.regrets[a] <= config_.prune_floor &&
                state.betting_round < 3 &&
                actions[a] != 0)
            {
                continue;
            }

            MCCFRState child = state;
            apply_action_to_state(child, actions[a], cp);
            action_values[a] = traverse_mccfr_pruned(child, traverser, ctx);
            explored[a] = true;
            node_value += data.current_strategy[a] * action_values[a];
        }

        for (int a = 0; a < num_actions; a++) {
            if (explored[a]) {
                int32_t regret_delta = static_cast<int32_t>(action_values[a] - node_value);
                data.regrets[a] += regret_delta;
                data.regrets[a] = std::max(data.regrets[a], config_.regret_floor);
            }
        }

        return node_value;
    } else {
        std::discrete_distribution<int> dist(
            data.current_strategy, data.current_strategy + num_actions);
        int sampled = dist(ctx.rng);
        MCCFRState child = state;
        apply_action_to_state(child, actions[sampled], cp);
        return traverse_mccfr_pruned(child, traverser, ctx);
    }
}

void MCCFRTrainer::update_strategy(MCCFRState& state, int player, ThreadContext& ctx) {
    if (state.betting_round > 0) return;

    int active_count = 0;
    for (int i = 0; i < state.num_players; i++) {
        if (!state.player_folded[i]) active_count++;
    }
    if (active_count <= 1) return;

    int cp = state.current_player;

    InfosetKey key = compute_key(state, cp);

    std::vector<int> actions = get_available_actions(state, cp);

    int num_actions = static_cast<int>(actions.size());
    InfosetData& data = store_.get_or_create(key, num_actions);
    data.calculate_strategy();

    if (cp == player) {
        std::discrete_distribution<int> dist(
            data.current_strategy, data.current_strategy + num_actions);
        int sampled = dist(ctx.rng);
        data.avg_strategy[sampled] += 1;

        MCCFRState child = state;
        apply_action_to_state(child, actions[sampled], cp);
        update_strategy(child, player, ctx);
    } else {
        for (int a = 0; a < num_actions; a++) {
            MCCFRState child = state;
            apply_action_to_state(child, actions[a], cp);
            update_strategy(child, player, ctx);
        }
    }
}

const float* MCCFRTrainer::get_strategy(InfosetKey key) const {
    auto* data = const_cast<InfosetStore&>(store_).get(key);
    return data ? data->current_strategy : nullptr;
}

std::vector<float> MCCFRTrainer::get_average_strategy(InfosetKey key) const {
    auto* data = const_cast<InfosetStore&>(store_).get(key);
    if (!data) return {};

    std::vector<float> avg(data->num_actions);
    float sum = 0;
    for (int i = 0; i < data->num_actions; i++) {
        sum += std::max(data->avg_strategy[i], (int32_t)0);
    }
    if (sum > 0) {
        for (int i = 0; i < data->num_actions; i++) {
            avg[i] = std::max(data->avg_strategy[i], (int32_t)0) / sum;
        }
    } else {
        float uniform = 1.0f / data->num_actions;
        for (int i = 0; i < data->num_actions; i++) {
            avg[i] = uniform;
        }
    }
    return avg;
}

// Binary checkpoint format:
//   Magic: "GBTC" (4 bytes)
//   Version: uint32
//   iterations_done: int64
//   num_entries: uint64
//   For each entry:
//     key: uint64
//     num_actions: int32
//     regrets[num_actions]: int32[]
//     avg_strategy[num_actions]: int32[]

void MCCFRTrainer::save_checkpoint(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open checkpoint file for writing: " + path);

    // Header
    out.write("GBTC", 4);
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    int64_t iters = iterations_done_.load();
    out.write(reinterpret_cast<const char*>(&iters), sizeof(iters));

    uint64_t num_entries = store_.size();
    out.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));

    // Entries
    store_.for_each([&](InfosetKey key, const InfosetData& data) {
        out.write(reinterpret_cast<const char*>(&key), sizeof(key));
        out.write(reinterpret_cast<const char*>(&data.num_actions), sizeof(data.num_actions));
        out.write(reinterpret_cast<const char*>(data.regrets),
                  sizeof(int32_t) * data.num_actions);
        out.write(reinterpret_cast<const char*>(data.avg_strategy),
                  sizeof(int32_t) * data.num_actions);
    });
}

void MCCFRTrainer::load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open checkpoint file for reading: " + path);

    // Verify magic
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "GBTC")
        throw std::runtime_error("Invalid checkpoint file (bad magic)");

    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1)
        throw std::runtime_error("Unsupported checkpoint version: " + std::to_string(version));

    int64_t iters;
    in.read(reinterpret_cast<char*>(&iters), sizeof(iters));
    iterations_done_ = iters;

    uint64_t num_entries;
    in.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));

    for (uint64_t i = 0; i < num_entries; i++) {
        InfosetKey key;
        int num_actions;
        in.read(reinterpret_cast<char*>(&key), sizeof(key));
        in.read(reinterpret_cast<char*>(&num_actions), sizeof(num_actions));

        InfosetData& data = store_.get_or_create(key, num_actions);
        in.read(reinterpret_cast<char*>(data.regrets),
                sizeof(int32_t) * num_actions);
        in.read(reinterpret_cast<char*>(data.avg_strategy),
                sizeof(int32_t) * num_actions);
        data.calculate_strategy();
    }
}

void MCCFRTrainer::get_payoffs(const MCCFRState& state, float* payoffs) {
    static constexpr int INITIAL_STACK = 10000;

    // Base: everyone starts at net = remaining_stack - initial_stack (negative if invested)
    for (int i = 0; i < state.num_players; i++) {
        payoffs[i] = static_cast<float>(state.player_stacks[i] - INITIAL_STACK);
    }

    std::vector<int> in_hand;
    for (int i = 0; i < state.num_players; i++) {
        if (!state.player_folded[i]) {
            in_hand.push_back(i);
        }
    }

    if (in_hand.size() == 1) {
        payoffs[in_hand[0]] += static_cast<float>(state.pot);
        return;
    }

    HandScore best_score = 0;
    std::vector<int> winners;

    for (int p : in_hand) {
        int cards[7] = {
            state.hole_cards[p][0], state.hole_cards[p][1],
            state.board[0], state.board[1], state.board[2],
            state.board[3], state.board[4]
        };
        HandScore score = evaluate_7cards(cards);
        if (score > best_score) {
            best_score = score;
            winners.clear();
            winners.push_back(p);
        } else if (score == best_score) {
            winners.push_back(p);
        }
    }

    float share = static_cast<float>(state.pot) / winners.size();
    for (int w : winners) {
        payoffs[w] += share;
    }
}

// Build available actions for a player at the current state.
// Action encoding:
//   0 = fold (only when facing a bet)
//   1 = call/check
//   N >= 100 = raise TO total bet of N
//   -1 = all-in (special)
static std::vector<int> get_available_actions(const MCCFRState& state, int player) {
    int current_bet = 0;
    for (int i = 0; i < state.num_players; i++) {
        current_bet = std::max(current_bet, state.player_bets[i]);
    }
    int to_call = current_bet - state.player_bets[player];
    int stack = state.player_stacks[player];

    std::vector<int> actions;

    // Fold (only if facing a bet)
    if (to_call > 0) actions.push_back(0);

    // Call/Check
    actions.push_back(1);

    // Raises: only if we have chips after calling and haven't hit raise cap
    static constexpr int MAX_RAISES_PER_ROUND = 2;
    int chips_after_call = stack - to_call;
    if (chips_after_call <= 0) return actions;
    if (state.raises_this_round >= MAX_RAISES_PER_ROUND) return actions;

    int min_raise_to = current_bet + 100; // min raise = previous bet + BB
    int max_raise_to = state.player_bets[player] + stack;
    if (max_raise_to <= current_bet) return actions;

    // Clamp min raise
    if (min_raise_to > max_raise_to) min_raise_to = max_raise_to;

    if (state.betting_round == 0) {
        // Preflop: standard open (2.5x BB = 250) and pot-sized raise
        int standard_open = 250;
        if (standard_open >= min_raise_to && standard_open < max_raise_to) {
            actions.push_back(standard_open);
        }
        int pot_raise = state.pot + 2 * to_call + current_bet;
        if (pot_raise >= min_raise_to && pot_raise < max_raise_to &&
            pot_raise != standard_open) {
            actions.push_back(pot_raise);
        }
        // All-in
        if (max_raise_to > min_raise_to) {
            actions.push_back(-1);
        }
    } else {
        // Postflop: pot-sized raise only (keeps tree small)
        int pot_raise = current_bet + state.pot + to_call;
        if (pot_raise >= min_raise_to && pot_raise <= max_raise_to) {
            actions.push_back(pot_raise);
        } else if (max_raise_to >= min_raise_to) {
            // Can't make full pot raise, just go all-in
            actions.push_back(-1);
        }
    }

    return actions;
}

// Helper: apply abstract action to state
static void apply_action_to_state(MCCFRState& state, int action, int player) {
    int current_bet = 0;
    for (int i = 0; i < state.num_players; i++) {
        current_bet = std::max(current_bet, state.player_bets[i]);
    }
    int to_call = current_bet - state.player_bets[player];

    // Record action in history (with bounds check)
    if (state.action_history_len < 511) {
        state.action_history[state.action_history_len++] = action;
    }
    state.actions_this_round++;

    if (action == 0) {
        // Fold
        state.player_folded[player] = true;
    } else if (action == 1) {
        // Call/Check
        int actual = std::min(to_call, state.player_stacks[player]);
        state.player_stacks[player] -= actual;
        state.player_bets[player] += actual;
        state.pot += actual;
        if (state.player_stacks[player] == 0) {
            state.player_all_in[player] = true;
        }
    } else if (action == -1) {
        // All-in
        int actual = state.player_stacks[player];
        state.player_stacks[player] = 0;
        state.player_bets[player] += actual;
        state.pot += actual;
        state.player_all_in[player] = true;
        state.last_raiser = player;
        state.raises_this_round++;
    } else {
        // Raise to amount 'action' (action >= 100)
        int raise_to = action;
        int chips_needed = raise_to - state.player_bets[player];
        int actual = std::min(chips_needed, state.player_stacks[player]);
        state.player_stacks[player] -= actual;
        state.player_bets[player] += actual;
        state.pot += actual;
        state.last_raiser = player;
        state.raises_this_round++;
        if (state.player_stacks[player] == 0) {
            state.player_all_in[player] = true;
        }
    }

    // Count active players (can still act)
    int active = 0;
    for (int i = 0; i < state.num_players; i++) {
        if (!state.player_folded[i] && !state.player_all_in[i]) {
            active++;
        }
    }

    // Count players still in hand
    int in_hand = 0;
    for (int i = 0; i < state.num_players; i++) {
        if (!state.player_folded[i]) in_hand++;
    }

    // Only one player left -> hand over
    if (in_hand <= 1) {
        state.betting_round = 4;
        return;
    }

    // No active players (everyone all-in or folded) -> run out board
    if (active == 0) {
        state.betting_round = 4;
        state.board_len = 5;
        return;
    }

    // Find next active player
    int next = player;
    bool found_next = false;
    for (int k = 0; k < state.num_players; k++) {
        next = (next + 1) % state.num_players;
        if (!state.player_folded[next] && !state.player_all_in[next]) {
            found_next = true;
            break;
        }
    }

    if (!found_next || active <= 1) {
        int max_bet = 0;
        for (int i = 0; i < state.num_players; i++) {
            max_bet = std::max(max_bet, state.player_bets[i]);
        }
        bool needs_action = false;
        for (int i = 0; i < state.num_players; i++) {
            if (!state.player_folded[i] && !state.player_all_in[i]) {
                if (state.player_bets[i] < max_bet) {
                    needs_action = true;
                    break;
                }
            }
        }
        if (!needs_action) {
            if (active <= 1) {
                state.betting_round = 4;
                state.board_len = 5;
                return;
            }
        }
    }

    // Check if betting round is over
    int new_max_bet = 0;
    for (int i = 0; i < state.num_players; i++) {
        new_max_bet = std::max(new_max_bet, state.player_bets[i]);
    }

    bool all_matched = true;
    for (int i = 0; i < state.num_players; i++) {
        if (!state.player_folded[i] && !state.player_all_in[i]) {
            if (state.player_bets[i] != new_max_bet) {
                all_matched = false;
                break;
            }
        }
    }

    bool round_over = false;
    if (all_matched) {
        if (state.last_raiser >= 0) {
            round_over = (next == state.last_raiser);
        } else {
            round_over = (state.actions_this_round >= active);
        }
    }

    if (round_over) {
        state.betting_round++;

        if (state.betting_round == 1) state.board_len = 3;
        else if (state.betting_round == 2) state.board_len = 4;
        else if (state.betting_round == 3) state.board_len = 5;

        for (int i = 0; i < state.num_players; i++) {
            state.player_bets[i] = 0;
        }

        state.actions_this_round = 0;
        state.raises_this_round = 0;
        state.last_raiser = -1;

        for (int i = 0; i < state.num_players; i++) {
            int p = (i + 1) % state.num_players;
            if (!state.player_folded[p] && !state.player_all_in[p]) {
                state.current_player = p;
                state.round_starter = p;
                return;
            }
        }
        state.betting_round = 4;
        state.board_len = 5;
    } else {
        state.current_player = next;
    }
}

} // namespace gambletron
