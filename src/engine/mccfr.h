#pragma once

#include "game_tree.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <random>
#include <thread>
#include <vector>

namespace gambletron {

// Configuration matching Pluribus paper parameters
struct MCCFRConfig {
    int num_players = 6;
    int num_iterations = 1000;
    int num_threads = 1;           // Number of traverser threads

    // Linear CFR: discount regrets every discount_interval iterations
    // by factor t/(t+1). Stop after lcfr_threshold iterations.
    int discount_interval = 100;   // Paper: every 10 minutes
    int lcfr_threshold = 4000;     // Paper: first 400 minutes

    // Pruning: after prune_threshold iterations, 95% of iterations
    // skip actions with regret below prune_floor
    int prune_threshold = 2000;    // Paper: 200 minutes
    int32_t prune_floor = -300000000;
    int32_t regret_floor = -310000000;

    // Strategy update interval (for preflop average strategy)
    int strategy_interval = 10000;

    // Snapshot interval (for postflop strategy averaging)
    int snapshot_start = 8000;     // Paper: 800 minutes
    int snapshot_interval = 2000;  // Paper: 200 minutes
};

// Callback types for abstraction lookup
using InfosetKeyFn = std::function<InfosetKey(
    int player,                          // acting player
    int betting_round,                   // 0-3
    std::vector<int> hole_cards,         // player's 2 hole cards
    std::vector<int> board,              // community cards (0-5)
    int board_len,
    std::vector<int> action_seq,         // action sequence
    int action_seq_len
)>;

// Abstract game state for MCCFR traversal
struct MCCFRState {
    int pot;
    int betting_round;
    int current_player;
    int num_players;
    bool player_folded[6];
    bool player_all_in[6];
    int player_bets[6];    // Bets this round
    int player_stacks[6];
    int hole_cards[6][2];  // 2 cards per player
    int board[5];
    int board_len;

    // Action history as flat array
    int action_history[512];
    int action_history_len;

    // Round tracking
    int actions_this_round;
    int raises_this_round; // Number of raises in current round
    int last_raiser;       // -1 if no raise this round
    int round_starter;     // first player to act this round

    // Available actions at current node
    std::vector<int> available_actions;
};

// Built-in C++ infoset key function (no Python callback needed)
InfosetKey builtin_infoset_key(
    int player, int betting_round,
    const int* hole_cards, int hole_len,
    const int* board, int board_len,
    const int* action_seq, int action_seq_len);

class MCCFRTrainer {
public:
    // Constructor with Python callback (single-threaded only)
    MCCFRTrainer(const MCCFRConfig& config, InfosetKeyFn key_fn);

    // Constructor using built-in C++ key function (supports multi-threading)
    MCCFRTrainer(const MCCFRConfig& config);

    // Run MCCFR training
    void train(int num_iterations);

    // Get the infoset store (for serialization)
    InfosetStore& get_store() { return store_; }
    const InfosetStore& get_store() const { return store_; }

    // Get current strategy at an infoset
    const float* get_strategy(InfosetKey key) const;

    // Get average strategy at an infoset (for preflop)
    std::vector<float> get_average_strategy(InfosetKey key) const;

    // Save/load full training state (regrets + avg strategy)
    void save_checkpoint(const std::string& path) const;
    void load_checkpoint(const std::string& path);

    // Set iteration count (for resuming)
    void set_iterations_done(int64_t n) { iterations_done_ = n; }

    // Stats
    int64_t iterations_done() const { return iterations_done_; }

private:
    // Per-thread traversal context
    struct ThreadContext {
        std::mt19937 rng;
    };

    // External-sampling MCCFR traversal
    float traverse_mccfr(MCCFRState& state, int traverser, ThreadContext& ctx);

    // Pruned version
    float traverse_mccfr_pruned(MCCFRState& state, int traverser, ThreadContext& ctx);

    // Update average strategy (preflop only)
    void update_strategy(MCCFRState& state, int player, ThreadContext& ctx);

    // Deal cards and create initial state
    MCCFRState create_initial_state(ThreadContext& ctx);

    // Get terminal payoffs
    void get_payoffs(const MCCFRState& state, float* payoffs);

    // Compute infoset key (uses built-in or Python callback)
    InfosetKey compute_key(const MCCFRState& state, int player);

    // Single-threaded training iteration
    void run_iteration(int total_iter, ThreadContext& ctx);

    MCCFRConfig config_;
    InfosetKeyFn key_fn_;    // Python callback (may be empty)
    bool use_builtin_key_;   // True if using C++ key function
    InfosetStore store_;
    std::atomic<int64_t> iterations_done_{0};
};

} // namespace gambletron
