#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace gambletron {

enum class NodeType : uint8_t {
    CHANCE,    // Deal cards
    PLAYER,    // Player decision
    TERMINAL,  // Hand is over
};

// Compact infoset key: combines abstract bucket + action sequence hash
using InfosetKey = uint64_t;

struct GameNode {
    NodeType type;
    int player;          // Acting player (-1 for chance/terminal)
    int betting_round;   // 0-3
    int num_actions;     // Number of available actions
    int pot;             // Current pot size
    int* payoffs;        // Terminal payoffs per player (only for TERMINAL)

    // Children indexed by action
    std::vector<int> actions;        // Action identifiers
    std::vector<GameNode*> children; // Child nodes
};

// Stores regrets and strategy for a single infoset
struct InfosetData {
    int num_actions;
    int32_t* regrets;       // Cumulative regrets per action
    int32_t* avg_strategy;  // Average strategy counters (preflop only)
    float* current_strategy; // Current strategy probabilities

    InfosetData() : num_actions(0), regrets(nullptr), avg_strategy(nullptr),
                    current_strategy(nullptr) {}

    void allocate(int n_actions) {
        num_actions = n_actions;
        regrets = new int32_t[n_actions]();
        avg_strategy = new int32_t[n_actions]();
        current_strategy = new float[n_actions];
        float uniform = 1.0f / n_actions;
        for (int i = 0; i < n_actions; i++) {
            current_strategy[i] = uniform;
        }
    }

    ~InfosetData() {
        delete[] regrets;
        delete[] avg_strategy;
        delete[] current_strategy;
    }

    // No copy
    InfosetData(const InfosetData&) = delete;
    InfosetData& operator=(const InfosetData&) = delete;

    // Move
    InfosetData(InfosetData&& other) noexcept
        : num_actions(other.num_actions), regrets(other.regrets),
          avg_strategy(other.avg_strategy), current_strategy(other.current_strategy) {
        other.regrets = nullptr;
        other.avg_strategy = nullptr;
        other.current_strategy = nullptr;
    }

    void calculate_strategy() {
        float sum = 0;
        for (int i = 0; i < num_actions; i++) {
            sum += std::max(regrets[i], (int32_t)0);
        }
        if (sum > 0) {
            for (int i = 0; i < num_actions; i++) {
                current_strategy[i] = std::max(regrets[i], (int32_t)0) / sum;
            }
        } else {
            float uniform = 1.0f / num_actions;
            for (int i = 0; i < num_actions; i++) {
                current_strategy[i] = uniform;
            }
        }
    }
};

// Thread-safe sharded infoset storage with lazy allocation
class InfosetStore {
    static constexpr int NUM_SHARDS = 64;

    struct Shard {
        std::unordered_map<InfosetKey, InfosetData> store;
        mutable std::shared_mutex mutex;
    };

public:
    InfosetData& get_or_create(InfosetKey key, int num_actions) {
        auto& shard = shards_[key % NUM_SHARDS];
        // Fast path: shared lock for read
        {
            std::shared_lock<std::shared_mutex> lock(shard.mutex);
            auto it = shard.store.find(key);
            if (it != shard.store.end()) return it->second;
        }
        // Slow path: exclusive lock for insert
        std::unique_lock<std::shared_mutex> lock(shard.mutex);
        auto it = shard.store.find(key);
        if (it != shard.store.end()) return it->second;
        auto [inserted, success] = shard.store.emplace(key, InfosetData());
        inserted->second.allocate(num_actions);
        return inserted->second;
    }

    InfosetData* get(InfosetKey key) {
        auto& shard = shards_[key % NUM_SHARDS];
        auto it = shard.store.find(key);
        return it != shard.store.end() ? &it->second : nullptr;
    }

    size_t size() const {
        size_t total = 0;
        for (int i = 0; i < NUM_SHARDS; i++) {
            total += shards_[i].store.size();
        }
        return total;
    }

    // Apply Linear CFR discount to regrets only.
    // Average strategy accumulates without discounting (per Pluribus paper).
    // MUST be called with no concurrent access (between training batches).
    void discount(float factor) {
        for (int i = 0; i < NUM_SHARDS; i++) {
            for (auto& [key, data] : shards_[i].store) {
                for (int j = 0; j < data.num_actions; j++) {
                    data.regrets[j] = static_cast<int32_t>(data.regrets[j] * factor);
                }
            }
        }
    }

    // Iteration support for serialization (call only when no concurrent access)
    template<typename Fn>
    void for_each(Fn&& fn) const {
        for (int i = 0; i < NUM_SHARDS; i++) {
            for (const auto& [key, data] : shards_[i].store) {
                fn(key, data);
            }
        }
    }

    template<typename Fn>
    void for_each(Fn&& fn) {
        for (int i = 0; i < NUM_SHARDS; i++) {
            for (auto& [key, data] : shards_[i].store) {
                fn(key, data);
            }
        }
    }

private:
    Shard shards_[NUM_SHARDS];
};

} // namespace gambletron
