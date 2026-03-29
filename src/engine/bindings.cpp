#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "hand_eval.h"
#include "mccfr.h"

namespace py = pybind11;

PYBIND11_MODULE(gambletron_engine, m) {
    m.doc() = "Gambletron C++ engine: fast hand evaluation and MCCFR";

    // Hand evaluation
    m.def(
        "fast_evaluate_hand",
        [](const std::vector<int>& cards) -> uint32_t {
            return gambletron::evaluate_hand(cards);
        },
        py::arg("cards"),
        "Evaluate the best 5-card poker hand from a list of card integers.");

    m.def(
        "hand_category",
        [](uint32_t score) -> int {
            return static_cast<int>(gambletron::get_category(score));
        },
        py::arg("score"),
        "Extract the hand category from a score.");

    // MCCFR Config
    py::class_<gambletron::MCCFRConfig>(m, "MCCFRConfig")
        .def(py::init<>())
        .def_readwrite("num_players", &gambletron::MCCFRConfig::num_players)
        .def_readwrite("num_iterations", &gambletron::MCCFRConfig::num_iterations)
        .def_readwrite("num_threads", &gambletron::MCCFRConfig::num_threads)
        .def_readwrite("discount_interval", &gambletron::MCCFRConfig::discount_interval)
        .def_readwrite("lcfr_threshold", &gambletron::MCCFRConfig::lcfr_threshold)
        .def_readwrite("prune_threshold", &gambletron::MCCFRConfig::prune_threshold)
        .def_readwrite("prune_floor", &gambletron::MCCFRConfig::prune_floor)
        .def_readwrite("regret_floor", &gambletron::MCCFRConfig::regret_floor)
        .def_readwrite("strategy_interval", &gambletron::MCCFRConfig::strategy_interval)
        .def_readwrite("snapshot_start", &gambletron::MCCFRConfig::snapshot_start)
        .def_readwrite("snapshot_interval", &gambletron::MCCFRConfig::snapshot_interval);

    // MCCFR Trainer
    py::class_<gambletron::MCCFRTrainer>(m, "MCCFRTrainer")
        .def(py::init<const gambletron::MCCFRConfig&, gambletron::InfosetKeyFn>(),
             py::arg("config"), py::arg("key_fn"),
             "Create trainer with Python key function (single-threaded only)")
        .def(py::init<const gambletron::MCCFRConfig&>(),
             py::arg("config"),
             "Create trainer with built-in C++ key function (supports multi-threading)")
        .def("train", &gambletron::MCCFRTrainer::train,
             py::arg("num_iterations"),
             py::call_guard<py::gil_scoped_release>(),
             "Run MCCFR training for the given number of iterations.")
        .def("iterations_done", &gambletron::MCCFRTrainer::iterations_done)
        .def("save_checkpoint", &gambletron::MCCFRTrainer::save_checkpoint,
             py::arg("path"),
             "Save full training state (regrets + avg strategy) to binary file.")
        .def("load_checkpoint", &gambletron::MCCFRTrainer::load_checkpoint,
             py::arg("path"),
             "Load training state from binary checkpoint file.")
        .def("num_infosets", [](const gambletron::MCCFRTrainer& t) {
            return t.get_store().size();
        })
        .def("get_raw_data", [](const gambletron::MCCFRTrainer& t, uint64_t key) -> py::dict {
            auto* data = const_cast<gambletron::InfosetStore&>(t.get_store()).get(key);
            if (!data) return py::dict();
            py::dict result;
            py::list regrets, avg_strat;
            for (int i = 0; i < data->num_actions; i++) {
                regrets.append(data->regrets[i]);
                avg_strat.append(data->avg_strategy[i]);
            }
            result["regrets"] = regrets;
            result["avg_strategy"] = avg_strat;
            result["num_actions"] = data->num_actions;
            return result;
        })
        .def("get_strategy", [](const gambletron::MCCFRTrainer& t,
                                 uint64_t key) -> std::vector<float> {
            const float* strat = t.get_strategy(key);
            if (!strat) return {};
            auto* data = const_cast<gambletron::InfosetStore&>(t.get_store()).get(key);
            if (!data) return {};
            return std::vector<float>(strat, strat + data->num_actions);
        })
        .def("get_average_strategy", &gambletron::MCCFRTrainer::get_average_strategy)
        .def("get_all_strategies", [](const gambletron::MCCFRTrainer& t) {
            py::dict result;
            t.get_store().for_each([&](gambletron::InfosetKey key, const gambletron::InfosetData& data) {
                py::list probs;
                float sum = 0;
                for (int i = 0; i < data.num_actions; i++) {
                    sum += std::max(data.regrets[i], (int32_t)0);
                }
                for (int i = 0; i < data.num_actions; i++) {
                    float p = sum > 0 ? std::max(data.regrets[i], (int32_t)0) / sum
                                      : 1.0f / data.num_actions;
                    probs.append(p);
                }
                result[py::int_(key)] = probs;
            });
            return result;
        });
}
