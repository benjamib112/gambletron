Gambletron: Pluribus Poker AI for Automated Physical Table

 Context

 A complete Pluribus-style poker AI system for a physical automated poker table. The system:
 1. Works as a standalone digital demo (training + play without hardware)
 2. Connects to a physical table (Raspberry Pi 5 with GPIO dealer, HID RFID readers, serial chip controller)
 3. Supports 1-6 player no-limit Texas Hold'em where AI fills empty seats

 The AI implements the Pluribus algorithm: offline blueprint computation via Monte Carlo CFR, then real-time depth-limited subgame solving during play.

 Physical Table Architecture

 The physical table runs on a Raspberry Pi 5 with:
 - GPIO dealer: A card dealer machine wired to GPIO pins 5 and 6. A 200ms active-high pulse triggers shuffle + deal (12 cards, 2 to each of 6 seats sequentially).
 - HID RFID readers: 6 USB RFID readers (one per seat) on a 6-way USB splitter. Each reader presents as an HID keyboard device — it types the UID digits and presses Enter. Cards have dual RFID chips (2 UIDs per card, 104 total, hardcoded in protocol.py).
 - Serial chip controller: Arduino managing physical chips, connected via USB serial.
 - Touchscreen display: The pygame display runs on a touchscreen. A "DEAL" button appears between hands for players to trigger the next deal.
 - Community cards are virtual: Only hole cards are dealt physically. Community cards (flop/turn/river) are generated in software from the remaining 40 cards (excluding the 12 dealt).

 Physical hand flow:
 1. "DEAL" button shown on touchscreen after hand ends
 2. Players return all cards to the dealer machine
 3. Player taps "DEAL" on touchscreen
 4. RPi pulses GPIO → dealer shuffles and deals 12 cards (~6 seconds)
 5. RFID readers detect cards as they slide over (buffered, ~8s timeout)
 6. Game plays out: AI uses physical cards, community cards generated virtually
 7. Showdown / fold-out → back to step 1

 Language & Tech Stack

 - Python: Game framework, hardware interface, orchestration, CLI
 - C++ (via pybind11): Performance-critical MCCFR engine, hand evaluation
 - Build: CMake for C++, pip/setuptools for Python bindings
 - Dependencies: NumPy, pybind11, scikit-learn (k-means clustering), pyserial (chip controller), python-evdev (HID readers), gpiod (GPIO on RPi 5), pygame (display)

 Project Structure

 gambletron/
 ├── CMakeLists.txt
 ├── pyproject.toml
 ├── setup.py                      # pybind11 build integration
 │
 ├── src/
 │   ├── engine/                    # C++ core (compiled to shared lib)
 │   │   ├── hand_eval.cpp/.h       # Fast poker hand evaluation (7-card)
 │   │   ├── mccfr.cpp/.h           # External-sampling MCCFR with Linear CFR + pruning
 │   │   ├── subgame_solver.cpp/.h  # Real-time depth-limited subgame solving
 │   │   ├── game_tree.cpp/.h       # Abstract game tree representation
 │   │   ├── abstraction.cpp/.h     # Info/action abstraction bucket lookup
 │   │   └── bindings.cpp           # pybind11 module exposing C++ to Python
 │   │
 │   └── gambletron/                # Python package
 │       ├── __init__.py
 │       │
 │       ├── poker/                 # Poker framework (pure Python)
 │       │   ├── __init__.py
 │       │   ├── card.py            # Card, Deck, Suit, Rank
 │       │   ├── hand.py            # Hand evaluation (Python fallback + C++ binding)
 │       │   ├── state.py           # GameState: pot, bets, community cards, player states
 │       │   ├── game.py            # Game: runs a full hand (accepts optional TableController)
 │       │   ├── table.py           # Table: manages multi-hand sessions, passes controller to Game
 │       │   └── rules.py           # NLTH rules: blinds, betting logic, raise validation
 │       │
 │       ├── players/               # Player abstractions
 │       │   ├── __init__.py
 │       │   ├── base.py            # Player ABC: get_action(game_state) -> Action
 │       │   ├── human.py           # HumanPlayer: CLI input for digital demo
 │       │   ├── random_player.py   # RandomPlayer: for testing
 │       │   └── ai.py              # AIPlayer: Pluribus agent (blueprint + search)
 │       │
 │       ├── ai/                    # Pluribus AI implementation
 │       │   ├── __init__.py
 │       │   ├── blueprint.py       # Blueprint training orchestration (calls C++ MCCFR)
 │       │   ├── search.py          # Real-time search orchestration (calls C++ solver)
 │       │   ├── abstraction.py     # Information abstraction (card bucketing via k-means)
 │       │   ├── action_abstraction.py  # Action abstraction (bet sizing)
 │       │   ├── belief.py          # Belief/range tracking (1326 hand probabilities)
 │       │   └── strategy.py        # Strategy storage, loading, serialization
 │       │
 │       ├── hardware/              # Physical table interface
 │       │   ├── __init__.py
 │       │   ├── interface.py       # ABCs: CardInput, ChipInterface, SeatSensor, TableController
 │       │   ├── simulated.py       # SimulatedHardware: for training & digital demo
 │       │   ├── gpio_dealer.py     # GPIODealer: pulses GPIO 5+6 to trigger physical dealer
 │       │   ├── hid_reader.py      # HIDCardReader/Pool: reads RFID UIDs from HID devices via evdev
 │       │   ├── physical_table.py  # PhysicalTableController: composes GPIO + HID + serial chips
 │       │   ├── serial_comm.py     # SerialConnection, SerialChipInterface, SerialSeatSensor
 │       │   └── protocol.py        # Serial protocol + hardcoded RFID UID-to-card mapping (104 UIDs)
 │       │
 │       ├── display/               # Pygame table display
 │       │   ├── __init__.py
 │       │   ├── card_loader.py     # Loads card PNG assets
 │       │   ├── events.py          # Dataclass events (HandStart, Action, ShowReadyButton, etc.)
 │       │   ├── process.py         # Display subprocess (multiprocessing.Queue + Event for ready button)
 │       │   ├── renderer.py        # PygameRenderer: table, seats, cards, winner banner, DEAL button
 │       │   └── sink.py            # NullDisplaySink + QueueDisplaySink (game → display process)
 │       │
 │       └── cli/                   # Command-line interfaces
 │           ├── __init__.py
 │           ├── demo.py            # Software-only demo (human vs AI, AI vs AI, no hardware)
 │           ├── play.py            # Physical table session (GPIO + RFID + touchscreen)
 │           └── train.py           # Training CLI (run blueprint computation)
 │
 └── tests/
     ├── test_card.py
     ├── test_hand_eval.py
     ├── test_game.py
     ├── test_mccfr.py
     ├── test_abstraction.py
     ├── test_search.py
     └── test_physical.py           # Controller path, RFID mapping, card exclusion, HID parsing, side pots

 Implementation Phases (ordered by dependency)

 Phase 1: Poker Framework

 Build the complete Texas Hold'em game engine. This is the foundation everything else depends on.

 Files: src/gambletron/poker/*, src/gambletron/players/base.py, src/gambletron/players/random_player.py

 1. Card representation (card.py): Rank (2-A), Suit (CDHS) enums, Card class, Deck class with shuffle/deal. Cards represented as integers 0-51 internally for C++ interop.
 2. Game state (state.py): GameState tracks: players in hand, community cards, pot, player chip stacks, current bets, betting round (preflop/flop/turn/river), dealer position, whose turn it is. Immutable snapshots for AI reasoning.
 3. Rules engine (rules.py): Validates legal actions (fold/call/raise), enforces min raise = $100, subsequent raises >= previous raise size, max = remaining stack. Handles blinds ($50/$100), position rotation.
 4. Hand evaluation (hand.py): Python implementation first, C++ later. Evaluate best 5-card hand from 7 cards. Standard hand rankings.
 5. Game runner (game.py): Game class runs one complete hand: post blinds, deal, run 4 betting rounds, showdown/award pot. Calls Player.get_action() for decisions. Accepts optional TableController — when present, deals via controller instead of software deck.
 6. Table manager (table.py): Table manages sessions of hands with 1-6 players, rotates dealer, tracks stack sizes across hands. Passes table_controller through to Game.
 7. Player base (players/base.py): Abstract Player class with get_action(visible_state) -> Action. Action = fold/call/raise(amount).
 8. Random player (players/random_player.py): Uniformly random legal actions. For testing.

 Phase 2: Hand Evaluation in C++

 Fast hand evaluation is critical for MCCFR performance (billions of evaluations).

 Files: src/engine/hand_eval.cpp, src/engine/bindings.cpp, CMakeLists.txt, setup.py

 1. Implement lookup-table-based 7-card hand evaluator in C++
 2. pybind11 bindings so Python can call it
 3. Python hand.py auto-detects and uses C++ version when available, falls back to pure Python

 Phase 3: Information & Action Abstraction

 Required before blueprint training can begin.

 Files: src/gambletron/ai/abstraction.py, src/gambletron/ai/action_abstraction.py, src/engine/abstraction.cpp

 1. Information abstraction (abstraction.py):
   - Preflop: lossless abstraction (169 canonical hands using suit isomorphism)
   - Flop/Turn/River (blueprint): 200 buckets per round via k-means on hand-strength features (expected hand strength, hand potential, negative potential)
   - Flop/Turn/River (search): 500 buckets per round, potential-aware with earth-mover distance
   - Feature computation uses Monte Carlo rollouts to estimate hand equity
 2. Action abstraction (action_abstraction.py):
   - Blueprint: fine-grained preflop (up to 14 raise sizes as pot fractions), coarser postflop
   - Rounds 3-4 blueprint: first raise in {0.5x, 1x, all-in}, subsequent in {1x, all-in}
   - Search: 1-6 raise sizes per decision point
   - Fold and call always included
   - Pseudo-harmonic action mapping for off-tree opponent actions

 Phase 4: Blueprint Strategy (MCCFR in C++)

 The core offline training. This is the most compute-intensive component.

 Files: src/engine/mccfr.cpp, src/engine/game_tree.cpp, src/gambletron/ai/blueprint.py, src/gambletron/ai/strategy.py

 1. Abstract game tree (game_tree.cpp): C++ representation of the abstracted poker game tree. Nodes contain action lists, infoset mapping. Lazy memory allocation (only allocate regrets when sequence first encountered).
 2. MCCFR engine (mccfr.cpp): Implements Algorithm 1 from the paper:
   - External-sampling MCCFR: TRAVERSE-MCCFR traverses game tree, one traverser per iteration
   - Linear CFR weighting: every 10-min-equivalent, discount regrets and avg strategy by T/(T+1). Stop discounting after 400 min equivalent.
   - Negative-regret pruning: after 200-min equivalent, 95% of iterations skip actions with regret < -300,000,000 (except last round or terminal). Remaining 5% explore all.
   - Regrets as 4-byte integers, floor at -310,000,000
   - UPDATE-STRATEGY: every 10,000 iterations, update average strategy (preflop only)
   - Post-flop: snapshot current strategy every 200-min-equivalent after initial 800 min, average snapshots for final blueprint
   - Multi-threaded (one traverser thread per player)
 3. Blueprint orchestration (blueprint.py): Python wrapper that configures and launches C++ MCCFR, handles checkpointing, strategy serialization.
 4. Strategy storage (strategy.py): Serialize/deserialize blueprint. Compressed format for deployment (Pi has limited memory). Maps abstract infosets to action probability distributions.

 Phase 5: Real-Time Search Engine

 The search component that improves upon the blueprint during actual play.

 Files: src/engine/subgame_solver.cpp, src/gambletron/ai/search.py, src/gambletron/ai/belief.py

 1. Belief tracking (belief.py):
   - Maintain probability distribution over 1326 possible private card pairs for each player
   - Update via Bayes' rule after each observed action using current strategy profile
   - Initialize uniformly (1/1326 each)
 2. Subgame solver (subgame_solver.cpp): Implements Algorithm 2 (nested search):
   - Subgame rooted at start of current betting round
   - Depth limits:
     - Round 1: leaf at start of round 2
     - Round 2 with >2 players: leaf at start of round 3 OR after 2nd raise, whichever earlier
     - Otherwise: solve to end of game
   - Leaf values: each player chooses among 4 continuation strategies:
     i. Unmodified blueprint
     ii. Blueprint biased toward fold (fold prob * 5, renormalize)
     iii. Blueprint biased toward call (call prob * 5, renormalize)
     iv. Blueprint biased toward raise (raise probs * 5, renormalize)
   - Continuation strategies compressed: sample one action per abstract infoset
   - Off-tree actions: add to subgame and re-solve from root
   - Freeze Pluribus's own action probs for actions already chosen (actual hand only)
   - New betting round → new subgame root, re-solve
   - Uses Linear MCCFR (large/early subgames) or vector-based Linear CFR (small/late)
   - Play final iteration strategy (not average) for unpredictability
 3. Search orchestration (search.py): Python wrapper coordinating belief updates, subgame construction, calling C++ solver, extracting action.

 Phase 6: AI Player Integration

 Wire the blueprint + search into a playable agent.

 Files: src/gambletron/players/ai.py

 1. AIPlayer implements Player.get_action():
   - Round 1 (preflop): use blueprint strategy directly (unless opponent bet is far off-tree → search)
   - Rounds 2-4: invoke real-time search
   - Maintain belief state across the hand
   - Sample action from computed strategy for actual hand

 Phase 7: Digital Demo & Physical Table CLIs

 Two entry points: software demo and physical table session.

 Files: src/gambletron/cli/demo.py, src/gambletron/cli/play.py, src/gambletron/cli/train.py

 1. Demo (demo.py): Software-only. Launch a 6-player table with human (CLI input) or AI players. Optional pygame display. No hardware dependencies.
 2. Physical table (play.py): Full hardware session. Initializes GPIO dealer, HID RFID readers, serial chip controller, and touchscreen display. Game loop: show DEAL button → wait for touch → GPIO pulse → RFID read → play hand → repeat.
 3. Training CLI (train.py): Configure and launch blueprint training, show progress, save checkpoints.

 Phase 8: Hardware Interface

 Connect to the physical table.

 Files: src/gambletron/hardware/*

 1. Abstract interfaces (interface.py):
   - CardInput: ABC with wait_for_card(seat) -> Card, wait_for_community_cards(n) -> List[Card]
   - ChipInterface: ABC with get_player_stack(seat) -> int, dispense_chips(seat, amount), collect_bet(seat, amount), collect_pot(), award_pot(seat, amount)
   - SeatSensor: ABC with get_occupied_seats() -> List[int], is_seat_occupied(seat) -> bool
   - TableController: ABC combining card + chip + seat + deal_card_to + deal_community + signal_player_turn + signal_hand_over
 2. Simulated hardware (simulated.py): Implements interfaces using a software deck. Used for training and digital demo.
 3. GPIO dealer (gpio_dealer.py): Pulses GPIO pins 5+6 high for 200ms via gpiod to trigger physical shuffle+deal.
 4. HID RFID readers (hid_reader.py): Uses python-evdev to read from 6 /dev/input/eventX devices (one per seat). Grabs exclusively so UIDs don't leak. Parses keystroke sequences (digits + Enter) into UIDs, maps to card ints via hardcoded RFIDCardMap.
 5. Physical table controller (physical_table.py): Composes GPIODealer + HIDCardReaderPool + SerialChipInterface. trigger_deal() fires GPIO and waits up to 8s for all 12 cards. Community cards generated virtually from remaining 40. deal_card_to/deal_community are no-ops (dealing already happened).
 6. Serial protocol (protocol.py): Newline-delimited JSON messages for Pi ↔ chip controller Arduino. Also contains hardcoded _RFID_MAPPING (104 UIDs → 52 cards, 2 UIDs per card due to dual-chip cards).
 7. Serial hardware (serial_comm.py): SerialConnection, SerialChipInterface, SerialSeatSensor for chip management via Arduino.

 Key Design Decisions

 1. Player ABC with get_action(visible_state): All player types (human, AI, random) implement the same interface. The game engine doesn't know or care what kind of player it's dealing with.
 2. TableController abstraction: Game accepts an optional TableController. When present, hole cards come from the controller (physical RFID or mock); when absent, cards come from the software deck. Community cards route through the same interface — physical controller generates them virtually from the remaining 40 cards.
 3. Separate entry points for demo vs physical: demo.py is hardware-free (no GPIO/evdev/serial deps). play.py handles the physical lifecycle (ready button, GPIO trigger, RFID wait, error recovery).
 4. C++ for hot paths only: Hand evaluation and MCCFR traversal are the bottleneck. Everything else stays in Python.
 5. Strategy as data: The trained blueprint is a serialized data file. The AI player loads it at startup. Separates training from play — just copy the strategy file to the Pi.
 6. Hardcoded RFID mapping: Cards have dual RFID chips producing 2 possible UIDs each. All 104 UIDs are hardcoded in protocol.py (no external config file needed at runtime).
 7. Virtual community cards: Only hole cards are dealt physically. Community cards are randomly generated from the 40 remaining cards, ensuring no duplicates with dealt cards. This avoids needing a board RFID reader.

 Verification Plan

 1. Unit tests: Card representation, hand evaluation (compare C++ vs Python), game rules (legal actions, pot calculation, showdown)
 2. Physical integration tests (test_physical.py): Game with controller path, PhysicalCardInput exclusion logic, RFID mapping completeness (all 52 cards reachable, each has exactly 2 UIDs), HID reader keystroke parsing, side pot distribution with multiple all-ins
 3. MCCFR correctness: Test on Kuhn poker where Nash equilibrium is known. Verify convergence.
 4. Integration test: Run full training on small abstraction, then play AI vs random players — AI should win significantly.
 5. Digital demo: Play interactively against AI, verify game flow.
 6. Benchmark: Measure MCCFR iterations/second, search time per decision, to ensure real-time play is feasible.

Quick start:

# Build C++ engine
cd build && cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) && make -j$(nproc) && cd ..

# Train a blueprint (small example)
PYTHONPATH=src:build python3 -m gambletron.cli.train -n 1000 -p 6 -o blueprint.pkl

# Software demo: play against AI
PYTHONPATH=src:build python3 -m gambletron.cli.demo --blueprint blueprint.pkl

# Software demo: AI vs AI
PYTHONPATH=src:build python3 -m gambletron.cli.demo --ai-only --hands 100

# Physical table session (on RPi 5)
PYTHONPATH=src:build python3 -m gambletron.cli.play --chip-port /dev/ttyUSB0

# Run tests
PYTHONPATH=src:build python3 -m pytest tests/ -v
