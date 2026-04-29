"""Physical table session: run poker on the real hardware.

GPIO dealer, HID RFID readers, serial chip controller, touchscreen display.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from gambletron.players.ai import AIPlayer, Difficulty
from gambletron.poker.table import Table


def main() -> None:
    parser = argparse.ArgumentParser(description="Gambletron Physical Table")
    parser.add_argument(
        "-n", "--num-players", type=int, default=6,
        help="Number of players (2-6)"
    )
    parser.add_argument(
        "--hands", type=int, default=0,
        help="Number of hands to play (0 = unlimited)"
    )
    parser.add_argument(
        "--stack", type=int, default=10000,
        help="Starting stack"
    )
    parser.add_argument(
        "--blueprint", type=str, default=None,
        help="Path to blueprint strategy file"
    )
    parser.add_argument(
        "-d", "--difficulty", type=str, default="expert",
        choices=["easy", "medium", "hard", "expert", "superhuman"],
        help="AI difficulty level (default: expert)"
    )
    parser.add_argument(
        "--models-dir", type=str, default="models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--chip-port", type=str, default="/dev/ttyUSB0",
        help="Serial port for the chip controller Arduino (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--asset-dir", type=str, default="assets/cards",
        help="Directory containing card PNG files"
    )
    parser.add_argument(
        "--windowed", action="store_true",
        help="Run display in a window (for testing)"
    )

    args = parser.parse_args()
    difficulty = Difficulty(args.difficulty)
    fullscreen = not args.windowed

    # Load blueprint
    blueprint_path = args.blueprint
    if blueprint_path is None:
        for ext in (".bin", ".pkl"):
            snapshot = Path(args.models_dir) / f"{difficulty.value}{ext}"
            if snapshot.exists():
                blueprint_path = str(snapshot)
                break
        if blueprint_path is None:
            for ext in (".bin", ".pkl"):
                main_bp = Path(args.models_dir) / f"blueprint{ext}"
                if main_bp.exists():
                    blueprint_path = str(main_bp)
                    break

    blueprint = None
    if blueprint_path and Path(blueprint_path).exists():
        if blueprint_path.endswith(".bin"):
            from gambletron.ai.strategy import CheckpointStrategy
            blueprint = CheckpointStrategy.from_file(blueprint_path)
        else:
            from gambletron.ai.strategy import Strategy
            blueprint = Strategy.from_file(blueprint_path)
        print(f"Loaded blueprint with {len(blueprint)} infosets")
    else:
        print("No blueprint found — AI plays uniform random")

    # Start display with ready-button support
    from gambletron.display.process import start_display_process
    from gambletron.display.sink import QueueDisplaySink

    _display_proc, _queue, ready_event = start_display_process(
        asset_dir=args.asset_dir,
        fullscreen=fullscreen,
        with_ready_button=True,
    )
    display_sink = QueueDisplaySink(_queue)

    # Connect physical table hardware
    from gambletron.hardware.physical_table import PhysicalTableController

    controller = PhysicalTableController(
        chip_port=args.chip_port,
        num_seats=args.num_players,
    )
    controller.connect()
    print(f"Physical table connected (chips={args.chip_port})")

    # Create AI players for all seats
    players = [
        AIPlayer(name=f"AI_{i}", blueprint=blueprint, difficulty=difficulty)
        for i in range(args.num_players)
    ]

    table = Table(
        players=players,
        starting_stack=args.stack,
        display_sink=display_sink,
        table_controller=controller,
    )

    print(f"\nGambletron Physical Table")
    print(f"Players: {args.num_players}, Difficulty: {difficulty.value}")
    print(f"Press the DEAL button on screen to start each hand.\n")

    hand_num = 0
    max_hands = args.hands if args.hands > 0 else float("inf")

    try:
        while hand_num < max_hands:
            # Show ready button and wait for touchscreen press
            display_sink.show_ready_button()
            ready_event.clear()
            ready_event.wait()
            display_sink.hide_ready_button()

            # Trigger physical dealer (GPIO pulse + wait for RFID reads)
            try:
                cards = controller.trigger_deal()
            except RuntimeError as e:
                print(f"Deal error: {e}")
                print("Retrying — return cards and press DEAL again.")
                continue

            hand_num += 1

            # Play the hand
            changes = table.play_hand()

            if hand_num % 10 == 0 or hand_num == max_hands:
                print(f"Hand {hand_num}: {[f'{c:+d}' for c in changes]}")

        print(f"\nSession complete after {hand_num} hands.")
        for i, p in enumerate(players):
            print(f"  {p.name}: {table.total_results[i]:+d} chips (stack: ${table.stacks[i]})")
    except KeyboardInterrupt:
        print(f"\nSession ended after {hand_num} hands.")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
