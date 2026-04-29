"""Interactive digital demo: play poker against AI agents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from gambletron.players.ai import AIPlayer, Difficulty
from gambletron.players.human import HumanPlayer
from gambletron.players.random_player import RandomPlayer
from gambletron.poker.table import Table


def run_demo(
    num_players: int = 6,
    human_seat: int = 0,
    num_hands: int = 10,
    starting_stack: int = 10000,
    blueprint_path: Optional[str] = None,
    difficulty: Difficulty = Difficulty.EXPERT,
    models_dir: str = "models",
    display: bool = True,
    asset_dir: str = "assets/cards",
    fullscreen: bool = True,
) -> None:
    """Run an interactive poker demo."""
    print("=" * 60)
    print("  GAMBLETRON - Pluribus Poker AI Demo")
    print("=" * 60)
    print(f"  Players: {num_players}")
    print(f"  Your seat: {human_seat}")
    print(f"  Starting stack: ${starting_stack}")
    print(f"  Blinds: $50/$100")
    print(f"  Difficulty: {difficulty.value}")

    # Auto-detect blueprint: use explicit path, or find difficulty snapshot
    if blueprint_path is None:
        # Prefer .bin (checkpoint) over .pkl (legacy)
        for ext in (".bin", ".pkl"):
            snapshot = Path(models_dir) / f"{difficulty.value}{ext}"
            if snapshot.exists():
                blueprint_path = str(snapshot)
                break
        if blueprint_path is None:
            for ext in (".bin", ".pkl"):
                main_bp = Path(models_dir) / f"blueprint{ext}"
                if main_bp.exists():
                    blueprint_path = str(main_bp)
                    break

    if blueprint_path:
        print(f"  Blueprint: {blueprint_path}")
    else:
        print("  Blueprint: None (AI plays uniform random)")
    print(f"  Display: {'enabled' if display else 'disabled'}")
    print("=" * 60)

    # Load blueprint if available
    blueprint = None
    if blueprint_path and Path(blueprint_path).exists():
        if blueprint_path.endswith(".bin"):
            from gambletron.ai.strategy import CheckpointStrategy
            blueprint = CheckpointStrategy.from_file(blueprint_path)
        else:
            from gambletron.ai.strategy import Strategy
            blueprint = Strategy.from_file(blueprint_path)
        print(f"Loaded blueprint with {len(blueprint)} infosets")

    # Start display process (if enabled)
    display_sink = None
    _display_proc = None
    if display:
        try:
            from gambletron.display.process import start_display_process
            from gambletron.display.sink import QueueDisplaySink
            _display_proc, _queue, _ = start_display_process(
                asset_dir=asset_dir, fullscreen=fullscreen,
            )
            display_sink = QueueDisplaySink(_queue)
            print("Display process started.")
        except Exception as e:
            print(f"Warning: could not start display ({e}). Continuing without.")

    # Create players
    players = []
    for i in range(num_players):
        if i == human_seat:
            players.append(HumanPlayer(name="You"))
        else:
            players.append(
                AIPlayer(
                    name=f"AI_{i}",
                    blueprint=blueprint,
                    difficulty=difficulty,
                )
            )

    # Run the game
    table = Table(
        players=players,
        starting_stack=starting_stack,
        display_sink=display_sink,
    )

    for hand_num in range(1, num_hands + 1):
        print(f"\n{'#'*50}")
        print(f"  Hand {hand_num}/{num_hands}")
        print(f"{'#'*50}")

        changes = table.play_hand()

        print(f"\nResults for hand {hand_num}:")
        for i, p in enumerate(players):
            marker = " <-- YOU" if i == human_seat else ""
            print(f"  {p.name}: {changes[i]:+d} (stack: ${table.stacks[i]}){marker}")

    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for i, p in enumerate(players):
        marker = " <-- YOU" if i == human_seat else ""
        print(f"  {p.name}: {table.total_results[i]:+d} chips{marker}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gambletron Poker Demo")
    parser.add_argument(
        "-n", "--num-players", type=int, default=6, help="Number of players (2-6)"
    )
    parser.add_argument(
        "-s", "--seat", type=int, default=0, help="Your seat number (0-indexed)"
    )
    parser.add_argument(
        "--hands", type=int, default=10, help="Number of hands to play"
    )
    parser.add_argument(
        "--stack", type=int, default=10000, help="Starting stack"
    )
    parser.add_argument(
        "--blueprint", type=str, default=None, help="Path to blueprint strategy file"
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
        "--ai-only", action="store_true", help="Watch AI vs AI (no human player)"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable the pygame display"
    )
    parser.add_argument(
        "--windowed", action="store_true", help="Run display in a window (for testing)"
    )
    parser.add_argument(
        "--asset-dir", type=str, default="assets/cards",
        help="Directory containing card PNG files (default: assets/cards)"
    )

    args = parser.parse_args()
    difficulty = Difficulty(args.difficulty)

    use_display = not args.no_display
    fullscreen  = not args.windowed

    if args.ai_only:
        run_ai_only(
            args.num_players, args.hands, args.stack, args.blueprint, difficulty,
            display=use_display, asset_dir=args.asset_dir, fullscreen=fullscreen,
        )
    else:
        run_demo(
            num_players=args.num_players,
            human_seat=args.seat,
            num_hands=args.hands,
            starting_stack=args.stack,
            blueprint_path=args.blueprint,
            difficulty=difficulty,
            models_dir=args.models_dir,
            display=use_display,
            asset_dir=args.asset_dir,
            fullscreen=fullscreen,
        )


def run_ai_only(
    num_players: int,
    num_hands: int,
    starting_stack: int,
    blueprint_path: Optional[str],
    difficulty: Difficulty = Difficulty.EXPERT,
    display: bool = False,
    asset_dir: str = "assets/cards",
    fullscreen: bool = True,
) -> None:
    """Run AI vs AI game (no human interaction)."""
    print("AI vs AI game")
    print(f"Players: {num_players}, Hands: {num_hands}, Difficulty: {difficulty.value}")

    blueprint = None
    if blueprint_path and Path(blueprint_path).exists():
        if blueprint_path.endswith(".bin"):
            from gambletron.ai.strategy import CheckpointStrategy
            blueprint = CheckpointStrategy.from_file(blueprint_path)
        else:
            from gambletron.ai.strategy import Strategy
            blueprint = Strategy.from_file(blueprint_path)

    display_sink = None
    if display:
        try:
            from gambletron.display.process import start_display_process
            from gambletron.display.sink import QueueDisplaySink
            _, _queue, _ = start_display_process(asset_dir=asset_dir, fullscreen=fullscreen)
            display_sink = QueueDisplaySink(_queue)
        except Exception as e:
            print(f"Warning: could not start display ({e}). Continuing without.")

    players = [
        AIPlayer(name=f"AI_{i}", blueprint=blueprint, difficulty=difficulty)
        for i in range(num_players)
    ]

    table = Table(players=players, starting_stack=starting_stack, display_sink=display_sink)

    for hand_num in range(1, num_hands + 1):
        changes = table.play_hand()
        if hand_num % 100 == 0 or hand_num == num_hands:
            print(f"Hand {hand_num}: {[f'{c:+d}' for c in changes]}")

    print(f"\nFinal results after {num_hands} hands:")
    for i, p in enumerate(players):
        print(f"  {p.name}: {table.total_results[i]:+d} chips (stack: ${table.stacks[i]})")


if __name__ == "__main__":
    main()
