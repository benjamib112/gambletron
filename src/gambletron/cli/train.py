"""Training CLI: run blueprint MCCFR computation."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gambletron Blueprint Training")
    parser.add_argument(
        "-n", "--iterations", type=int, default=1_000_000,
        help="Total MCCFR iterations to reach (default: 1M)"
    )
    parser.add_argument(
        "-p", "--players", type=int, default=6,
        help="Number of players (2-6)"
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=0,
        help="Number of threads (0=auto-detect, default: 0)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="models/blueprint.pkl",
        help="Output file for final trained strategy"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Directory for all outputs (snapshots, checkpoints)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=10000,
        help="Iterations between progress reports and checkpoints"
    )
    parser.add_argument(
        "--no-snapshots", action="store_true",
        help="Disable automatic difficulty snapshots"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start from scratch even if a checkpoint exists"
    )

    args = parser.parse_args()

    # Auto-detect thread count
    num_threads = args.threads
    if num_threads <= 0:
        num_threads = os.cpu_count() or 1
        num_threads = min(num_threads, 8)

    print("=" * 60)
    print("  GAMBLETRON - Blueprint Strategy Training")
    print("=" * 60)
    print(f"  Players:    {args.players}")
    print(f"  Target:     {args.iterations:,} iterations")
    print(f"  Threads:    {num_threads}")
    print(f"  Output:     {args.output}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)

    try:
        from gambletron.ai.blueprint import BlueprintTrainer
        from gambletron.players.ai import (
            DIFFICULTY_SNAPSHOT_TARGETS,
            Difficulty,
        )
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("Make sure the C++ engine is built:")
        print("  cd build && cmake .. && make -j$(nproc)")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.bin"

    trainer = BlueprintTrainer(
        num_players=args.players,
        num_threads=num_threads,
    )

    # Resume from checkpoint if available
    resumed = False
    if not args.no_resume and checkpoint_path.exists():
        try:
            trainer.load_checkpoint(str(checkpoint_path))
            prev_iters = trainer.trainer.iterations_done()
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"\n  Resuming from checkpoint: {prev_iters:,} iterations, "
                  f"{trainer.trainer.num_infosets():,} infosets ({size_mb:.1f} MB)")
            resumed = True
        except Exception as e:
            print(f"\n  Warning: failed to load checkpoint ({e}), starting fresh")

    prev_iters = trainer.trainer.iterations_done()
    remaining = args.iterations - prev_iters

    if remaining <= 0:
        print(f"\n  Already at {prev_iters:,} iterations (target: {args.iterations:,})")
        print("  Use a higher -n value to train further.")
        print(f"  Checkpoint at: {checkpoint_path}")
        return

    # Configure difficulty snapshots
    snapshot_points = None
    if not args.no_snapshots:
        snapshot_points = {}
        for diff in Difficulty:
            target = DIFFICULTY_SNAPSHOT_TARGETS[diff]
            if target <= args.iterations:
                snapshot_points[diff.value] = target
        if snapshot_points:
            print("\n  Difficulty snapshots:")
            for name, iters in sorted(snapshot_points.items(), key=lambda x: x[1]):
                status = "done" if prev_iters >= iters else "pending"
                print(f"    {name:12s} -> iteration {iters:>12,}  [{status}]")
            print()

    if resumed:
        print(f"\n  Training {remaining:,} more iterations "
              f"({prev_iters:,} -> {args.iterations:,})...")
    else:
        print(f"\nStarting training...")

    start = time.time()

    strategy = trainer.train(
        num_iterations=remaining,
        checkpoint_dir=str(output_dir),
        checkpoint_interval=args.checkpoint_interval,
        snapshot_points=snapshot_points,
        verbose=True,
    )

    elapsed = time.time() - start
    total_done = trainer.trainer.iterations_done()
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Total iterations: {total_done:,}")
    print(f"  This session:     {remaining:,} iterations at {remaining / elapsed:,.0f} it/s")
    print(f"  Infosets:         {trainer.trainer.num_infosets():,}")

    # Save final checkpoint
    trainer.save_checkpoint(str(checkpoint_path))
    cp_size = checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else 0
    print(f"\nCheckpoint saved to {checkpoint_path} ({cp_size:.1f} MB)")
    print("  (run the same command again to resume training)")

    if snapshot_points:
        print("\nDifficulty snapshots:")
        for name in sorted(snapshot_points.keys()):
            p = output_dir / f"{name}.bin"
            if p.exists():
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  {name:12s} -> {p} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
