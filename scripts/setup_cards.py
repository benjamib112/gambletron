#!/usr/bin/env python3
"""
Download the Vector Playing Cards SVG deck and render to PNG files.

Run once before using the display (requires internet access at setup time):
    pip install cairosvg
    python scripts/setup_cards.py

Cards are saved to assets/cards/card_{int}.png  (0-51) and back.png.
Card integer encoding matches the rest of the codebase: rank*4 + suit,
where rank 0=2 ... 12=Ace and suit 0=clubs 1=diamonds 2=hearts 3=spades.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
import io
from pathlib import Path

# ── Output config ─────────────────────────────────────────────────────────────
CARD_W = 200   # PNG output width  (5:7 ratio)
CARD_H = 280   # PNG output height

OUT_DIR = Path("assets/cards")
SVG_CACHE = OUT_DIR / "_svg"

# ── Source: notpeter/Vector-Playing-Cards (Public Domain) ─────────────────────
# Individual SVG files named {rank}_of_{suit}.svg
ZIP_URL = (
    "https://github.com/notpeter/Vector-Playing-Cards"
    "/archive/refs/heads/master.zip"
)
ZIP_PREFIX = "Vector-Playing-Cards-master/"

SVG_RANKS = ["ace", "2", "3", "4", "5", "6", "7", "8", "9", "10",
             "jack", "queen", "king"]
SVG_SUITS = ["clubs", "diamonds", "hearts", "spades"]

# Map SVG rank names -> our rank index (0=2 ... 12=Ace)
_RANK_IDX = {name: i for i, name in enumerate(SVG_RANKS)}
_SUIT_IDX = {"clubs": 0, "diamonds": 1, "hearts": 2, "spades": 3}


def card_int(rank: str, suit: str) -> int:
    return _RANK_IDX[rank] * 4 + _SUIT_IDX[suit]


def convert(svg_path: Path, png_path: Path) -> bool:
    """Convert SVG -> PNG using cairosvg (preferred) or inkscape fallback."""
    try:
        import cairosvg  # type: ignore
        cairosvg.svg2png(
            url=str(svg_path.resolve()),
            write_to=str(png_path),
            output_width=CARD_W,
            output_height=CARD_H,
        )
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  cairosvg error: {e}")
        return False

    # Fallback: inkscape CLI
    import subprocess
    r = subprocess.run(
        ["inkscape", str(svg_path),
         f"--export-width={CARD_W}",
         f"--export-filename={png_path}"],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"  inkscape error: {r.stderr.decode()[:200]}")
    return r.returncode == 0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_CACHE.mkdir(exist_ok=True)

    # ── Download zip ──────────────────────────────────────────────────────────
    zip_cache = SVG_CACHE / "vector-playing-cards.zip"
    if not zip_cache.exists():
        print(f"Downloading card deck from GitHub...")
        try:
            with urllib.request.urlopen(ZIP_URL, timeout=60) as resp:
                data = resp.read()
            zip_cache.write_bytes(data)
            print(f"  Downloaded {len(data):,} bytes")
        except Exception as e:
            print(f"Download failed: {e}")
            sys.exit(1)

    # ── Extract SVGs ──────────────────────────────────────────────────────────
    print("Extracting SVG files...")
    with zipfile.ZipFile(zip_cache) as zf:
        names = zf.namelist()
        for suit in SVG_SUITS:
            for rank in SVG_RANKS:
                fname = f"{rank}_of_{suit}.svg"
                svg_dest = SVG_CACHE / fname
                if svg_dest.exists():
                    continue
                # Find matching entry in zip (may be in a subdirectory)
                matches = [n for n in names if n.endswith(fname)]
                if not matches:
                    print(f"  WARNING: {fname} not found in zip")
                    continue
                svg_dest.write_bytes(zf.read(matches[0]))

        # Back card
        back_svg = SVG_CACHE / "back.svg"
        if not back_svg.exists():
            back_matches = [n for n in names
                            if n.endswith(".svg") and "back" in n.lower()]
            if back_matches:
                back_svg.write_bytes(zf.read(back_matches[0]))

    # ── Convert to PNG ────────────────────────────────────────────────────────
    total = len(SVG_RANKS) * len(SVG_SUITS) + 1
    done = errors = skipped = 0

    for suit in SVG_SUITS:
        for rank in SVG_RANKS:
            cint = card_int(rank, suit)
            png_path = OUT_DIR / f"card_{cint}.png"
            if png_path.exists():
                skipped += 1
                continue
            svg_path = SVG_CACHE / f"{rank}_of_{suit}.svg"
            if not svg_path.exists():
                print(f"  SKIP (no SVG): {rank} of {suit}")
                errors += 1
                continue
            print(f"  {rank}_of_{suit} -> card_{cint}.png ... ", end="", flush=True)
            if convert(svg_path, png_path):
                print("OK")
                done += 1
            else:
                errors += 1

    # Back
    back_png = OUT_DIR / "back.png"
    if not back_png.exists():
        back_svg = SVG_CACHE / "back.svg"
        if back_svg.exists():
            print("  back.svg -> back.png ... ", end="", flush=True)
            if convert(back_svg, back_png):
                print("OK")
                done += 1
            else:
                errors += 1
        else:
            print("  WARNING: no back.svg found — display will use fallback card back")
    else:
        skipped += 1

    print()
    print(f"Done: {done} converted, {skipped} already existed, {errors} errors")
    if errors:
        print("\nIf conversion failed, ensure cairosvg is installed:")
        print("  pip install cairosvg")
        print("Or install inkscape:")
        print("  sudo apt install inkscape")
        sys.exit(1)


if __name__ == "__main__":
    main()
