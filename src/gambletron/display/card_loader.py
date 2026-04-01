"""Load and cache card PNG images for pygame rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pygame

# Rank/suit name mapping used by setup_cards.py when generating filenames
_RANK_NAMES = [
    "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "jack", "queen", "king", "ace",
]
_SUIT_NAMES = ["clubs", "diamonds", "hearts", "spades"]

# Suit colours for the fallback placeholder cards
_SUIT_COLORS = {
    0: (30, 30, 30),    # clubs  – near-black
    1: (180, 20, 20),   # diamonds – red
    2: (180, 20, 20),   # hearts   – red
    3: (30, 30, 30),    # spades – near-black
}
_SUIT_SYMBOLS = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
_RANK_DISPLAY = [
    "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "J", "Q", "K", "A",
]


class CardLoader:
    """Load scaled card images from the assets directory.

    Falls back to procedurally drawn placeholder cards if PNGs are missing,
    so the display works even before setup_cards.py has been run.
    """

    def __init__(self, asset_dir: Path, card_size: Tuple[int, int]) -> None:
        self.asset_dir = Path(asset_dir)
        self.card_w, self.card_h = card_size
        self._cache: Dict[int, pygame.Surface] = {}
        self._back: Optional[pygame.Surface] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, card_int: int) -> pygame.Surface:
        """Return a pygame.Surface for the given card integer (0-51)."""
        if card_int not in self._cache:
            self._cache[card_int] = self._load(card_int)
        return self._cache[card_int]

    def get_back(self) -> pygame.Surface:
        """Return the card-back image."""
        if self._back is None:
            path = self.asset_dir / "back.png"
            if path.exists():
                img = pygame.image.load(str(path)).convert_alpha()
                self._back = pygame.transform.smoothscale(
                    img, (self.card_w, self.card_h)
                )
            else:
                self._back = self._draw_back()
        return self._back

    def preload_all(self) -> None:
        """Pre-load all 52 cards and the back into the cache."""
        for i in range(52):
            self.get(i)
        self.get_back()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self, card_int: int) -> pygame.Surface:
        path = self.asset_dir / f"card_{card_int}.png"
        if path.exists():
            img = pygame.image.load(str(path)).convert_alpha()
            return pygame.transform.smoothscale(img, (self.card_w, self.card_h))
        return self._draw_placeholder(card_int)

    def _draw_placeholder(self, card_int: int) -> pygame.Surface:
        """Draw a simple card face when the PNG asset is missing."""
        w, h = self.card_w, self.card_h
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        # White card body with rounded corners
        surf.fill((0, 0, 0, 0))
        pygame.draw.rect(surf, (252, 252, 252), (0, 0, w, h), border_radius=8)
        pygame.draw.rect(surf, (200, 200, 200), (0, 0, w, h), 2, border_radius=8)

        rank_idx = card_int // 4
        suit_idx = card_int % 4
        color = _SUIT_COLORS[suit_idx]
        symbol = _SUIT_SYMBOLS[suit_idx]
        rank = _RANK_DISPLAY[rank_idx]

        font_sm = pygame.font.SysFont("dejavusans", max(14, h // 10))
        font_lg = pygame.font.SysFont("dejavusans", max(28, h // 4))

        # Top-left rank + suit
        rank_surf = font_sm.render(rank, True, color)
        suit_surf = font_sm.render(symbol, True, color)
        surf.blit(rank_surf, (6, 4))
        surf.blit(suit_surf, (6, 4 + rank_surf.get_height()))

        # Center suit symbol (large)
        big = font_lg.render(symbol, True, color)
        surf.blit(big, big.get_rect(center=(w // 2, h // 2)))

        # Bottom-right (rotated 180°)
        rank_r = pygame.transform.rotate(rank_surf, 180)
        suit_r = pygame.transform.rotate(suit_surf, 180)
        surf.blit(rank_r, (w - rank_r.get_width() - 6, h - rank_r.get_height() - suit_r.get_height() - 4))
        surf.blit(suit_r, (w - suit_r.get_width() - 6, h - suit_r.get_height() - 4))

        return surf

    def _draw_back(self) -> pygame.Surface:
        """Draw a simple card back when back.png is missing."""
        w, h = self.card_w, self.card_h
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(surf, (15, 40, 120), (0, 0, w, h), border_radius=8)
        pygame.draw.rect(surf, (180, 160, 220), (0, 0, w, h), 2, border_radius=8)
        # Simple cross-hatch pattern
        for i in range(0, w, 12):
            pygame.draw.line(surf, (25, 55, 140), (i, 0), (i, h), 1)
        for j in range(0, h, 12):
            pygame.draw.line(surf, (25, 55, 140), (0, j), (w, j), 1)
        return surf
