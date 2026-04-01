"""Display subprocess: pygame window driven by a multiprocessing.Queue."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Tuple

import pygame

from gambletron.display.card_loader import CardLoader
from gambletron.display.renderer import PygameRenderer


def run_display(
    queue: multiprocessing.Queue,
    asset_dir: str,
    fullscreen: bool = True,
    target_fps: int = 60,
) -> None:
    """Entry point for the display subprocess.  Blocks until the window closes."""
    # Tell SDL which display to use (first display by default)
    os.environ.setdefault("SDL_VIDEO_FULLSCREEN_DISPLAY", "0")

    pygame.init()
    pygame.mouse.set_visible(False)

    if fullscreen:
        info = pygame.display.Info()
        w, h = info.current_w, info.current_h
        flags = pygame.FULLSCREEN | pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF
        screen = pygame.display.set_mode((w, h), flags)
    else:
        # Windowed mode for development / testing
        screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)

    pygame.display.set_caption("Gambletron")

    card_loader = CardLoader(
        asset_dir=Path(asset_dir),
        card_size=(150, 210),   # base size; renderer rescales for actual screen
    )

    renderer = PygameRenderer(screen, card_loader)

    # Preload cards *after* renderer has set the correct scaled size
    card_loader.preload_all()

    clock = pygame.time.Clock()

    while True:
        # ── System events ─────────────────────────────────────────────────────
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
                return
            if evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE:
                pygame.quit()
                return

        # ── Drain the game-event queue ────────────────────────────────────────
        while True:
            try:
                game_evt = queue.get_nowait()
                renderer.handle_event(game_evt)
            except Exception:
                break   # queue.Empty or any other issue

        # ── Draw ──────────────────────────────────────────────────────────────
        renderer.render()
        pygame.display.flip()
        clock.tick(target_fps)


def start_display_process(
    asset_dir: str,
    fullscreen: bool = True,
) -> Tuple[multiprocessing.Process, multiprocessing.Queue]:
    """Spawn the display subprocess.  Returns (process, queue).

    The caller pushes display events into the queue; the subprocess consumes
    them at its own frame rate without ever blocking the game loop.
    """
    q: multiprocessing.Queue = multiprocessing.Queue(maxsize=512)
    proc = multiprocessing.Process(
        target=run_display,
        args=(q, asset_dir, fullscreen),
        daemon=True,
        name="gambletron-display",
    )
    proc.start()
    return proc, q
