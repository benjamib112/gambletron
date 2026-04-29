"""Display subprocess: pygame window driven by a multiprocessing.Queue."""

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import pygame

from gambletron.display.card_loader import CardLoader
from gambletron.display.events import ActionEvent
from gambletron.display.renderer import PygameRenderer

# How long (seconds) to display an action before processing the next event.
ACTION_DISPLAY_SECONDS = 2.0


def run_display(
    queue: multiprocessing.Queue,
    asset_dir: str,
    fullscreen: bool = True,
    target_fps: int = 60,
    ready_event: Optional[multiprocessing.Event] = None,
) -> None:
    """Entry point for the display subprocess.  Blocks until the window closes."""
    # Tell SDL which display to use (first display by default)
    os.environ.setdefault("SDL_VIDEO_FULLSCREEN_DISPLAY", "0")

    pygame.init()
    if ready_event:
        pygame.mouse.set_visible(True)
    else:
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
    action_hold_until = 0.0   # monotonic timestamp until which we pause event processing

    while True:
        # ── System events ─────────────────────────────────────────────────────
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
                return
            if evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE:
                pygame.quit()
                return
            if evt.type in (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
                if evt.type == pygame.FINGERDOWN:
                    pos = (int(evt.x * renderer.W), int(evt.y * renderer.H))
                else:
                    pos = evt.pos
                if renderer.ready_button_visible and renderer.ready_button_rect:
                    if renderer.ready_button_rect.collidepoint(pos):
                        renderer.ready_button_visible = False
                        if ready_event:
                            ready_event.set()

        # ── Drain the game-event queue (paused during action hold) ────────────
        now = time.monotonic()
        if now >= action_hold_until:
            # Clear the action label once the hold period expires
            if renderer.action_seat is not None:
                renderer.action_seat = None
                renderer.action_desc = ""

            while True:
                try:
                    game_evt = queue.get_nowait()
                    renderer.handle_event(game_evt)
                    # After an ActionEvent, hold for 1 second so players can read it
                    if isinstance(game_evt, ActionEvent):
                        action_hold_until = time.monotonic() + ACTION_DISPLAY_SECONDS
                        break   # stop draining; render this action first
                except Exception:
                    break   # queue.Empty or any other issue

        # ── Draw ──────────────────────────────────────────────────────────────
        renderer.render()
        pygame.display.flip()
        clock.tick(target_fps)


def start_display_process(
    asset_dir: str,
    fullscreen: bool = True,
    with_ready_button: bool = False,
) -> Tuple[multiprocessing.Process, multiprocessing.Queue, Optional[multiprocessing.Event]]:
    """Spawn the display subprocess.

    Returns (process, queue, ready_event).
    ready_event is set when the touchscreen 'Ready to Deal' button is pressed.
    It is None when with_ready_button is False.
    """
    q: multiprocessing.Queue = multiprocessing.Queue(maxsize=512)
    ready_event = multiprocessing.Event() if with_ready_button else None
    proc = multiprocessing.Process(
        target=run_display,
        args=(q, asset_dir, fullscreen),
        kwargs={"ready_event": ready_event},
        daemon=True,
        name="gambletron-display",
    )
    proc.start()
    return proc, q, ready_event
