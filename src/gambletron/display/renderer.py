"""pygame renderer for the Gambletron table display."""

from __future__ import annotations

from typing import Dict, List, Optional

import pygame

from gambletron.display.card_loader import CardLoader
from gambletron.display.events import (
    ActionEvent,
    CommunityCardsEvent,
    HandEndEvent,
    HandStartEvent,
    PreflopEvent,
    ShowdownEvent,
    WinnerEvent,
)

# ── Palette ───────────────────────────────────────────────────────────────────
FELT_GREEN   = ( 26,  71,  42)   # main background
FELT_DARK    = ( 15,  45,  25)   # inset panels
FELT_EDGE    = (  8,  28,  14)   # dividers
GOLD         = (212, 175,  55)   # accent / pot
TEXT_PRIMARY = (240, 230, 210)   # warm white
TEXT_DIM     = (140, 130, 110)   # secondary labels
TO_ACT_GREEN = ( 50, 225, 105)   # "to act" value
DEALER_GOLD  = (255, 225,  50)   # dealer button
CARD_SLOT_BG = ( 15,  45,  22)   # empty card slot fill
CARD_SLOT_BD = ( 35,  75,  45)   # empty card slot border
ACTION_COLOR = (170, 215, 170)   # last-action text
WINNER_BG    = (  8,   8,   4)   # winner banner background
WINNER_GOLD  = (255, 215,   0)   # winner headline text
WINNER_BORDER= (180, 145,  30)   # winner banner border


class PygameRenderer:
    """Stateful pygame renderer.  Call handle_event() with display events,
    then call render() each frame to draw the current state."""

    # ── Layout constants (for 1080p). Scale factor applied in __init__. ───────
    _CARD_W_BASE    = 150
    _CARD_H_BASE    = 210
    _CARD_GAP_BASE  = 18
    _SM_CARD_W_BASE = 100
    _SM_CARD_H_BASE = 140
    _HEADER_H_BASE  = 82
    _MARGIN_BASE    = 44

    def __init__(
        self,
        screen: pygame.Surface,
        card_loader: CardLoader,
    ) -> None:
        self.screen = screen
        self.W, self.H = screen.get_size()

        # Scale all layout values proportionally to actual screen height
        s = self.H / 1080
        self.CARD_W    = int(self._CARD_W_BASE    * s)
        self.CARD_H    = int(self._CARD_H_BASE    * s)
        self.CARD_GAP  = int(self._CARD_GAP_BASE  * s)
        self.SM_W      = int(self._SM_CARD_W_BASE * s)
        self.SM_H      = int(self._SM_CARD_H_BASE * s)
        self.HEADER_H  = int(self._HEADER_H_BASE  * s)
        self.MARGIN    = int(self._MARGIN_BASE     * s)

        self.cards = card_loader
        # Re-build card loader's size to match scaled dimensions
        self.cards.card_w = self.CARD_W
        self.cards.card_h = self.CARD_H
        self.cards._cache.clear()
        self.cards._back = None

        self._load_fonts(s)
        self._reset_state()

    # ── State management ──────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self.hand_num: int = 0
        self.dealer_pos: Optional[int] = None
        self.num_players: int = 6
        self.community_cards: List[int] = []
        self.pot: int = 0
        self.current_player: Optional[int] = None
        self.betting_round: str = ""
        self.player_folded: List[bool] = []
        self.last_action: str = ""
        self.showdown_cards: Dict[int, List[int]] = {}
        self.winner_info: Optional[dict] = None
        self.idle: bool = True

    def handle_event(self, event: object) -> None:
        if isinstance(event, HandStartEvent):
            self._reset_state()
            self.hand_num    = event.hand_num
            self.dealer_pos  = event.dealer_pos
            self.num_players = event.num_players
            self.idle        = False
            self.betting_round = "PREFLOP"

        elif isinstance(event, PreflopEvent):
            self.pot            = event.pot
            self.current_player = event.current_player
            self.dealer_pos     = event.dealer_pos

        elif isinstance(event, ActionEvent):
            self.last_action    = f"Seat {event.seat}  {event.description}"
            self.pot            = event.pot
            self.current_player = event.current_player
            self.betting_round  = event.betting_round
            self.player_folded  = list(event.player_folded)

        elif isinstance(event, CommunityCardsEvent):
            self.community_cards = list(event.community_cards)
            self.pot             = event.pot
            self.current_player  = event.current_player
            self.betting_round   = event.betting_round

        elif isinstance(event, ShowdownEvent):
            self.showdown_cards  = dict(event.hole_cards)
            self.community_cards = list(event.community_cards)
            self.pot             = event.pot
            self.current_player  = None
            self.betting_round   = "SHOWDOWN"

        elif isinstance(event, WinnerEvent):
            self.winner_info = {
                "seats":    event.seats,
                "pot":      event.pot_won,
                "desc":     event.hand_desc,
            }
            self.current_player = None

        elif isinstance(event, HandEndEvent):
            pass  # winner banner persists until next HandStartEvent

    # ── Per-frame render ──────────────────────────────────────────────────────

    def render(self) -> None:
        self.screen.fill(FELT_GREEN)
        if self.idle:
            self._draw_idle()
            return
        self._draw_header()
        self._draw_community_cards()
        self._draw_pot()
        self._draw_info_bar()
        self._draw_last_action()
        if self.showdown_cards:
            self._draw_showdown_cards()
        if self.winner_info:
            self._draw_winner_banner()

    # ── Idle ──────────────────────────────────────────────────────────────────

    def _draw_idle(self) -> None:
        cx, cy = self.W // 2, self.H // 2
        title = self.f_title.render("GAMBLETRON", True, GOLD)
        sub   = self.f_medium.render("Waiting for hand to start…", True, TEXT_DIM)
        self.screen.blit(title, title.get_rect(center=(cx, cy - 36)))
        self.screen.blit(sub,   sub.get_rect(center=(cx, cy + 28)))

    # ── Header ────────────────────────────────────────────────────────────────

    def _draw_header(self) -> None:
        cx = self.W // 2
        m  = self.MARGIN

        title = self.f_title.render("GAMBLETRON", True, GOLD)
        self.screen.blit(title, (m, m // 2))

        if self.betting_round:
            rnd = self.f_large.render(self.betting_round, True, TEXT_PRIMARY)
            self.screen.blit(rnd, rnd.get_rect(center=(cx, self.HEADER_H // 2)))

        hand_txt = self.f_small.render(f"Hand #{self.hand_num}", True, TEXT_DIM)
        self.screen.blit(hand_txt, hand_txt.get_rect(
            right=self.W - m, centery=self.HEADER_H // 2))

        pygame.draw.line(
            self.screen, FELT_EDGE,
            (0, self.HEADER_H), (self.W, self.HEADER_H), 2,
        )

    # ── Community cards ───────────────────────────────────────────────────────

    def _draw_community_cards(self) -> None:
        total_w = 5 * self.CARD_W + 4 * self.CARD_GAP
        x0 = (self.W - total_w) // 2
        y  = self.HEADER_H + int(18 * self.H / 1080)

        for i in range(5):
            x    = x0 + i * (self.CARD_W + self.CARD_GAP)
            rect = pygame.Rect(x, y, self.CARD_W, self.CARD_H)
            if i < len(self.community_cards):
                self.screen.blit(self.cards.get(self.community_cards[i]), rect)
            else:
                pygame.draw.rect(
                    self.screen, CARD_SLOT_BG, rect, border_radius=10)
                pygame.draw.rect(
                    self.screen, CARD_SLOT_BD, rect, 2, border_radius=10)

    # ── Pot ───────────────────────────────────────────────────────────────────

    def _draw_pot(self) -> None:
        card_bottom = self.HEADER_H + int(18 * self.H / 1080) + self.CARD_H
        y  = card_bottom + int(20 * self.H / 1080)
        cx = self.W // 2

        lbl = self.f_tiny.render("POT", True, TEXT_DIM)
        amt = self.f_large.render(f"${self.pot:,}", True, GOLD)
        self.screen.blit(lbl, lbl.get_rect(center=(cx, y + lbl.get_height() // 2)))
        self.screen.blit(amt, amt.get_rect(center=(cx, y + lbl.get_height() + 6 + amt.get_height() // 2)))

    # ── Info bar: Dealer left / To Act right ──────────────────────────────────

    def _draw_info_bar(self) -> None:
        card_bottom = self.HEADER_H + int(18 * self.H / 1080) + self.CARD_H
        y  = card_bottom + int(105 * self.H / 1080)
        m  = self.MARGIN

        # ── Dealer ────────────────────────────────────────────────────────────
        if self.dealer_pos is not None:
            lbl = self.f_tiny.render("DEALER", True, TEXT_DIM)
            val = self.f_medium.render(f"Seat {self.dealer_pos}", True, DEALER_GOLD)
            self.screen.blit(lbl, (m, y))
            self.screen.blit(val, (m, y + lbl.get_height() + 4))

            # Dealer button circle next to the seat text
            btn_x = m + val.get_width() + int(18 * self.W / 1920)
            btn_y = y + lbl.get_height() + 4 + val.get_height() // 2
            r = int(14 * self.H / 1080)
            pygame.draw.circle(self.screen, DEALER_GOLD, (btn_x, btn_y), r)
            pygame.draw.circle(self.screen, FELT_DARK,   (btn_x, btn_y), r - 3)
            d = self.f_tiny.render("D", True, DEALER_GOLD)
            self.screen.blit(d, d.get_rect(center=(btn_x, btn_y)))

        # ── To Act ────────────────────────────────────────────────────────────
        if self.current_player is not None and self.winner_info is None:
            lbl = self.f_tiny.render("TO ACT", True, TEXT_DIM)
            val = self.f_medium.render(f"Seat {self.current_player}", True, TO_ACT_GREEN)
            rx  = self.W - m
            self.screen.blit(lbl, lbl.get_rect(right=rx, top=y))
            self.screen.blit(val, val.get_rect(right=rx, top=y + lbl.get_height() + 4))

    # ── Last action ───────────────────────────────────────────────────────────

    def _draw_last_action(self) -> None:
        if not self.last_action:
            return
        card_bottom = self.HEADER_H + int(18 * self.H / 1080) + self.CARD_H
        y  = card_bottom + int(175 * self.H / 1080)
        cx = self.W // 2
        surf = self.f_small.render(self.last_action, True, ACTION_COLOR)
        self.screen.blit(surf, surf.get_rect(center=(cx, y)))

    # ── Showdown hole cards ───────────────────────────────────────────────────

    def _draw_showdown_cards(self) -> None:
        if not self.showdown_cards:
            return

        card_bottom = self.HEADER_H + int(18 * self.H / 1080) + self.CARD_H
        y = card_bottom + int(220 * self.H / 1080)

        seats  = sorted(self.showdown_cards)
        n      = len(seats)
        hand_w = 2 * self.SM_W + 6
        gap    = max(30, int(50 * self.W / 1920))
        total  = n * hand_w + (n - 1) * gap
        x0     = (self.W - total) // 2

        for idx, seat in enumerate(seats):
            cards = self.showdown_cards[seat]
            x     = x0 + idx * (hand_w + gap)

            lbl = self.f_tiny.render(f"Seat {seat}", True, TEXT_DIM)
            self.screen.blit(lbl, lbl.get_rect(
                centerx=x + hand_w // 2, bottom=y - 4))

            for ci, card_int in enumerate(cards):
                surf = pygame.transform.smoothscale(
                    self.cards.get(card_int), (self.SM_W, self.SM_H))
                self.screen.blit(surf, (x + ci * (self.SM_W + 6), y))

    # ── Winner banner ─────────────────────────────────────────────────────────

    def _draw_winner_banner(self) -> None:
        info  = self.winner_info
        seats = info["seats"]
        pot   = info["pot"]
        desc  = info.get("desc", {})

        bh = int(110 * self.H / 1080)
        by = self.H - bh - int(20 * self.H / 1080)
        bx = self.MARGIN
        bw = self.W - 2 * self.MARGIN

        banner = pygame.Rect(bx, by, bw, bh)
        pygame.draw.rect(self.screen, WINNER_BG, banner, border_radius=14)
        pygame.draw.rect(self.screen, WINNER_BORDER, banner, 3, border_radius=14)

        cx = self.W // 2
        if len(seats) == 1:
            seat      = seats[0]
            hand_name = desc.get(seat, "")
            line1     = f"WINNER  —  Seat {seat}  —  {hand_name}" if hand_name else f"WINNER  —  Seat {seat}"
            line2     = f"Wins  ${pot:,}"
        else:
            seat_str  = "  &  ".join(f"Seat {s}" for s in seats)
            hand_parts = [desc[s] for s in seats if s in desc]
            line1     = f"SPLIT POT  —  {seat_str}"
            line2     = ("  /  ".join(hand_parts) + f"  —  ${pot:,} each") if hand_parts else f"Split  ${pot:,} each"

        w1 = self.f_winner.render(line1, True, WINNER_GOLD)
        w2 = self.f_small.render(line2, True, TEXT_PRIMARY)

        self.screen.blit(w1, w1.get_rect(center=(cx, by + bh // 2 - w1.get_height() // 2 - 4)))
        self.screen.blit(w2, w2.get_rect(center=(cx, by + bh // 2 + w2.get_height() // 2 + 4)))

    # ── Font loading ──────────────────────────────────────────────────────────

    def _load_fonts(self, scale: float) -> None:
        def pick(size_base: int, *names: str) -> pygame.font.Font:
            size = max(12, int(size_base * scale))
            for name in names:
                path = pygame.font.match_font(name)
                if path:
                    return pygame.font.Font(path, size)
            return pygame.font.SysFont("freesans", size)

        self.f_title  = pick(52, "dejavusansbold", "dejavusans", "liberationsansbold", "freesansbold")
        self.f_winner = pick(48, "dejavusansbold", "dejavusans", "liberationsansbold", "freesansbold")
        self.f_large  = pick(42, "dejavusans", "liberationsans", "freesans")
        self.f_medium = pick(34, "dejavusans", "liberationsans", "freesans")
        self.f_small  = pick(26, "dejavusans", "liberationsans", "freesans")
        self.f_tiny   = pick(20, "dejavusans", "liberationsans", "freesans")
