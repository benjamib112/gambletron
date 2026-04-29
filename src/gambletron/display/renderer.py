"""pygame renderer for the Gambletron table display.

Oval-table layout inspired by Poker Patio.  Players sit around the
perimeter; community cards and pot are in the centre.  No hole cards
are ever shown face-up (the display is visible to all players).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import pygame

from gambletron.display.card_loader import CardLoader
from gambletron.display.events import (
    ActionEvent,
    CommunityCardsEvent,
    HandEndEvent,
    HandStartEvent,
    HideReadyButtonEvent,
    PreflopEvent,
    ShowdownEvent,
    ShowReadyButtonEvent,
    WinnerEvent,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG_COLOR       = ( 16,  48,  28)   # outer background
TABLE_FELT     = ( 30,  82,  48)   # table oval fill
TABLE_BORDER   = ( 55, 120,  70)   # table oval rim
TABLE_RIM2     = ( 38,  95,  55)   # inner rim accent

GOLD           = (212, 175,  55)
TEXT_PRIMARY   = (240, 230, 210)
TEXT_DIM       = (140, 130, 110)
TEXT_DARK      = ( 80,  75,  65)

SEAT_BG        = ( 20,  55,  32)   # seat circle fill
SEAT_BORDER    = ( 60, 115,  70)   # seat circle border
SEAT_ACTIVE_BG = ( 25,  70,  40)   # active player fill
SEAT_ACTIVE_BD = ( 50, 225, 105)   # active player border
SEAT_FOLDED_BG = ( 18,  40,  26)   # folded player fill
SEAT_FOLDED_BD = ( 45,  65,  48)   # folded player border
DEALER_GOLD    = (255, 225,  50)

CARD_SLOT_BG   = ( 15,  45,  22)
CARD_SLOT_BD   = ( 35,  75,  45)

ACTION_COLOR   = (170, 215, 170)
WINNER_BG      = (  8,   8,   4, 220)
WINNER_GOLD    = (255, 215,   0)
WINNER_BORDER  = (180, 145,  30)

POT_COLOR      = (255, 225,  80)
ALLIN_COLOR    = (255,  70,  70)   # ALL-IN marker
THINK_COLOR_W  = (240, 240, 255)   # thinking spinner arc (white)
THINK_COLOR_B  = ( 60, 130, 255)   # thinking spinner dots (blue)

READY_BTN_BG   = ( 30, 120,  60)   # ready button fill
READY_BTN_BD   = ( 60, 200, 100)   # ready button border
READY_BTN_TEXT = (255, 255, 255)   # ready button text


# ── Seat positions around the oval ────────────────────────────────────────────
# Angles (degrees, 0 = right, counter-clockwise) for 2-6 players.
# Arranged so seat 0 is at the bottom-centre.
_SEAT_ANGLES: Dict[int, List[float]] = {
    2: [270, 90],
    3: [270, 30, 150],
    4: [240, 300, 60, 120],
    5: [270, 330, 30, 90, 150],  # NOLINT
    6: [240, 300, 0, 60, 120, 180],
}


class PygameRenderer:
    """Stateful pygame renderer.  Call handle_event() with display events,
    then call render() each frame to draw the current state."""

    def __init__(
        self,
        screen: pygame.Surface,
        card_loader: CardLoader,
    ) -> None:
        self.screen = screen
        self.W, self.H = screen.get_size()
        self.s = self.H / 1080            # global scale factor

        # Card sizes (community cards in centre of table)
        self.CARD_W = int(100 * self.s)
        self.CARD_H = int(140 * self.s)
        self.CARD_GAP = int(12 * self.s)

        self.cards = card_loader
        self.cards.card_w = self.CARD_W
        self.cards.card_h = self.CARD_H
        self.cards._cache.clear()
        self.cards._back = None

        # Table oval geometry — shifted up slightly to leave room for bottom seats + banner
        self.cx = self.W // 2
        self.cy = int(self.H * 0.45)
        self.table_rx = int(self.W * 0.40)
        self.table_ry = int(self.H * 0.33)

        # Seat circle radius
        self.seat_r = int(42 * self.s)

        self._load_fonts(self.s)
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
        self.player_stacks: List[int] = []
        self.idle: bool = True
        # Action display: seat index + description shown briefly after each action
        self.action_seat: Optional[int] = None
        self.action_desc: str = ""
        # Ready button for physical table
        self.ready_button_visible: bool = False
        self.ready_button_rect: Optional[pygame.Rect] = None

    def handle_event(self, event: object) -> None:
        if isinstance(event, HandStartEvent):
            self._reset_state()
            self.hand_num      = event.hand_num
            self.dealer_pos    = event.dealer_pos
            self.num_players   = event.num_players
            self.player_stacks = list(event.player_stacks)
            self.idle          = False
            self.betting_round = "PREFLOP"

        elif isinstance(event, PreflopEvent):
            self.pot            = event.pot
            self.current_player = event.current_player
            self.dealer_pos     = event.dealer_pos
            self.player_stacks  = list(event.player_stacks)

        elif isinstance(event, ActionEvent):
            self.last_action    = f"Seat {event.seat + 1}  {event.description}"
            self.action_seat    = event.seat
            self.action_desc    = event.description
            self.pot            = event.pot
            self.current_player = event.current_player
            self.betting_round  = event.betting_round
            self.player_folded  = list(event.player_folded)
            self.player_stacks  = list(event.player_stacks)

        elif isinstance(event, CommunityCardsEvent):
            self.community_cards = list(event.community_cards)
            self.pot             = event.pot
            self.current_player  = event.current_player
            self.betting_round   = event.betting_round
            self.player_stacks   = list(event.player_stacks)

        elif isinstance(event, ShowdownEvent):
            self.showdown_cards  = dict(event.hole_cards)
            self.community_cards = list(event.community_cards)
            self.pot             = event.pot
            self.player_stacks   = list(event.player_stacks)
            self.current_player  = None
            self.betting_round   = "SHOWDOWN"

        elif isinstance(event, WinnerEvent):
            self.winner_info = {
                "seats": event.seats,
                "pot":   event.pot_won,
                "desc":  event.hand_desc,
            }
            self.current_player = None

        elif isinstance(event, HandEndEvent):
            pass

        elif isinstance(event, ShowReadyButtonEvent):
            self.ready_button_visible = True

        elif isinstance(event, HideReadyButtonEvent):
            self.ready_button_visible = False

    # ── Per-frame render ──────────────────────────────────────────────────────

    def render(self) -> None:
        self.screen.fill(BG_COLOR)
        if self.idle:
            self._draw_idle()
            return
        self._draw_table()
        self._draw_seats()
        self._draw_community_cards()
        self._draw_pot()
        self._draw_header()
        if self.winner_info:
            self._draw_winner_banner()
        if self.ready_button_visible:
            self._draw_ready_button()

    # ── Idle ──────────────────────────────────────────────────────────────────

    def _draw_idle(self) -> None:
        title = self.f_title.render("GAMBLETRON", True, GOLD)
        sub   = self.f_medium.render("Waiting for hand to start...", True, TEXT_DIM)
        self.screen.blit(title, title.get_rect(center=(self.cx, self.cy - int(36 * self.s))))
        self.screen.blit(sub,   sub.get_rect(center=(self.cx, self.cy + int(28 * self.s))))

    # ── Table oval ────────────────────────────────────────────────────────────

    def _draw_table(self) -> None:
        cx, cy = self.cx, self.cy
        rx, ry = self.table_rx, self.table_ry
        rim = int(6 * self.s)

        # Outer rim
        pygame.draw.ellipse(
            self.screen, TABLE_BORDER,
            (cx - rx - rim, cy - ry - rim, 2 * (rx + rim), 2 * (ry + rim)),
        )
        # Inner rim accent
        pygame.draw.ellipse(
            self.screen, TABLE_RIM2,
            (cx - rx - rim // 2, cy - ry - rim // 2, 2 * (rx + rim // 2), 2 * (ry + rim // 2)),
        )
        # Felt fill
        pygame.draw.ellipse(
            self.screen, TABLE_FELT,
            (cx - rx, cy - ry, 2 * rx, 2 * ry),
        )

    # ── Seat positions ────────────────────────────────────────────────────────

    def _seat_positions(self) -> List[Tuple[int, int]]:
        """Return (x, y) centre for each seat, placed on the table rim."""
        n = self.num_players
        angles = _SEAT_ANGLES.get(n, _SEAT_ANGLES[6][:n])
        # Seats sit just outside the table edge
        offset = self.seat_r + int(10 * self.s)
        positions = []
        for deg in angles:
            rad = math.radians(deg)
            x = self.cx + int((self.table_rx + offset) * math.cos(rad))
            y = self.cy - int((self.table_ry + offset) * math.sin(rad))
            positions.append((x, y))
        return positions

    # ── Draw seats ────────────────────────────────────────────────────────────

    def _draw_seats(self) -> None:
        positions = self._seat_positions()
        now = time.time()

        for i, (sx, sy) in enumerate(positions):
            is_folded = i < len(self.player_folded) and self.player_folded[i]
            is_active = (self.current_player == i and self.winner_info is None)
            is_dealer = (self.dealer_pos == i)
            is_action_seat = (self.action_seat == i and self.action_desc)

            # Pick colours
            if is_folded:
                bg, bd, bw = SEAT_FOLDED_BG, SEAT_FOLDED_BD, 2
            elif is_active:
                bg, bd, bw = SEAT_ACTIVE_BG, SEAT_ACTIVE_BD, 3
            else:
                bg, bd, bw = SEAT_BG, SEAT_BORDER, 2

            r = self.seat_r

            # Circle
            pygame.draw.circle(self.screen, bg, (sx, sy), r)
            pygame.draw.circle(self.screen, bd, (sx, sy), r, bw)

            # Seat number inside circle
            num_color = TEXT_DIM if is_folded else TEXT_PRIMARY
            num_surf = self.f_seat_num.render(str(i + 1), True, num_color)
            self.screen.blit(num_surf, num_surf.get_rect(center=(sx, sy)))

            # Stack below the seat circle
            stack_bottom = sy + r + int(6 * self.s)
            if i < len(self.player_stacks):
                stack = self.player_stacks[i]
                stack_color = TEXT_DARK if is_folded else TEXT_PRIMARY
                stack_surf = self.f_stack.render(f"${stack:,}", True, stack_color)
                self.screen.blit(stack_surf, stack_surf.get_rect(
                    centerx=sx, top=stack_bottom))
                stack_bottom += stack_surf.get_height() + int(2 * self.s)

                # ALL-IN marker
                is_all_in = stack == 0 and not is_folded
                if is_all_in:
                    allin_surf = self.f_allin.render("ALL-IN", True, ALLIN_COLOR)
                    self.screen.blit(allin_surf, allin_surf.get_rect(
                        centerx=sx, top=stack_bottom))

            # Dealer button
            if is_dealer:
                btn_r = int(14 * self.s)
                btn_x = sx + r + int(4 * self.s)
                btn_y = sy - r + int(4 * self.s)
                pygame.draw.circle(self.screen, DEALER_GOLD, (btn_x, btn_y), btn_r)
                pygame.draw.circle(self.screen, BG_COLOR,    (btn_x, btn_y), btn_r - int(3 * self.s))
                d_surf = self.f_dealer_btn.render("D", True, DEALER_GOLD)
                self.screen.blit(d_surf, d_surf.get_rect(center=(btn_x, btn_y)))

            # Thinking spinner for the active player
            if is_active:
                self._draw_thinking_spinner(sx, sy, now)

        # Action label drawn once, centred at top of the table
        if self.action_seat is not None and self.action_desc:
            self._draw_action_label()

    def _draw_thinking_spinner(self, sx: int, sy: int, now: float) -> None:
        """Draw an animated arc spinner at the upper-left of the seat circle."""
        # Upper-left offset so it doesn't overlap with the dealer button (upper-right)
        offset = self.seat_r + int(6 * self.s)
        spin_x = sx - int(offset * 0.7)
        spin_y = sy - int(offset * 0.7)

        spin_r = int(18 * self.s)
        # Rotating arc: sweep 240 degrees, rotate over time
        start_angle = (now * 4.0) % (2 * math.pi)
        sweep = math.radians(240)
        rect = pygame.Rect(spin_x - spin_r, spin_y - spin_r, spin_r * 2, spin_r * 2)
        pygame.draw.arc(self.screen, THINK_COLOR_W, rect, start_angle, start_angle + sweep, max(3, int(4 * self.s)))

        # Three small dots that orbit inside the arc
        for j in range(3):
            dot_angle = start_angle + j * math.radians(80)
            dx2 = int(spin_r * 0.55 * math.cos(dot_angle))
            dy2 = int(spin_r * 0.55 * -math.sin(dot_angle))
            dot_r = max(3, int(4 * self.s))
            pygame.draw.circle(self.screen, THINK_COLOR_B, (spin_x + dx2, spin_y + dy2), dot_r)

    def _draw_action_label(self) -> None:
        """Draw the most recent action in large bold text at the top-centre of the table."""
        text = f"Seat {self.action_seat + 1}  {self.action_desc}"
        surf = self.f_action_big.render(text, True, ACTION_COLOR)
        y = self.cy - self.table_ry + int(70 * self.s)
        self.screen.blit(surf, surf.get_rect(centerx=self.cx, top=y))

    # ── Community cards ───────────────────────────────────────────────────────

    def _draw_community_cards(self) -> None:
        total_w = 5 * self.CARD_W + 4 * self.CARD_GAP
        x0 = self.cx - total_w // 2
        y  = self.cy - self.CARD_H // 2 - int(30 * self.s)

        for i in range(5):
            x    = x0 + i * (self.CARD_W + self.CARD_GAP)
            rect = pygame.Rect(x, y, self.CARD_W, self.CARD_H)
            if i < len(self.community_cards):
                self.screen.blit(self.cards.get(self.community_cards[i]), rect)
            else:
                pygame.draw.rect(self.screen, CARD_SLOT_BG, rect, border_radius=8)
                pygame.draw.rect(self.screen, CARD_SLOT_BD, rect, 2, border_radius=8)

    # ── Pot ───────────────────────────────────────────────────────────────────

    def _draw_pot(self) -> None:
        y = self.cy + self.CARD_H // 2 - int(10 * self.s)

        lbl = self.f_pot_label.render("POT", True, TEXT_DIM)
        amt = self.f_pot.render(f"${self.pot:,}", True, POT_COLOR)
        self.screen.blit(lbl, lbl.get_rect(centerx=self.cx, top=y))
        self.screen.blit(amt, amt.get_rect(centerx=self.cx, top=y + lbl.get_height() + int(2 * self.s)))

    # ── Header (top strip) ────────────────────────────────────────────────────

    def _draw_header(self) -> None:
        m = int(20 * self.s)
        y = int(12 * self.s)

        title = self.f_header.render("GAMBLETRON", True, GOLD)
        self.screen.blit(title, (m, y))

        if self.betting_round:
            rnd = self.f_header.render(self.betting_round, True, TEXT_PRIMARY)
            self.screen.blit(rnd, rnd.get_rect(centerx=self.cx, top=y))

        hand_txt = self.f_header_sm.render(f"Hand #{self.hand_num}", True, TEXT_DIM)
        self.screen.blit(hand_txt, hand_txt.get_rect(right=self.W - m, top=y))

    # ── Winner banner ─────────────────────────────────────────────────────────

    def _draw_winner_banner(self) -> None:
        info  = self.winner_info
        seats = info["seats"]
        pot   = info["pot"]
        desc  = info.get("desc", {})

        bh = int(100 * self.s)
        bw = int(self.W * 0.55)
        bx = (self.W - bw) // 2
        by = self.H - bh - int(20 * self.s)

        banner = pygame.Rect(bx, by, bw, bh)
        # Semi-transparent background
        overlay = pygame.Surface((bw, bh), pygame.SRCALPHA)
        overlay.fill(WINNER_BG)
        self.screen.blit(overlay, (bx, by))
        pygame.draw.rect(self.screen, WINNER_BORDER, banner, 3, border_radius=14)

        if len(seats) == 1:
            seat      = seats[0]
            hand_name = desc.get(seat, "")
            line1 = f"WINNER  --  Seat {seat + 1}  --  {hand_name}" if hand_name else f"WINNER  --  Seat {seat + 1}"
            line2 = f"Wins  ${pot:,}"
        else:
            seat_str   = "  &  ".join(f"Seat {s + 1}" for s in seats)
            hand_parts = [desc[s] for s in seats if s in desc]
            line1 = f"SPLIT POT  --  {seat_str}"
            line2 = ("  /  ".join(hand_parts) + f"  --  ${pot:,} each") if hand_parts else f"Split  ${pot:,} each"

        w1 = self.f_winner.render(line1, True, WINNER_GOLD)
        w2 = self.f_action.render(line2, True, TEXT_PRIMARY)

        self.screen.blit(w1, w1.get_rect(center=(self.cx, by + bh // 2 - w1.get_height() // 2 - int(4 * self.s))))
        self.screen.blit(w2, w2.get_rect(center=(self.cx, by + bh // 2 + w2.get_height() // 2 + int(4 * self.s))))

    # ── Ready button ─────────────────────────────────────────────────────────

    def _draw_ready_button(self) -> None:
        btn_w = int(400 * self.s)
        btn_h = int(90 * self.s)
        btn_x = (self.W - btn_w) // 2
        btn_y = self.H // 2 - btn_h // 2

        rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        self.ready_button_rect = rect

        pygame.draw.rect(self.screen, READY_BTN_BG, rect, border_radius=18)
        pygame.draw.rect(self.screen, READY_BTN_BD, rect, 3, border_radius=18)

        text = self.f_ready_btn.render("DEAL", True, READY_BTN_TEXT)
        self.screen.blit(text, text.get_rect(center=rect.center))

        hint = self.f_header_sm.render("Return all cards to dealer, then tap to deal", True, TEXT_DIM)
        self.screen.blit(hint, hint.get_rect(centerx=self.cx, top=rect.bottom + int(16 * self.s)))

    # ── Font loading ──────────────────────────────────────────────────────────

    def _load_fonts(self, scale: float) -> None:
        def pick(size_base: int, bold: bool = False) -> pygame.font.Font:
            size = max(12, int(size_base * scale))
            if bold:
                names = ("dejavusansbold", "liberationsansbold", "freesansbold")
            else:
                names = ("dejavusans", "liberationsans", "freesans")
            for name in names:
                path = pygame.font.match_font(name)
                if path:
                    return pygame.font.Font(path, size)
            return pygame.font.SysFont("freesans", size, bold=bold)

        self.f_title      = pick(52, bold=True)
        self.f_header     = pick(28, bold=True)
        self.f_header_sm  = pick(22)
        self.f_seat_num   = pick(36, bold=True)     # number inside seat circle
        self.f_stack      = pick(28, bold=True)      # stack $ below seat
        self.f_pot_label  = pick(26)                 # "POT" label
        self.f_pot        = pick(48, bold=True)      # pot amount (large)
        self.f_action     = pick(26)
        self.f_winner     = pick(38, bold=True)
        self.f_dealer_btn   = pick(16, bold=True)
        self.f_action_big   = pick(52, bold=True)    # large action text top-centre
        self.f_allin        = pick(22, bold=True)    # ALL-IN marker below seat
        self.f_medium       = pick(34)
        self.f_ready_btn    = pick(44, bold=True)
