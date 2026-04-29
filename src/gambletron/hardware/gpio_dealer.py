"""GPIO-based card dealer control for Raspberry Pi 5.

Triggers the physical dealer's shuffle+deal cycle by pulsing
GPIO pins 5 and 6 high for a brief duration.
"""

from __future__ import annotations

import time
from typing import Optional

DEALER_GPIO_PINS = (5, 6)
PULSE_DURATION = 0.2


class GPIODealer:
    """Controls the physical card dealer via RPi 5 GPIO pins."""

    def __init__(
        self,
        pins: tuple = DEALER_GPIO_PINS,
        pulse_seconds: float = PULSE_DURATION,
    ) -> None:
        self._pins = pins
        self._pulse_seconds = pulse_seconds
        self._chip: Optional[object] = None
        self._lines: Optional[object] = None

    def open(self) -> None:
        import gpiod
        from gpiod.line import Direction, Value

        self._chip = gpiod.Chip("/dev/gpiochip4")
        config = {
            pin: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE)
            for pin in self._pins
        }
        self._lines = self._chip.request_lines(
            consumer="gambletron-dealer",
            config=config,
        )

    def close(self) -> None:
        if self._lines:
            self._lines.release()
            self._lines = None
        if self._chip:
            self._chip.close()
            self._chip = None

    def trigger(self) -> None:
        """Pulse dealer GPIO pins high to initiate shuffle+deal."""
        if not self._lines:
            raise RuntimeError("GPIODealer not opened")
        from gpiod.line import Value

        self._lines.set_value(self._pins[0], Value.ACTIVE)
        self._lines.set_value(self._pins[1], Value.ACTIVE)
        time.sleep(self._pulse_seconds)
        self._lines.set_value(self._pins[0], Value.INACTIVE)
        self._lines.set_value(self._pins[1], Value.INACTIVE)
