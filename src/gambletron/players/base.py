"""Abstract base class for all player types."""

from __future__ import annotations

from abc import ABC, abstractmethod

from gambletron.poker.state import Action, VisibleGameState


class Player(ABC):
    """Base class for poker players (human, AI, random, hardware-controlled)."""

    def __init__(self, name: str = "Player") -> None:
        self.name = name

    @abstractmethod
    def get_action(self, state: VisibleGameState) -> Action:
        """Given the visible game state, return an action."""
        ...

    def notify_hand_start(self, state: VisibleGameState) -> None:
        """Called at the start of each hand. Override for bookkeeping."""
        pass

    def notify_hand_end(self, state: VisibleGameState) -> None:
        """Called at the end of each hand with final state. Override for learning."""
        pass

    def notify_action(self, seat: int, action: Action) -> None:
        """Called when any player takes an action. Override for tracking."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"
