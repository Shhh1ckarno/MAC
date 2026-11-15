from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.hq import HQAgent
    from ..comm import Message


class Coordinator:
    name = "base"

    def allocate(self, hq: "HQAgent", t: int) -> None:
        pass

    def on_message(self, hq: "HQAgent", t: int, m: "Message") -> None:
        pass

    def engaged_crews(self, hq: "HQAgent") -> set[str]:
        return set()

    def summary(self) -> dict:
        return {}
