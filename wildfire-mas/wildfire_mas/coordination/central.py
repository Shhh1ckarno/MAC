from __future__ import annotations

from .base import Coordinator
from .hungarian import hungarian


class CentralCoordinator(Coordinator):
    def __init__(self, method: str = "greedy"):
        assert method in ("greedy", "hungarian")
        self.method = method
        self.name = f"central_{method}"
        self.decisions = 0

    def slots(self, hq, t):
        out = []
        for tk in hq.open_tasks():
            out += [tk] * tk.open_slots()
        return out

    def allocate(self, hq, t: int) -> None:
        free = hq.free_crews(t)
        slots = self.slots(hq, t)
        if not free or not slots:
            return
        self.decisions += 1
        if self.method == "greedy":
            for tk in slots:
                if not free:
                    break
                crew = min(free, key=lambda c: hq.cost_estimate(c, tk))
                free.remove(crew)
                hq.assign(t, tk, crew)
                hq.note(t, f"назначил {crew} → {tk.id} (жадно)")
        else:
            slots = slots[:len(free)]
            cost = [[hq.cost_estimate(c, tk) for tk in slots] for c in free]
            assign = hungarian(cost)
            for i, j in enumerate(assign):
                if j >= 0:
                    hq.assign(t, slots[j], free[i])
                    hq.note(t, f"назначил {free[i]} → {slots[j].id} (венгерский алгоритм)")

    def summary(self) -> dict:
        return {"decisions": self.decisions}
