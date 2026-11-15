from __future__ import annotations

from dataclasses import dataclass

from ..comm import Channel, Message, Perf
from .base import Agent, AgentAPI, Knowledge


@dataclass
class Reservation:
    crew: str
    t_req: int
    eta: int
    slot: int
    msg_id: int


class WaterPointAgent(Agent):
    kind = "water"
    roles = ("ρ7 Поставщик ресурса",)

    def __init__(self, aid: str, api: AgentAPI, knowledge: Knowledge, index: int):
        super().__init__(aid, api, knowledge)
        self.queue: list[Reservation] = []
        self.serving: dict[str, int] = {}
        self.served = 0
        self.wait_times: list[int] = []
        self.arrived_at: dict[str, int] = {}
        self.hb_offset = index
        self.state = "READY"

    def order_queue(self, t: int) -> list[Reservation]:
        return sorted(self.queue, key=lambda r: (r.t_req, r.msg_id))

    def _estimate_slot(self, t: int, eta: int) -> int:
        cap = self.cfg.water_capacity
        busy_until = sorted(self.serving.values())
        frees = busy_until + [t] * max(0, cap - len(busy_until))
        frees = sorted(frees)[:cap] if frees else [t]
        for r in self.order_queue(t):
            start = max(frees[0], r.eta)
            frees[0] = start + self.cfg.refill_time
            frees.sort()
        return max(frees[0], eta)

    def step(self, t: int, inbox: list[Message]) -> None:
        api = self.api
        for m in inbox:
            if m.perf == Perf.REQUEST and m.ctype == "SlotRequest":
                if api.broken:
                    api.send(Perf.REFUSE, m.sender, {"type": "SlotGrant", "reason": "out_of_service"},
                             channel=Channel.RES, reply=m.id, cid=m.cid)
                    continue
                self.queue = [r for r in self.queue if r.crew != m.sender]
                eta = int(m.content.get("eta", t))
                slot = self._estimate_slot(t, eta)
                self.queue.append(Reservation(m.sender, t, eta, slot, m.id))
                api.send(Perf.AGREE, m.sender, {"type": "SlotGrant", "slot": slot, "queue": len(self.queue)},
                         channel=Channel.RES, reply=m.id, cid=m.cid)
            elif m.perf == Perf.CANCEL and m.ctype == "SlotRequest":
                self.queue = [r for r in self.queue if r.crew != m.sender]

        if api.broken:
            self.state = "BROKEN"
            return

        for crew, t_end in list(self.serving.items()):
            if t >= t_end:
                api.act(("serve", crew, self.cfg.crew_tank))
                api.send(Perf.INFORM, crew, {"type": "RefillDone"}, channel=Channel.RES)
                del self.serving[crew]
                self.served += 1

        present = set(api.crews_at_post()) - set(self.serving)
        reserved = {r.crew for r in self.queue}
        for c in present & reserved:
            self.arrived_at.setdefault(c, t)
        order = [r.crew for r in self.order_queue(t) if r.crew in present]
        for crew in order:
            if len(self.serving) >= self.cfg.water_capacity:
                break
            self.serving[crew] = t + self.cfg.refill_time
            self.wait_times.append(t - self.arrived_at.pop(crew, t))
            self.queue = [r for r in self.queue if r.crew != crew]
        for c in list(self.arrived_at):
            if c not in present:
                del self.arrived_at[c]

        for r in list(self.queue):
            if t > max(r.eta, r.slot) + self.cfg.noshow_timeout and r.crew not in present:
                self.queue.remove(r)
                api.send(Perf.CANCEL, r.crew, {"type": "SlotGrant", "reason": "no_show"}, channel=Channel.RES)

        self.state = "BUSY" if self.serving else "READY"
        if (t + self.hb_offset) % self.cfg.hb_period == 0:
            api.send(Perf.INFORM, self.K.hq_id, {"type": "Status", "kind": "water", "queue": len(self.queue),
                                                 "serving": list(self.serving)})

    def public_state(self) -> dict:
        return {**super().public_state(), "pos": list(self.api.pos), "queue": [r.crew for r in self.queue],
                "serving": list(self.serving), "served": self.served}
