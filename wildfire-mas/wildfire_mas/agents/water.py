from __future__ import annotations

import math
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
    cid: str | None
    priority: float


class WaterPointAgent(Agent):
    kind = "water"
    roles = ("ρ7 Поставщик ресурса",)

    def __init__(self, aid: str, api: AgentAPI, knowledge: Knowledge, index: int):
        super().__init__(aid, api, knowledge)
        self.queue: list[Reservation] = []
        self.serving: dict[str, tuple[Reservation, int]] = {}
        self.served = 0
        self.requests = 0
        self.wait_times: list[int] = []
        self.waits_by_priority: list[tuple[float, int]] = []
        self.arrived_at: dict[str, int] = {}
        self.peak_serving = 0
        self.peak_waiting = 0
        self.hb_offset = index
        self.state = "READY"

    def order_queue(self, t: int) -> list[Reservation]:
        def key(r: Reservation):
            if self.cfg.refill_policy == "fifo":
                return (r.t_req, r.msg_id)
            aged = t - self.arrived_at.get(r.crew, t) >= 2 * self.cfg.refill_time
            return (not aged, 0 if aged else -r.priority, r.t_req, r.msg_id)

        return sorted(self.queue, key=key)

    def _estimate_slot(self, t: int, eta: int) -> int:
        busy_until = [max(t, end) for _, end in self.serving.values()]
        frees = sorted(busy_until + [t] * (self.cfg.water_capacity - len(busy_until)))
        for r in self.order_queue(t):
            frees[0] = max(frees[0], r.eta) + self.cfg.refill_time
            frees.sort()
        return max(frees[0], eta)

    def _cancel(self, r: Reservation, reason: str) -> None:
        self.api.send(Perf.CANCEL, r.crew, {"type": "SlotGrant", "reason": reason},
                      channel=Channel.RES, cid=r.cid)
        self.api.record("refill_cancelled", crew=r.crew, cid=r.cid, reason=reason)

    def _handle_request(self, t: int, m: Message) -> None:
        api = self.api
        if m.perf == Perf.REQUEST and m.ctype == "SlotRequest":
            self.requests += 1
            if api.broken:
                api.send(Perf.REFUSE, m.sender, {"type": "SlotGrant", "reason": "out_of_service"},
                         channel=Channel.RES, reply=m.id, cid=m.cid)
                return
            if m.sender in self.serving:
                return
            self.queue = [r for r in self.queue if r.crew != m.sender]
            self.arrived_at.pop(m.sender, None)
            eta = int(m.content.get("eta", t))
            slot = self._estimate_slot(t, eta)
            priority = m.content.get("priority", 0)
            if type(priority) not in (int, float) or not math.isfinite(priority) or priority < 0:
                priority = 0
            self.queue.append(Reservation(m.sender, t, eta, slot, m.id, m.cid, priority))
            api.send(Perf.AGREE, m.sender, {"type": "SlotGrant", "slot": slot, "queue": len(self.queue)},
                     channel=Channel.RES, reply=m.id, cid=m.cid)
        elif m.perf == Perf.CANCEL and m.ctype == "SlotRequest":
            self.queue = [r for r in self.queue if (r.crew, r.cid) != (m.sender, m.cid)]
            current = self.serving.get(m.sender)
            if current and current[0].cid == m.cid:
                del self.serving[m.sender]

    def step(self, t: int, inbox: list[Message]) -> None:
        api = self.api
        for m in inbox:
            self._handle_request(t, m)

        if api.broken:
            self.state = "BROKEN"
            if self.cfg.reliability_guards:
                for r in self.queue + [r for r, _ in self.serving.values()]:
                    self._cancel(r, "out_of_service")
                self.queue.clear()
                self.serving.clear()
                self.arrived_at.clear()
            return

        at_post = set(api.crews_at_post())
        completed = []
        for crew, (r, t_end) in list(self.serving.items()):
            if crew not in at_post:
                self._cancel(r, "left_post")
                del self.serving[crew]
            elif t >= t_end:
                completed.append(crew)
                api.send(Perf.INFORM, crew, {"type": "RefillDone"}, channel=Channel.RES, cid=r.cid)
                api.record("refill_done", crew=crew, cid=r.cid)
                del self.serving[crew]
                self.served += 1
        if completed:
            api.act(("serve", completed))

        present = at_post - set(self.serving) - set(completed)
        eligible = {r.crew for r in self.queue} & present
        self.arrived_at = {crew: self.arrived_at.get(crew, t) for crew in eligible}
        order = [r for r in self.order_queue(t) if r.crew in eligible]
        for r in order[:self.cfg.water_capacity - len(self.serving)]:
            self.serving[r.crew] = (r, t + self.cfg.refill_time)
            wait = t - self.arrived_at.pop(r.crew)
            self.wait_times.append(wait)
            self.waits_by_priority.append((r.priority, wait))
            self.queue.remove(r)
            reason = "fifo"
            if self.cfg.refill_policy == "priority":
                reason = "aged" if wait >= 2 * self.cfg.refill_time else "priority"
            api.record("refill_started", crew=r.crew, priority=r.priority, wait=wait, reason=reason, cid=r.cid)
            if self.cfg.refill_policy == "priority" or self.cfg.reliability_guards:
                api.send(Perf.INFORM, r.crew, {"type": "SlotStarted"}, channel=Channel.RES, cid=r.cid)
        self.peak_serving = max(self.peak_serving, len(self.serving))
        self.peak_waiting = max(self.peak_waiting, len(self.arrived_at))

        for r in list(self.queue):
            if t > max(r.eta, r.slot) + self.cfg.noshow_timeout and r.crew not in present:
                self.queue.remove(r)
                self._cancel(r, "no_show")

        self.state = "BUSY" if self.serving else "READY"
        if (t + self.hb_offset) % self.cfg.hb_period == 0:
            api.send(Perf.INFORM, self.K.hq_id, {"type": "Status", "kind": "water", "queue": len(self.queue),
                                                 "serving": list(self.serving)})

    def queue_snapshot(self, t: int) -> dict:
        serving = [{"crew": r.crew, "until": end} for r, end in self.serving.values()]
        requests = []
        for r in self.order_queue(t):
            arrival = self.arrived_at.get(r.crew)
            requests.append({"crew": r.crew, "priority": r.priority,
                             "waiting": max(0, t - arrival - 1) if arrival is not None else None})
        return {"id": self.id, "broken": self.api.broken, "requests": requests,
                "serving": serving, "served": self.served, "capacity": self.cfg.water_capacity}

    def public_state(self) -> dict:
        return {**super().public_state(), "pos": list(self.api.pos), "queue": [r.crew for r in self.queue],
                "serving": list(self.serving), "served": self.served}
