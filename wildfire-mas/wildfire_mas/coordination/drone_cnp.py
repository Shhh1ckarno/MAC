from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..comm import Channel, Perf
from ..env import Pos, cheb
from .base import Coordinator


class PassiveCoordinator(Coordinator):
    name = "cnp_drone"


@dataclass
class Contract:
    id: str
    cells: set[Pos]
    t_created: int
    crews: dict[str, int] = field(default_factory=dict)
    round: dict | None = None
    cooldown: int = -1
    t_update: int = -100
    t_seen: int = 0
    recruit: bool = False

    def demand(self, per_crew: int, n_crews: int) -> int:
        return int(min(n_crews, max(1, math.ceil(len(self.cells) / per_crew))))


class DroneContractManager:
    HOVER = 15
    CONTRACT_TTL = 60

    def __init__(self, drone):
        self.drone = drone
        self.contracts: dict[str, Contract] = {}
        self.n = 0
        self.stats = {"rounds": 0, "awards": 0, "no_bids": 0}

    @property
    def api(self):
        return self.drone.api

    def _content(self, c: Contract) -> dict:
        cells = sorted(c.cells)[:60]
        cx = round(sum(x for x, _ in c.cells) / len(c.cells))
        cy = round(sum(y for _, y in c.cells) / len(c.cells))
        return {"type": "Task", "task_id": c.id, "kind": "EXTINGUISH", "cells": [list(p) for p in cells],
                "center": [cx, cy], "priority": float(len(c.cells)), "hops": 0}

    def _send(self, perf, to, content, cid=None):
        return self.api.send(perf, to, content, channel=Channel.RADIO, cid=cid)

    def update(self, t: int, burning: dict[Pos, float], view: list[Pos]) -> None:
        seen = set(view)
        for c in self.contracts.values():
            old = c.cells
            near = {p for p in burning if old and min(cheb(p, q) for q in old) <= 2}
            c.cells = {p for p in old if p not in seen} | near
            if near:
                c.t_seen = t
        for comp in self._clusters(burning):
            if any(comp & self._dilate(c.cells) for c in self.contracts.values()):
                continue
            self.n += 1
            cid = f"{self.drone.id}-K{self.n}"
            self.contracts[cid] = Contract(cid, comp, t, t_seen=t)
            self.drone.note(t, f"новый очаг {cid}: {len(comp)} кл., объявляю торги по радио")
        for cid, c in list(self.contracts.items()):
            for crew, ta in list(c.crews.items()):
                if t - ta > self.CONTRACT_TTL:
                    del c.crews[crew]
            if not c.cells or (t - c.t_seen > 40 and not c.crews):
                if c.crews:
                    self._send(Perf.CANCEL, list(c.crews), {"type": "Task", "task_id": cid, "reason": "done"})
                del self.contracts[cid]

    def _dilate(self, cells):
        return {(x + dx, y + dy) for x, y in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)}

    def _clusters(self, burning: dict[Pos, float]) -> list[set[Pos]]:
        left, out = set(burning), []
        while left:
            stack = [left.pop()]
            comp = set(stack)
            while stack:
                x, y = stack.pop()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        q = (x + dx, y + dy)
                        if q in left:
                            left.remove(q)
                            comp.add(q)
                            stack.append(q)
            out.append(comp)
        return out

    def handle(self, t: int, m) -> bool:
        c = next((c for c in self.contracts.values() if c.round and c.round["cid"] == m.cid), None)
        if m.perf == Perf.PROPOSE and m.ctype == "Bid":
            if c is not None:
                c.round["bids"][m.sender] = float(m.content["cost"])
            else:
                self._send(Perf.REJECT_PROPOSAL, m.sender, {"type": "Award", "reason": "late"}, cid=m.cid)
            return True
        if m.perf == Perf.REFUSE and m.ctype == "Bid":
            return True
        if m.ctype in ("Commit", "Result", "Task") and m.perf in (Perf.INFORM, Perf.FAILURE):
            for c in self.contracts.values():
                if m.content.get("task_id") == c.id:
                    if m.perf == Perf.FAILURE or m.ctype == "Result":
                        c.crews.pop(m.sender, None)
            return True
        return False

    def act(self, t: int) -> Pos | None:
        cfg = self.drone.cfg
        n_crews = len(self.drone.K.crews)
        hover = None
        for c in self.contracts.values():
            r = c.round
            if r and t >= r["deadline"]:
                need = c.demand(cfg.cells_per_crew, n_crews) - len(c.crews)
                winners = [crew for crew, _ in sorted(r["bids"].items(), key=lambda kv: kv[1])][:max(0, need)]
                for crew, cost in r["bids"].items():
                    if crew in winners:
                        c.crews[crew] = t
                        self.stats["awards"] += 1
                        self._send(Perf.ACCEPT_PROPOSAL, crew, {"type": "Award", "task": self._content(c), "price": cost},
                                   cid=r["cid"])
                        self.drone.note(t, f"контракт {c.id} → {crew}")
                    else:
                        self._send(Perf.REJECT_PROPOSAL, crew, {"type": "Award", "task_id": c.id}, cid=r["cid"])
                if not winners:
                    self.stats["no_bids"] += 1
                    c.recruit = True
                else:
                    c.recruit = False
                c.round = None
            need = c.demand(cfg.cells_per_crew, n_crews) - len(c.crews)
            close = c.cells and min(cheb(p, self.api.pos) for p in c.cells) <= cfg.drone_radius + 2
            if c.round is None and need > 0 and t >= c.cooldown and (close or c.recruit):
                cid = f"cfp-{c.id}-{t}"
                c.round = {"cid": cid, "deadline": t + cfg.bid_deadline + 1, "bids": {}}
                self._send(Perf.CFP, list(self.drone.K.crews), self._content(c), cid=cid)
                self.stats["rounds"] += 1
            if c.crews and t - c.t_update >= 5 and c.cells:
                self._send(Perf.INFORM, list(c.crews), {"type": "TaskUpdate", "task_id": c.id,
                                                       "cells": [list(p) for p in sorted(c.cells)[:60]], "t_obs": t})
                c.t_update = t
            if hover is None and c.cells and need > 0 and c.recruit and t - c.t_seen <= 40:
                hover = self.drone.K.base
            elif (c.round is not None or (need > 0 and t - c.t_created < self.HOVER)) and hover is None and c.cells:
                pos = self.api.pos
                hover = min(c.cells, key=lambda p: cheb(p, pos))
        return hover
