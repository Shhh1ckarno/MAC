from __future__ import annotations

from dataclasses import dataclass, field

from ..comm import Perf
from .base import Coordinator


@dataclass
class AuctionRound:
    cid: str
    lots: list[str]
    deadline: int
    invited: set[str]
    bids: dict[str, dict[str, float]] = field(default_factory=dict)
    responded: set[str] = field(default_factory=set)


class AuctionCoordinator(Coordinator):
    def __init__(self, pricing: str = "first"):
        assert pricing in ("first", "vickrey")
        self.pricing = pricing
        self.name = f"auction_{pricing}"
        self.round: AuctionRound | None = None
        self.awarded: dict[str, int] = {}
        self.n_rounds = 0
        self.prices: list[dict] = []

    def engaged_crews(self, hq) -> set[str]:
        return set(self.awarded)

    def allocate(self, hq, t: int) -> None:
        r = self.round
        if r is not None and (t >= r.deadline or r.responded >= r.invited):
            self._close(hq, t, r)
            self.round = None
        for crew, ta in list(self.awarded.items()):
            if t - ta > 2 * hq.cfg.cmd_latency + 3:
                del self.awarded[crew]
        if self.round is not None:
            return
        lots = hq.open_tasks()
        free = hq.free_crews(t)
        if not lots or not free:
            return
        cid = f"auction-{t}"
        self.round = AuctionRound(cid, [tk.id for tk in lots], t + hq.cfg.bid_deadline + 1, set(free))
        content = {"type": "Lots", "lots": [{"task_id": tk.id, "cells": hq.task_cells_payload(tk, 30),
                                             "priority": tk.priority, "slots": tk.open_slots()} for tk in lots]}
        hq.api.send(Perf.CFP, free, content, cid=cid)
        self.n_rounds += 1
        hq.note(t, f"аукцион: {len(lots)} лотов, {len(free)} участников")

    def _close(self, hq, t: int, r: AuctionRound) -> None:
        won: dict[str, tuple[str, float]] = {}
        busy = {c for x in hq.tasks.values() if x.status != "DONE" for c in x.assigned}
        for task_id in r.lots:
            tk = hq.tasks.get(task_id)
            if tk is None or tk.status == "DONE":
                continue
            for _ in range(tk.open_slots()):
                cands = sorted((b[task_id], c) for c, b in r.bids.items()
                               if task_id in b and c not in won and c not in busy and c not in self.awarded)
                if not cands:
                    break
                bid, crew = cands[0]
                price = bid if self.pricing == "first" or len(cands) < 2 else cands[1][0]
                won[crew] = (task_id, price)
                hq.assign(t, tk, crew, via="AUCTION")
                self.prices.append({"t": t, "task": task_id, "crew": crew, "bid": bid, "price": price})
        for crew in r.bids:
            if crew in won:
                task_id, price = won[crew]
                self.awarded[crew] = t
                hq.api.send(Perf.ACCEPT_PROPOSAL, crew, {"type": "Award", "task": hq.task_content(hq.tasks[task_id]),
                                                         "price": price}, cid=r.cid)
                hq.note(t, f"лот {task_id} → {crew} (цена {price})")
            else:
                hq.api.send(Perf.REJECT_PROPOSAL, crew, {"type": "Award"}, cid=r.cid)

    def on_message(self, hq, t: int, m) -> None:
        r = self.round
        if m.perf == Perf.PROPOSE and m.ctype == "Bids":
            if r is not None and m.cid == r.cid:
                r.bids[m.sender] = {k: float(v) for k, v in m.content["bids"].items()}
                r.responded.add(m.sender)
            else:
                hq.api.send(Perf.REJECT_PROPOSAL, m.sender, {"type": "Award", "reason": "late"}, cid=m.cid)
        elif m.perf == Perf.REFUSE and m.ctype == "Bid":
            if r is not None and m.cid == r.cid:
                r.responded.add(m.sender)
        elif m.perf == Perf.INFORM and m.ctype == "Commit":
            self.awarded.pop(m.sender, None)
            tk = hq.tasks.get(m.content["task_id"])
            if tk and m.sender in tk.assigned:
                tk.assigned[m.sender] = "active"
                tk.status = "ACTIVE"
        elif m.perf == Perf.FAILURE and m.ctype == "Commit":
            self.awarded.pop(m.sender, None)
            tk = hq.tasks.get(m.content["task_id"])
            if tk and m.sender in tk.assigned:
                del tk.assigned[m.sender]

    def summary(self) -> dict:
        return {"rounds": self.n_rounds, "awards": len(self.prices)}
