from __future__ import annotations

from dataclasses import dataclass, field

from ..comm import Perf
from .base import Coordinator


@dataclass
class Round:
    cid: str
    task_id: str
    deadline: int
    invited: set[str]
    bids: dict[str, float] = field(default_factory=dict)
    responded: set[str] = field(default_factory=set)


class CNPCoordinator(Coordinator):
    name = "cnp"

    def __init__(self):
        self.rounds: dict[str, Round] = {}
        self.awarded: dict[str, int] = {}
        self.cooldown: dict[str, int] = {}
        self.n_rounds = 0
        self.n_bids = 0

    def engaged_crews(self, hq) -> set[str]:
        return set(self.awarded)

    def allocate(self, hq, t: int) -> None:
        for cid, r in list(self.rounds.items()):
            if t >= r.deadline or r.responded >= r.invited:
                self._close(hq, t, r)
                del self.rounds[cid]
        in_round = {r.task_id for r in self.rounds.values()}
        invited_now = {c for r in self.rounds.values() for c in r.invited}
        free = [c for c in hq.free_crews(t) if c not in invited_now]
        for tk in hq.open_tasks():
            if tk.id in in_round or not free or t < self.cooldown.get(tk.id, -1):
                continue
            cid = f"cfp-{tk.id}-{t}"
            self.rounds[cid] = Round(cid, tk.id, t + hq.cfg.bid_deadline + 1, set(free))
            hq.api.send(Perf.CFP, free, hq.task_content(tk), cid=cid)
            self.n_rounds += 1
            hq.note(t, f"CFP по {tk.id} → {len(free)} расчётов")
        for crew, ta in list(self.awarded.items()):
            if t - ta > 2 * hq.cfg.cmd_latency + 3:
                del self.awarded[crew]

    def _close(self, hq, t: int, r: Round) -> None:
        tk = hq.tasks.get(r.task_id)
        bidders = sorted(r.bids.items(), key=lambda kv: kv[1])
        winners = []
        if tk is not None and tk.status != "DONE":
            busy = {c for x in hq.tasks.values() if x.status != "DONE" for c in x.assigned}
            for crew, cost in bidders:
                if len(winners) >= tk.open_slots():
                    break
                if crew in busy or crew in self.awarded:
                    continue
                winners.append(crew)
        for crew, cost in bidders:
            if crew in winners:
                hq.assign(t, tk, crew, via="CNP")
                self.awarded[crew] = t
                hq.api.send(Perf.ACCEPT_PROPOSAL, crew, {"type": "Award", "task": hq.task_content(tk), "price": cost},
                            cid=r.cid)
                hq.note(t, f"контракт {tk.id} → {crew} (стоимость {cost})")
            else:
                hq.api.send(Perf.REJECT_PROPOSAL, crew, {"type": "Award", "task_id": r.task_id}, cid=r.cid)
        if not winners:
            self.cooldown[r.task_id] = t + 3

    def on_message(self, hq, t: int, m) -> None:
        r = self.rounds.get(m.cid) if m.cid else None
        if m.perf == Perf.PROPOSE and m.ctype == "Bid":
            if r is not None:
                r.bids[m.sender] = float(m.content["cost"])
                r.responded.add(m.sender)
                self.n_bids += 1
            else:
                hq.api.send(Perf.REJECT_PROPOSAL, m.sender, {"type": "Award", "reason": "late"}, cid=m.cid)
        elif m.perf == Perf.REFUSE and m.ctype == "Bid":
            if r is not None:
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
        return {"rounds": self.n_rounds, "bids": self.n_bids}
