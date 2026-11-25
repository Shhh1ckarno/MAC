from __future__ import annotations

import math

import numpy as np

from ..comm import Channel, Message, Perf
from ..env import Pos, cheb
from .base import Agent, AgentAPI, FireBelief, Knowledge, astar, path_cost

NEIGH8 = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]


class CrewAgent(Agent):
    kind = "crew"
    roles = ("ρ6 Исполнитель тушения",)

    def __init__(self, aid: str, api: AgentAPI, knowledge: Knowledge, index: int):
        super().__init__(aid, api, knowledge)
        ter = knowledge.terrain
        self.belief = FireBelief(ter)
        self.task: dict | None = None
        self.task_since = -1
        self.intention = "IDLE"
        self.state = "IDLE"
        self.path: list[Pos] = []
        self.path_goal: Pos | None = None
        self.slot: dict | None = None
        self.wp_wait: dict[str, int] = {}
        self.wp_unavailable_until: dict[str, int] = {}
        self.outstanding: dict | None = None
        self.hb_offset = index
        self.last_report_t = -100
        self.reported: dict[Pos, float] = {}
        self.busy_ticks = 0
        self.wait_ticks = 0
        self.danger_events = 0
        self.tasks_done = 0
        self.settle_cells = [p for s in ter.settlements for p in s]

    def _blocked(self) -> np.ndarray:
        return self.belief.burning

    def _plan_to(self, goals: set[Pos]) -> list[Pos] | None:
        goals = {g for g in goals if self.K.terrain.passable(g) and not self.belief.burning[g[1], g[0]]}
        if not goals:
            return None
        return astar(self.K.terrain, self.api.pos, goals, self._blocked())

    def _attack_goals(self, cells: list[Pos]) -> set[Pos]:
        goals = set()
        for (x, y) in cells:
            for dx, dy in NEIGH8:
                goals.add((x + dx, y + dy))
        return {g for g in goals if self.K.terrain.inside(g)}

    def task_cells(self) -> list[Pos]:
        if not self.task:
            return []
        cells = [tuple(c) for c in self.task["cells"] if self.belief.burning[c[1], c[0]]]
        if cells:
            near = {p for p in self.belief.burning_cells() if min(cheb(p, c) for c in cells[:40]) <= 3}
            cells = list(set(cells) | near)
        return cells

    def estimate_cost(self, cells: list[Pos]) -> float:
        if not cells:
            return math.inf
        pos = self.api.pos
        nearest = min(cells, key=lambda c: cheb(c, pos))
        path = astar(self.K.terrain, pos, self._attack_goals([nearest]), self._blocked(), max_expand=3000)
        eta = path_cost(self.K.terrain, path) if path is not None else 2.0 * cheb(pos, nearest)
        if self.api.water < max(4, self.cfg.crew_tank // 4):
            wp = min(self.K.water_points.values(), key=lambda w: cheb(w, pos))
            eta += cheb(pos, wp) + self.cfg.refill_time + cheb(wp, nearest)
        return round(eta, 2)

    def _free(self) -> bool:
        return self.task is None and not self.api.broken

    def _accept_task(self, t: int, task: dict, issuer: str, cid: str | None, ch: Channel = Channel.CMD) -> None:
        self.task = {"task_id": task["task_id"], "cells": [tuple(c) for c in task["cells"]],
                     "issuer": issuer, "priority": task.get("priority", 0), "cid": cid,
                     "hops": task.get("hops", 0), "ch": ch}
        self.task_since = t
        for c in self.task["cells"]:
            if self.belief.t_obs[c[1], c[0]] < t - 1:
                self.belief.update_cell(c, True, 0.5, t - 1, "task")
        self.path = []
        self.note(t, f"принял задачу {task['task_id']} от {issuer}")

    def _handle(self, t: int, m: Message) -> None:
        api = self.api
        ct = m.ctype
        if m.perf == Perf.INFORM and ct == "FireReport":
            self.belief.apply_area_report(tuple(m.content["area"]), m.content["cells"], m.content.get("t_obs", t), m.sender)
        elif m.perf == Perf.INFORM and ct == "TaskUpdate":
            if self.task and self.task["task_id"] == m.content["task_id"]:
                self.task["cells"] = [tuple(c) for c in m.content["cells"]]
                for c in self.task["cells"]:
                    if self.belief.t_obs[c[1], c[0]] < m.content.get("t_obs", t) - 2:
                        self.belief.update_cell(c, True, 0.5, m.content.get("t_obs", t) - 2, "hq")
        elif m.perf == Perf.REQUEST and ct == "Task":
            if self._free() and self.outstanding is None:
                self._accept_task(t, m.content, m.sender, m.cid)
                api.send(Perf.AGREE, m.sender, {"type": "Task", "task_id": m.content["task_id"]}, reply=m.id, cid=m.cid)
            else:
                reason = "broken" if api.broken else "busy"
                api.send(Perf.REFUSE, m.sender, {"type": "Task", "task_id": m.content["task_id"], "reason": reason},
                         reply=m.id, cid=m.cid)
        elif m.perf == Perf.CFP and ct in ("Task", "Lots"):
            self._pending_cfps.append(m)
        elif m.perf == Perf.ACCEPT_PROPOSAL:
            if self.outstanding and self.outstanding["cid"] == m.cid:
                self.outstanding = None
            if self._free():
                self._accept_task(t, m.content["task"], m.sender, m.cid, m.channel)
                api.send(Perf.INFORM, m.sender, {"type": "Commit", "task_id": m.content["task"]["task_id"]},
                         reply=m.id, cid=m.cid, channel=m.channel)
            else:
                api.send(Perf.FAILURE, m.sender, {"type": "Commit", "task_id": m.content["task"]["task_id"],
                                                  "reason": "busy"}, reply=m.id, cid=m.cid, channel=m.channel)
        elif m.perf == Perf.REJECT_PROPOSAL:
            if self.outstanding and self.outstanding["cid"] == m.cid:
                self.outstanding = None
        elif m.perf == Perf.CANCEL and ct == "Task":
            if self.task and self.task["task_id"] == m.content["task_id"]:
                self.note(t, f"задача {self.task['task_id']} снята ({m.content.get('reason', '')})")
                if m.content.get("reason") == "done":
                    self.tasks_done += 1
                self.task = None
                self.path = []
        elif ct in ("SlotGrant", "SlotStarted", "RefillDone") and m.channel == Channel.RES:
            self._handle_refill(t, m)

    def _handle_refill(self, t: int, m: Message) -> None:
        matches = self.slot and self.slot["wp"] == m.sender and self.slot["cid"] == m.cid
        if not matches:
            return
        if m.perf == Perf.AGREE and m.ctype == "SlotGrant":
            self.slot["granted"] = int(m.content["slot"])
            self.wp_wait[m.sender] = max(0, self.slot["granted"] - t)
        elif m.perf == Perf.INFORM and m.ctype == "SlotStarted":
            self.slot["started"] = True
            self.slot["deadline"] = t + self.cfg.refill_time + self.cfg.cmd_latency + 1
        elif m.perf in (Perf.REFUSE, Perf.CANCEL):
            if m.content.get("reason") == "out_of_service":
                self._avoid_post(t, m.sender, "out_of_service")
            self.slot = None
        elif m.perf == Perf.INFORM and m.ctype == "RefillDone":
            self.slot = None
            if self.intention == "REFILL":
                self.intention = "IDLE"
            self.note(t, f"заправлен на {m.sender}")

    def _avoid_post(self, t: int, wp: str, reason: str) -> None:
        if not self.cfg.reliability_guards:
            self.wp_wait[wp] = 999
            return
        if reason == "out_of_service":
            self.wp_unavailable_until[wp] = self.cfg.horizon + 1
        else:
            self.wp_unavailable_until[wp] = t + self.cfg.hb_timeout + self.cfg.noshow_timeout
        self.wp_wait.pop(wp, None)
        self.api.record("post_unavailable", post=wp, reason=reason)


    def _answer_cfps(self, t: int) -> None:
        cfps, self._pending_cfps = self._pending_cfps, []
        if self.outstanding and t > self.outstanding["expire"]:
            self.outstanding = None
        can_bid = self._free() and self.outstanding is None
        best = None
        if can_bid:
            scored = []
            for m in cfps:
                if m.ctype == "Task":
                    cost = self.estimate_cost([tuple(c) for c in m.content["cells"]])
                    if math.isfinite(cost):
                        scored.append((-(m.content.get("priority", 0)) / (1 + cost), m, cost))
                else:
                    scored.append((-1e9, m, None))
            if scored:
                scored.sort(key=lambda s: s[0])
                best = scored[0]
        for m in cfps:
            if best is not None and m is best[1]:
                if m.ctype == "Task":
                    self.api.send(Perf.PROPOSE, m.sender, {"type": "Bid", "task_id": m.content["task_id"],
                                                           "cost": best[2], "water": self.api.water},
                                  reply=m.id, cid=m.cid, channel=m.channel)
                else:
                    bids = {}
                    for lot in m.content["lots"]:
                        c = self.estimate_cost([tuple(x) for x in lot["cells"]])
                        if math.isfinite(c):
                            bids[lot["task_id"]] = c
                    self.api.send(Perf.PROPOSE, m.sender, {"type": "Bids", "bids": bids, "water": self.api.water},
                                  reply=m.id, cid=m.cid, channel=m.channel)
                self.outstanding = {"cid": m.cid, "expire": t + 2 * self.cfg.bid_deadline + 2}
            else:
                self.api.send(Perf.REFUSE, m.sender, {"type": "Bid", "reason": "busy" if not can_bid else "other"},
                              reply=m.id, cid=m.cid, channel=m.channel)

    def step(self, t: int, inbox: list[Message]) -> None:
        api = self.api
        self._pending_cfps: list[Message] = []
        for m in inbox:
            self._handle(t, m)
        if api.broken:
            if self.task:
                api.send(Perf.FAILURE, self.task["issuer"], {"type": "Task", "task_id": self.task["task_id"],
                                                             "reason": "broken"}, cid=self.task["cid"],
                         channel=self.task["ch"])
                self.task = None
            self.state = "BROKEN"
            self._answer_cfps(t)
            self._heartbeat(t, force=True)
            return

        obs = api.observe()
        for p, b, I in zip(obs.cells, obs.burning, obs.intensity):
            self.belief.update_cell(p, b, I, t, self.id)
        self._report(t, obs)
        self._answer_cfps(t)

        danger = self._danger()
        tcells = self.task_cells()
        if self.task and not tcells and t - self.task_since > 2:
            api.send(Perf.INFORM, self.task["issuer"], {"type": "Result", "task_id": self.task["task_id"],
                                                        "outcome": "done"}, cid=self.task["cid"], channel=self.task["ch"])
            self.note(t, f"задача {self.task['task_id']} выполнена")
            self.tasks_done += 1
            self.task = None
        low_water = api.water <= self.cfg.crew_refill_threshold
        at_wp = self.slot is not None and api.pos == self.K.water_points[self.slot["wp"]]
        if danger:
            self.intention = "RETREAT"
        elif low_water:
            self.intention = "REFILL"
        elif self.intention == "REFILL" and (not self.task or at_wp or api.water < self.cfg.crew_tank // 4):
            self.intention = "REFILL"
        elif self.task:
            if self.slot is not None:
                api.send(Perf.CANCEL, self.slot["wp"], {"type": "SlotRequest"},
                         channel=Channel.RES, cid=self.slot["cid"])
                self.slot = None
            self.intention = "EXTINGUISH"
        elif api.water < self.cfg.crew_tank // 2:
            self.intention = "REFILL"
        else:
            self.intention = "IDLE"

        if self.intention == "RETREAT":
            self.danger_events += 1
            self._retreat()
            self.intention = "EXTINGUISH" if self.task else "IDLE"
        elif self.intention == "REFILL":
            self._do_refill(t)
        elif self.intention == "EXTINGUISH":
            self._do_extinguish(t, tcells)
        elif self.cfg.mechanism == "cnp_drone" and cheb(api.pos, self.K.base) > 1:
            self.state = "TO_BASE"
            self._move_towards({self.K.base})
        else:
            self.state = "IDLE"
            api.act(("wait",))
        if self.task:
            self.busy_ticks += 1
        self._heartbeat(t)

    def _danger(self) -> bool:
        x, y = self.api.pos
        if self.belief.burning[y, x]:
            return True
        n = sum(1 for dx, dy in NEIGH8 if self.K.terrain.inside((x + dx, y + dy)) and self.belief.burning[y + dy, x + dx])
        return n >= 5

    def _retreat(self) -> None:
        x, y = self.api.pos
        best, best_n = None, 99
        for dx, dy in NEIGH8:
            p = (x + dx, y + dy)
            if not self.K.terrain.passable(p) or self.belief.burning[p[1], p[0]]:
                continue
            n = sum(1 for ex, ey in NEIGH8 if self.K.terrain.inside((p[0] + ex, p[1] + ey))
                    and self.belief.burning[p[1] + ey, p[0] + ex])
            if n < best_n:
                best, best_n = p, n
        self.state = "RETREAT"
        if best:
            self.api.act(("move", [best]))
        self.path = []

    def _choose_wp(self, t: int) -> str | None:
        available = (w for w in self.K.water_points if self.wp_unavailable_until.get(w, 0) <= t)
        return min(available, default=None,
                   key=lambda w: cheb(self.K.water_points[w], self.api.pos) + self.wp_wait.get(w, 0))

    def _do_refill(self, t: int) -> None:
        api = self.api
        if self.slot and self.cfg.reliability_guards and t >= self.slot["deadline"]:
            wp, cid = self.slot["wp"], self.slot["cid"]
            api.send(Perf.CANCEL, wp, {"type": "SlotRequest"}, channel=Channel.RES, cid=cid)
            self._avoid_post(t, wp, "timeout")
            self.slot = None
        if self.slot is None:
            wp = self._choose_wp(t)
            if wp is None:
                self.state = "WAIT_WATER"
                api.act(("wait",))
                return
            eta = t + cheb(self.K.water_points[wp], api.pos)
            cid = f"slot-{self.id}-{t}"
            deadline = eta + self.cfg.noshow_timeout + self.cfg.n_crews * self.cfg.refill_time
            deadline += 2 * max(1, self.cfg.cmd_latency)
            self.slot = {"wp": wp, "granted": None, "cid": cid, "started": False, "deadline": deadline}
            priority = self.task["priority"] if self.task else 0
            api.send(Perf.REQUEST, wp, {"type": "SlotRequest", "eta": eta,
                                       "amount": self.cfg.crew_tank - api.water, "priority": priority},
                     channel=Channel.RES, cid=cid)
        wp_pos = self.K.water_points[self.slot["wp"]]
        if api.pos == wp_pos:
            started = self.slot["started"]
            if self.cfg.refill_policy == "fifo" and not self.cfg.reliability_guards:
                granted = self.slot["granted"]
                started = granted is not None and granted <= t
            self.state = "REFILL" if started else "WAIT_SLOT"
            self.wait_ticks += 1
            api.act(("wait",))
            return
        self.state = "TO_WATER"
        self._move_towards({wp_pos})

    def _move_towards(self, goals: set[Pos]) -> None:
        goal_key = min(goals) if goals else None
        if self.api.pos in self.path:
            self.path = self.path[self.path.index(self.api.pos) + 1:]
        if self.path and cheb(self.path[0], self.api.pos) != 1:
            self.path = []
        if not self.path or self.path_goal != goal_key or any(self.belief.burning[p[1], p[0]] for p in self.path[:3]):
            self.path = self._plan_to(goals) or []
            self.path_goal = goal_key
        if self.path:
            self.api.act(("move", list(self.path[:3])))
        else:
            self.api.act(("wait",))

    def _target_score(self, p: Pos) -> float:
        ter = self.K.terrain
        n_front = 0
        for dx, dy in NEIGH8:
            q = (p[0] + dx, p[1] + dy)
            if ter.inside(q) and ter.combust[q[1], q[0]] > 0 and not self.belief.burning[q[1], q[0]]:
                n_front += 1
        d_s = min((cheb(p, s) for s in self.settle_cells), default=99)
        return n_front + 6.0 * math.exp(-d_s / 4.0) + 0.5 * self.belief.I[p[1], p[0]]

    def _do_extinguish(self, t: int, tcells: list[Pos]) -> None:
        api = self.api
        x, y = api.pos
        near = [(x + dx, y + dy) for dx, dy in NEIGH8
                if self.K.terrain.inside((x + dx, y + dy)) and self.belief.burning[y + dy, x + dx]]
        if near and api.water > 0:
            target = max(near, key=self._target_score)
            api.act(("extinguish", target))
            self.state = "EXTINGUISH"
            self.path = []
            return
        if not tcells:
            self.state = "IDLE"
            api.act(("wait",))
            return
        cand = sorted(tcells, key=lambda c: cheb(c, api.pos) - 0.5 * self._target_score(c))[:6]
        self.state = "TO_FIRE"
        self._move_towards(self._attack_goals(cand))

    def _report(self, t: int, obs) -> None:
        cur = {p: round(I, 2) for p, b, I in zip(obs.cells, obs.burning, obs.intensity) if b}
        new = any(p not in self.reported or abs(self.reported[p] - I) > 0.3 for p, I in cur.items())
        cleared = any(p not in cur and cheb(p, self.api.pos) <= self.cfg.crew_radius for p in self.reported)
        if not (new or cleared) or t - self.last_report_t < 2:
            return
        x, y = self.api.pos
        self.api.send(Perf.INFORM, self.K.hq_id, {"type": "FireReport", "area": [x, y, self.cfg.crew_radius],
                                                  "cells": [[p[0], p[1], I] for p, I in cur.items()], "t_obs": t,
                                                  "note": ""})
        self.last_report_t = t
        for p in list(self.reported):
            if cheb(p, (x, y)) <= self.cfg.crew_radius and p not in cur:
                del self.reported[p]
        self.reported.update(cur)

    def _heartbeat(self, t: int, force: bool = False) -> None:
        if (t + self.hb_offset) % self.cfg.hb_period == 0:
            self.api.send(Perf.INFORM, self.K.hq_id, {
                "type": "Status", "kind": "crew", "pos": list(self.api.pos), "water": self.api.water,
                "state": self.state, "task_id": self.task["task_id"] if self.task else None,
                "broken": self.api.broken})

    def public_state(self) -> dict:
        return {**super().public_state(), "pos": list(self.api.pos), "water": self.api.water,
                "task": self.task["task_id"] if self.task else None, "intention": self.intention,
                "path": [list(p) for p in self.path[:12]]}

