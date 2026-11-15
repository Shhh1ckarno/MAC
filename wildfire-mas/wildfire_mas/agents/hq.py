from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..comm import Message, Perf
from ..env import Pos, cheb
from .base import Agent, AgentAPI, FireBelief, Knowledge

NEIGH8 = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]


@dataclass
class Task:
    id: str
    cells: set[Pos]
    priority: float
    demand: int
    t_created: int
    status: str = "OPEN"
    assigned: dict[str, str] = field(default_factory=dict)
    t_assigned: dict[str, int] = field(default_factory=dict)
    t_first_assign: int | None = None
    t_update_sent: int = -100
    hops: int = 0
    reassignments: int = 0

    def open_slots(self) -> int:
        return max(0, self.demand - len(self.assigned))


@dataclass
class CrewInfo:
    pos: Pos
    water: int
    state: str = "IDLE"
    task_id: str | None = None
    t_last: int = 0
    alive: bool = True
    broken: bool = False


class HQAgent(Agent):
    kind = "hq"
    roles = ("ρ1 Координатор", "ρ2 Аналитик обстановки", "ρ3 Валидатор", "ρ5 Инициатор контракта")

    ALPHA, BETA, GAMMA_S, L = 1.0, 1.0, 0.25, 6.0

    def __init__(self, aid: str, api: AgentAPI, knowledge: Knowledge, coordinator):
        super().__init__(aid, api, knowledge)
        self.belief = FireBelief(knowledge.terrain)
        self.coordinator = coordinator
        self.tasks: dict[str, Task] = {}
        self._next_task = 1
        self.crews: dict[str, CrewInfo] = {c: CrewInfo(pos=knowledge.base, water=knowledge.cfg.crew_tank)
                                           for c in knowledge.crews}
        self.drones: dict[str, dict] = {d: {"t_last": 0, "alive": True} for d in knowledge.drones}
        self.water_status: dict[str, dict] = {}
        self.reputation: dict[str, float] = {a: 1.0 for a in knowledge.crews + knowledge.drones}
        self.dirty = False
        self.recent_notes: list[dict] = []
        self.events_since_decision = 0
        self.coord_time = 0.0
        self.stats = {"requests": 0, "refusals": 0, "failures": 0, "reassignments": 0, "cancels": 0}
        settle = knowledge.terrain.settlements
        self.settlements = [(s, float(sum(knowledge.terrain.value[y, x] for x, y in s))) for s in settle]
        self.state = "ACTIVE"

    def _handle(self, t: int, m: Message) -> None:
        ct = m.ctype
        if m.perf == Perf.INFORM and ct == "FireReport":
            if self.belief.apply_area_report(tuple(m.content["area"]), m.content["cells"], m.content.get("t_obs", t), m.sender):
                self.dirty = True
            note = m.content.get("note")
            if note:
                self.recent_notes.append({"t": t, "from": m.sender, "note": note})
                self.recent_notes = self.recent_notes[-12:]
        elif m.perf == Perf.INFORM and ct == "Status":
            kind = m.content.get("kind")
            if kind == "crew" and m.sender in self.crews:
                ci = self.crews[m.sender]
                ci.pos = tuple(m.content["pos"]); ci.water = m.content["water"]; ci.state = m.content["state"]
                ci.task_id = m.content.get("task_id"); ci.t_last = t; ci.broken = bool(m.content.get("broken"))
                ci.alive = not ci.broken
                if ci.broken:
                    self._release_crew(t, m.sender, "broken")
                else:
                    self._reconcile(t, m.sender, ci.task_id)
            elif kind == "drone" and m.sender in self.drones:
                self.drones[m.sender].update({"t_last": t, "alive": True, "pos": m.content["pos"],
                                              "battery": m.content["battery"], "state": m.content["state"]})
            elif kind == "water":
                self.water_status[m.sender] = {"t": t, "queue": m.content["queue"]}
        elif m.perf == Perf.AGREE and ct == "Task":
            task = self.tasks.get(m.content["task_id"])
            if task and m.sender in task.assigned:
                task.assigned[m.sender] = "active"
                task.status = "ACTIVE"
        elif m.perf == Perf.REFUSE and ct == "Task":
            self.stats["refusals"] += 1
            task = self.tasks.get(m.content["task_id"])
            if task and m.sender in task.assigned:
                del task.assigned[m.sender]
            if m.content.get("reason") == "broken" and m.sender in self.crews:
                self.crews[m.sender].alive = False
        elif m.perf == Perf.INFORM and ct == "Result":
            task = self.tasks.get(m.content["task_id"])
            if task and m.sender in task.assigned:
                del task.assigned[m.sender]
            self.events_since_decision += 1
        elif m.perf == Perf.FAILURE and ct == "Task":
            self.stats["failures"] += 1
            self._release_crew(t, m.sender, m.content.get("reason", "failure"))
            if m.content.get("reason") == "broken" and m.sender in self.crews:
                self.crews[m.sender].alive = False
        elif m.perf == Perf.AGREE and ct == "Patrol":
            pass
        else:
            self.coordinator.on_message(self, t, m)

    def _reconcile(self, t: int, crew: str, reported_task: str | None) -> None:
        for task in self.tasks.values():
            if crew in task.assigned and task.assigned[crew] == "active" and reported_task != task.id \
                    and t - task.t_assigned.get(crew, t) > 4:
                del task.assigned[crew]

    def _release_crew(self, t: int, crew: str, reason: str) -> None:
        for task in self.tasks.values():
            if crew in task.assigned:
                del task.assigned[crew]
                task.reassignments += 1
                self.stats["reassignments"] += 1
                self.note(t, f"{crew} снят с задачи {task.id} ({reason}), задача переоткрыта")

    def _priority(self, cells: set[Pos]) -> float:
        wd, ws = self.api.weather()
        n = len(cells)
        sI = float(sum(self.belief.I[y, x] for x, y in cells))
        cx = sum(x for x, _ in cells) / n; cy = sum(y for _, y in cells) / n
        threat = 0.0
        for s_cells, v in self.settlements:
            d = min(cheb(c, s) for c in cells for s in s_cells) if n <= 60 else \
                min(cheb((int(cx), int(cy)), s) for s in s_cells)
            sx = sum(x for x, _ in s_cells) / len(s_cells); sy = sum(y for _, y in s_cells) / len(s_cells)
            ang = math.atan2(sy - cy, sx - cx)
            cos_psi = math.cos(ang - wd) if ws > 1.0 else 0.0
            threat += v * math.exp(-d / self.L) * (1 + max(0.0, cos_psi))
        return round(self.ALPHA * n + self.BETA * sI + self.GAMMA_S * threat, 2)

    def _demand(self, cells: set[Pos]) -> int:
        return int(min(len(self.K.crews), max(1, math.ceil(len(cells) / self.cfg.cells_per_crew))))

    def _update_tasks(self, t: int) -> None:
        clusters = sorted(self.belief.clusters(), key=len, reverse=True)
        active = [tk for tk in self.tasks.values() if tk.status != "DONE"]
        matched: set[str] = set()
        for comp in clusters:
            best, best_ov = None, 0
            for tk in active:
                if tk.id in matched:
                    continue
                dil = {(x + dx, y + dy) for (x, y) in tk.cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
                ov = len(comp & dil)
                if ov > best_ov:
                    best, best_ov = tk, ov
            if best is not None:
                if best.cells != comp:
                    best.cells = comp
                best.priority = self._priority(comp)
                best.demand = self._demand(comp)
                matched.add(best.id)
            else:
                tid = f"T{self._next_task}"; self._next_task += 1
                tk = Task(tid, comp, self._priority(comp), self._demand(comp), t)
                self.tasks[tid] = tk
                matched.add(tid)
                self.events_since_decision += 1
                self.note(t, f"новый очаг {tid}: {len(comp)} кл., приоритет {tk.priority}")
        for tk in active:
            if tk.id not in matched:
                tk.status = "DONE"
                for crew in list(tk.assigned):
                    self.api.send(Perf.CANCEL, crew, {"type": "Task", "task_id": tk.id, "reason": "done"})
                    self.stats["cancels"] += 1
                tk.assigned.clear()
                self.events_since_decision += 1
                self.note(t, f"очаг {tk.id} ликвидирован (по данным карты)")

    def _release_surplus(self, t: int) -> None:
        for tk in self.tasks.values():
            if tk.status == "DONE":
                continue
            extra = len(tk.assigned) - (tk.demand + 1)
            if extra > 0:
                far = sorted(tk.assigned, key=lambda c: -min(cheb(self.crews[c].pos, p) for p in tk.cells))[:extra]
                for crew in far:
                    del tk.assigned[crew]
                    self.api.send(Perf.CANCEL, crew, {"type": "Task", "task_id": tk.id, "reason": "surplus"})
                    self.stats["cancels"] += 1

    def free_crews(self, t: int) -> list[str]:
        busy = {c for tk in self.tasks.values() if tk.status != "DONE" for c in tk.assigned}
        busy |= self.coordinator.engaged_crews(self)
        return [c for c, ci in self.crews.items() if ci.alive and not ci.broken and c not in busy]

    def alive_crews(self) -> list[str]:
        return [c for c, ci in self.crews.items() if ci.alive and not ci.broken]

    def open_tasks(self) -> list[Task]:
        return sorted([tk for tk in self.tasks.values() if tk.status != "DONE" and tk.open_slots() > 0],
                      key=lambda tk: -tk.priority)

    def task_cells_payload(self, tk: Task, limit: int = 60) -> list[list[int]]:
        cells = list(tk.cells)
        if len(cells) > limit:
            ter = self.K.terrain

            def front(p):
                return sum(1 for dx, dy in NEIGH8 if ter.inside((p[0] + dx, p[1] + dy))
                           and not self.belief.burning[p[1] + dy, p[0] + dx])
            cells = sorted(cells, key=front, reverse=True)[:limit]
        return [[x, y] for x, y in cells]

    def task_content(self, tk: Task) -> dict:
        n = len(tk.cells)
        cx = round(sum(x for x, _ in tk.cells) / n); cy = round(sum(y for _, y in tk.cells) / n)
        return {"type": "Task", "task_id": tk.id, "kind": "EXTINGUISH", "cells": self.task_cells_payload(tk),
                "center": [cx, cy], "priority": tk.priority, "hops": tk.hops}

    def cost_estimate(self, crew: str, tk: Task) -> float:
        ci = self.crews[crew]
        d = min(cheb(ci.pos, p) for p in tk.cells)
        cost = float(d)
        if ci.water < max(4, self.cfg.crew_tank // 4):
            wp = min(self.K.water_points.values(), key=lambda w: cheb(w, ci.pos))
            cost += cheb(ci.pos, wp) + self.cfg.refill_time + min(cheb(wp, p) for p in tk.cells)
        return cost

    def assign(self, t: int, tk: Task, crew: str, via: str = "REQUEST", cid: str | None = None) -> None:
        tk.assigned[crew] = "pending"
        tk.t_assigned[crew] = t
        if tk.t_first_assign is None:
            tk.t_first_assign = t
        if via == "REQUEST":
            self.api.send(Perf.REQUEST, crew, self.task_content(tk), cid=cid or f"{tk.id}-{crew}-{t}")
            self.stats["requests"] += 1

    def step(self, t: int, inbox: list[Message]) -> None:
        for m in inbox:
            self._handle(t, m)

        if t == 0:
            n = len(self.K.drones); W, H = self.K.terrain.width, self.K.terrain.height
            for i, d in enumerate(self.K.drones):
                zone = [i * W // n, 0, (i + 1) * W // n - 1, H - 1]
                self.api.send(Perf.REQUEST, d, {"type": "Patrol", "zone": zone}, cid=f"patrol-{d}")

        if t > self.cfg.hb_timeout:
            for c, ci in self.crews.items():
                if ci.alive and t - ci.t_last > self.cfg.hb_timeout:
                    ci.alive = False
                    self._release_crew(t, c, "timeout")
                    self.note(t, f"нет связи с {c} > {self.cfg.hb_timeout} тактов — считается недоступным")
            for d, di in self.drones.items():
                if di["alive"] and t - di["t_last"] > self.cfg.hb_timeout:
                    di["alive"] = False

        self._update_tasks(t)
        self._release_surplus(t)

        t0 = time.perf_counter()
        self.coordinator.allocate(self, t)
        self.coord_time += time.perf_counter() - t0

        for tk in self.tasks.values():
            if tk.status != "DONE" and tk.assigned and t - tk.t_update_sent >= 5:
                rcv = [c for c, s in tk.assigned.items() if s == "active"]
                if rcv:
                    self.api.send(Perf.INFORM, rcv, {"type": "TaskUpdate", "task_id": tk.id,
                                                     "cells": self.task_cells_payload(tk), "t_obs": t})
                    tk.t_update_sent = t
        self.state = f"ACTIVE · {self.coordinator.name}"

    def public_state(self) -> dict:
        return {**super().public_state(), "pos": list(self.K.base), "mechanism": self.coordinator.name,
                "tasks": [{"id": tk.id, "cells": len(tk.cells), "priority": tk.priority, "demand": tk.demand,
                           "assigned": dict(tk.assigned), "status": tk.status}
                          for tk in self.tasks.values() if tk.status != "DONE"],
                "believed_burning": len(self.belief.burning_cells())}
