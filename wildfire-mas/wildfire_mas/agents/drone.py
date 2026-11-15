from __future__ import annotations

import math

import numpy as np

from ..comm import Message, Perf
from ..coordination.drone_cnp import DroneContractManager
from ..env import Pos, cheb
from .base import Agent, AgentAPI, Knowledge


def wind_words(direction: float, speed: float) -> str:
    names = ["восток", "юго-восток", "юг", "юго-запад", "запад", "северо-запад", "север", "северо-восток"]
    k = int(round((direction % (2 * math.pi)) / (math.pi / 4))) % 8
    return f"ветер {speed:.0f} м/с, дует на {names[k]}"


class DroneAgent(Agent):
    kind = "drone"
    roles = ("ρ4 Разведчик",)

    def __init__(self, aid: str, api: AgentAPI, knowledge: Knowledge, index: int):
        super().__init__(aid, api, knowledge)
        ter = knowledge.terrain
        n = max(1, len(knowledge.drones))
        x0 = index * ter.width // n
        x1 = (index + 1) * ter.width // n - 1
        self.zone = (x0, 0, x1, ter.height - 1)
        self.last_seen = np.full((ter.height, ter.width), -1000, dtype=np.int32)
        self.target: Pos | None = None
        self.state = "PATROL"
        self.charge_left = 0
        self.last_report_t = -100
        self.last_report_cells: dict[Pos, float] = {}
        self.recent_fire_t = -100
        self.hb_offset = index
        self.cm = DroneContractManager(self) if knowledge.cfg.mechanism == "cnp_drone" else None

    def _need_return(self) -> bool:
        d = cheb(self.api.pos, self.K.base)
        return self.api.battery <= math.ceil(d / self.cfg.drone_speed) + self.cfg.drone_reserve

    def _choose_target(self, t: int) -> Pos:
        x0, y0, x1, y1 = self.zone
        best, best_s = None, -1e9
        px, py = self.api.pos
        step = max(2, self.cfg.drone_radius)
        for y in range(y0 + step // 2, y1 + 1, step):
            for x in range(x0 + step // 2, x1 + 1, step):
                age = t - self.last_seen[y, x]
                s = min(age, 200) - 0.7 * cheb((x, y), (px, py)) / self.cfg.drone_speed
                if s > best_s:
                    best, best_s = (x, y), s
        return best or self.K.base

    def step(self, t: int, inbox: list[Message]) -> None:
        api = self.api
        for m in inbox:
            if self.cm is not None and self.cm.handle(t, m):
                continue
            if m.perf == Perf.REQUEST and m.ctype == "Patrol":
                self.zone = tuple(m.content["zone"])
                self.target = None
                api.send(Perf.AGREE, m.sender, {"type": "Patrol", "zone": list(self.zone)}, reply=m.id, cid=m.cid)
                self.note(t, f"принял зону патрулирования {self.zone}")

        if api.broken:
            self.state = "BROKEN"
            return

        if self.state == "CHARGE":
            self.charge_left -= 1
            api.act(("charge",))
            if self.charge_left <= 0:
                self.state = "PATROL"
                self.note(t, "зарядка завершена")
            self._heartbeat(t)
            return

        obs = api.observe()
        burning = []
        for p, b, I in zip(obs.cells, obs.burning, obs.intensity):
            self.last_seen[p[1], p[0]] = t
            if b:
                burning.append((p, I))
        self._report(t, burning)
        hover = None
        if self.cm is not None:
            self.cm.update(t, dict(burning), obs.cells)
            hover = self.cm.act(t)

        if self.state != "RETURN" and self._need_return():
            self.state = "RETURN"
            self.note(t, f"низкий заряд ({api.battery}), возврат на базу")
        if self.state == "RETURN":
            if api.pos == self.K.base:
                self.state = "CHARGE"
                self.charge_left = self.cfg.drone_recharge
                api.act(("charge",))
            else:
                api.act(("fly", self.K.base))
            self._heartbeat(t)
            return

        if hover is not None:
            self.target = hover
            api.act(("fly", hover))
            self.state = "COORD"
            self._heartbeat(t)
            return
        if self.target is None or api.pos == self.target or t % 15 == 0 or self.state == "COORD":
            self.target = self._choose_target(t)
        api.act(("fly", self.target))
        self.state = "PATROL"
        self._heartbeat(t)

    def _report(self, t: int, burning: list[tuple[Pos, float]]) -> None:
        cur = {p: round(I, 2) for p, I in burning}
        if burning:
            self.recent_fire_t = t
        new = any(p not in self.last_report_cells or abs(self.last_report_cells[p] - I) > 0.3 for p, I in cur.items())
        cleared = any(p not in cur and cheb(p, self.api.pos) <= self.cfg.drone_radius - 1 for p in self.last_report_cells)
        if not (new or cleared or (burning and t - self.last_report_t >= 3)):
            return
        x, y = self.api.pos
        wd, ws = self.api.weather()
        note = (f"Наблюдаю {len(cur)} горящих клеток, макс. интенсивность {max(cur.values()):.1f}; {wind_words(wd, ws)}."
                if cur else "В зоне обзора огня нет.")
        content = {"type": "FireReport", "area": [x, y, self.cfg.drone_radius],
                   "cells": [[p[0], p[1], I] for p, I in cur.items()], "t_obs": t, "note": note}
        self.api.send(Perf.INFORM, self.K.hq_id, content)
        self.last_report_t = t
        for p in list(self.last_report_cells):
            if cheb(p, (x, y)) <= self.cfg.drone_radius and p not in cur:
                del self.last_report_cells[p]
        self.last_report_cells.update(cur)

    def _heartbeat(self, t: int) -> None:
        if (t + self.hb_offset) % self.cfg.hb_period == 0:
            self.api.send(Perf.INFORM, self.K.hq_id, {"type": "Status", "kind": "drone", "pos": list(self.api.pos),
                                                     "battery": self.api.battery, "state": self.state})

    def public_state(self) -> dict:
        return {**super().public_state(), "pos": list(self.api.pos), "battery": self.api.battery,
                "zone": list(self.zone), "target": list(self.target) if self.target else None}

