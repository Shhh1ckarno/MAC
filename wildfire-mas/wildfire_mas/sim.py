from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .agents import AgentAPI, CrewAgent, DroneAgent, HQAgent, Knowledge, WaterPointAgent
from .comm import MessageBus
from .config import SimConfig
from .coordination import make_coordinator
from .env import Body, Environment, Phase
from .metrics import Metrics
from .scenarios import apply_scenario


class Simulation:
    def __init__(self, cfg: SimConfig, trace_path: str | Path | None = None, coordinator=None):
        cfg, scenario = apply_scenario(cfg.validate())
        self.cfg = cfg
        ss = np.random.SeedSequence(cfg.seed)
        s_env, s_bus, s_agents = ss.spawn(3)
        self.env = Environment(cfg, s_env, scenario)
        self.rng_agents = np.random.default_rng(s_agents)
        self.secrets: dict[str, str] = {}
        self.bus = MessageBus(cfg, self.env.bodies, np.random.default_rng(s_bus), self.secrets)
        self.agents: dict[str, object] = {}
        self.agent_rngs: dict[str, np.random.Generator] = {}
        self.order: list[str] = []
        if cfg.agents_enabled:
            self._create_agents(coordinator)
        self.metrics = Metrics(self)
        self._trace = open(trace_path, "w", encoding="utf-8") if trace_path else None
        self.events: list[str] = []

    def _create_agents(self, coordinator) -> None:
        cfg, ter = self.cfg, self.env.terrain
        hq_id = "h"
        drones = [f"d{i + 1}" for i in range(cfg.n_drones)]
        crews = [f"f{j + 1}" for j in range(cfg.n_crews)]
        wps = {f"w{k + 1}": ter.water_points[k] for k in range(min(cfg.n_water, len(ter.water_points)))}
        K = Knowledge(terrain=ter, hq_id=hq_id, drones=drones, crews=crews, water_points=wps, base=ter.base, cfg=cfg)

        def body(aid, kind, pos, **kw):
            b = Body(aid, kind, pos, **kw)
            self.env.add_body(b)
            self.secrets[aid] = "%016x" % int(self.rng_agents.integers(0, 2 ** 63))
            self.agent_rngs[aid] = np.random.default_rng(int(self.rng_agents.integers(0, 2 ** 63)))
            return AgentAPI(self, b)

        for k, (wid, pos) in enumerate(wps.items()):
            self.agents[wid] = WaterPointAgent(wid, body(wid, "water", pos), K, k)
        for i, d in enumerate(drones):
            battery = cfg.drone_battery - i * cfg.drone_battery // (cfg.n_drones + 1)
            self.agents[d] = DroneAgent(d, body(d, "drone", ter.base, battery=battery), K, i)
        for j, c in enumerate(crews):
            self.agents[c] = CrewAgent(c, body(c, "crew", ter.base, water=cfg.crew_tank), K, j)
        if coordinator is None:
            if cfg.hq_mode == "llm":
                from .llm import LLMCoordinator
                coordinator = LLMCoordinator(cfg)
            else:
                coordinator = make_coordinator(cfg.mechanism)
        self.agents[hq_id] = HQAgent(hq_id, body(hq_id, "hq", ter.base), K, coordinator)
        self.order = list(wps) + drones + crews + [hq_id]

    @property
    def t(self) -> int:
        return self.env.t

    @property
    def done(self) -> bool:
        return self.env.t >= self.cfg.horizon

    def step(self) -> None:
        t = self.env.t
        self.bus.new_tick()
        inbox = self.bus.deliver(t)
        self._observe_deliveries(t, inbox)
        for aid in self.order:
            agent = self.agents[aid]
            agent.api.pending_action = None
            agent.step(t, inbox.get(aid, []))
        self._apply_actions()
        self.env.step_dynamics()
        self.metrics.on_tick(t)
        if self._trace:
            self._write_trace(t)

    def run(self) -> dict:
        while not self.done:
            self.step()
        if self._trace:
            self._trace.close()
            self._trace = None
        return self.metrics.summary()

    def _observe_deliveries(self, t: int, inbox) -> None:
        fire = self.env.fire
        for m in inbox.get("h", []):
            if m.ctype == "FireReport":
                for c in m.content.get("cells", []):
                    inc = int(fire.incident[int(c[1]), int(c[0])])
                    if inc >= 0 and inc in fire.incidents and fire.incidents[inc].t_detect is None:
                        fire.incidents[inc].t_detect = t

    def _apply_actions(self) -> None:
        env, cfg = self.env, self.cfg
        for aid in self.order:
            agent = self.agents[aid]
            a = agent.api.pending_action
            b = env.bodies[aid]
            if b.kind == "drone":
                if a and a[0] == "charge" and b.pos == env.terrain.base:
                    b.battery = min(cfg.drone_battery, b.battery + math.ceil(cfg.drone_battery / cfg.drone_recharge))
                    continue
                if a and a[0] == "fly":
                    env.move_drone(b, tuple(a[1]))
                if not b.broken:
                    b.battery -= 1
                    if b.battery <= 0:
                        b.broken = True
                        self.events.append(f"t={env.t} {aid}: батарея разряжена — дрон потерян")
            elif b.kind == "crew" and a:
                if a[0] == "move":
                    env.move_crew(b, [tuple(p) for p in a[1]])
                elif a[0] == "extinguish":
                    env.extinguish(b, tuple(a[1]))
            elif b.kind == "water" and a and a[0] == "serve":
                crew = env.bodies.get(a[1])
                if crew is not None and crew.pos == b.pos:
                    env.refill(crew, int(a[2]))

    def init_payload(self) -> dict:
        ter = self.env.terrain
        return {
            "type": "init", "width": ter.width, "height": ter.height,
            "terrain": ter.kind.flatten().tolist(), "base": list(ter.base),
            "settlements": [[list(p) for p in s] for s in ter.settlements],
            "water_points": {wid: list(a.api.pos) for wid, a in self.agents.items() if a.kind == "water"},
            "config": self.cfg.to_dict(),
        }

    def snapshot(self) -> dict:
        env, fire = self.env, self.env.fire
        W = env.terrain.width
        ph = fire.phase
        cells = [[int(y * W + x), int(ph[y, x]), int(round(fire.I[y, x] * 9))] for y, x in np.argwhere(ph != Phase.UNBURNT)]
        wet = [int(y * W + x) for y, x in np.argwhere((fire.moist > 0.15) & (ph != Phase.BURNING))]
        hq = self.agents.get("h")
        belief = [int(y * W + x) for y, x in np.argwhere(hq.belief.burning)] if hq else []
        wd, ws = env.wind()
        agents = []
        for aid in self.order:
            a = self.agents[aid]
            s = a.public_state()
            s["broken"] = env.bodies[aid].broken
            agents.append(s)
        log = []
        for aid in self.order:
            log += self.agents[aid].log[-3:]
        return {
            "type": "tick", "t": env.t, "cells": cells, "wet": wet, "belief": belief,
            "wind": {"deg": round(math.degrees(wd) % 360, 1), "speed": round(ws, 1)},
            "agents": agents, "msgs": [m.brief() for m in self.bus.sent_log][-120:],
            "metrics": self.metrics.live(),
            "log": sorted(set(log), key=lambda s: int(s.split()[0][2:]))[-25:],
            "llm": getattr(hq.coordinator, "history", [])[-1:] if hq else [],
        }

    def _write_trace(self, t: int) -> None:
        rec = {"t": t, "metrics": self.metrics.live(),
               "agents": [self.agents[a].public_state() for a in self.order],
               "msgs": [{**m.brief(), "content": m.content} for m in self.bus.sent_log]}
        self._trace.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
