from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .env import Phase

if TYPE_CHECKING:
    from .sim import Simulation


def gini(values: list[float]) -> float:
    v = np.asarray(values, dtype=float)
    if len(v) == 0 or v.sum() == 0:
        return 0.0
    diff = np.abs(v[:, None] - v[None, :]).sum()
    return float(diff / (2 * len(v) * v.sum()))


class Metrics:
    def __init__(self, sim: "Simulation"):
        self.sim = sim
        self.series: list[dict] = []

    def _damage(self) -> dict:
        env = self.sim.env
        ph, val = env.fire.phase, env.terrain.value
        burnt = ph == Phase.BURNT
        burning = ph == Phase.BURNING
        lost = 0
        for s in env.terrain.settlements:
            if any(burnt[y, x] or burning[y, x] for x, y in s):
                lost += 1
        return {
            "burning": int(burning.sum()),
            "burnt_cells": int(burnt.sum()),
            "burnt_value": round(float(val[burnt].sum()), 2),
            "burning_value": round(float(val[burning].sum()), 2),
            "settlements_lost": lost,
            "extinguished_cells": int((ph == Phase.EXTINGUISHED).sum()),
        }

    def on_tick(self, t: int) -> None:
        d = self._damage()
        st = self.sim.bus.stats
        d.update({"t": t, "msgs_sent": st["sent"], "msgs_delivered": st["deliveries"]})
        self.series.append(d)

    def live(self) -> dict:
        return self.series[-1] if self.series else {}

    def summary(self) -> dict:
        sim = self.sim
        cfg, env, bus = sim.cfg, sim.env, sim.bus
        d = self._damage()
        T = env.t
        incs = list(env.fire.incidents.values())
        ext = [i.t_ext - i.t_ign for i in incs if i.t_ext is not None]
        unext = [T - i.t_ign for i in incs if i.t_ext is None]
        det = [i.t_detect - i.t_ign for i in incs if i.t_detect is not None]
        resp = [i.t_first_attack - i.t_detect for i in incs if i.t_first_attack is not None and i.t_detect is not None]
        res = {
            "seed": cfg.seed, "scenario": cfg.scenario, "mechanism": cfg.mechanism if cfg.agents_enabled else "none",
            "hq_mode": cfg.hq_mode, "T": T,
            "J_burn": round(d["burnt_value"] + d["burning_value"], 2),
            "burnt_cells": d["burnt_cells"], "burning_at_T": d["burning"],
            "J_set": d["settlements_lost"],
            "incidents": len(incs), "incidents_extinguished": len(ext),
            "T_ext_mean": round(float(np.mean(ext)), 2) if ext else None,
            "T_ext_max": int(max(ext)) if ext else None,
            "T_unextinguished_age_mean": round(float(np.mean(unext)), 2) if unext else 0,
            "detect_latency_mean": round(float(np.mean(det)), 2) if det else None,
            "response_time_mean": round(float(np.mean(resp)), 2) if resp else None,
            "M_sent": bus.stats["sent"], "M_delivered": bus.stats["deliveries"], "M_lost": bus.stats["lost"],
            "bytes": bus.stats["bytes"],
            "by_perf": dict(bus.stats["by_perf"]), "by_type": dict(bus.stats["by_type"]),
            "by_channel": dict(bus.stats["by_channel"]),
        }
        hq = sim.agents.get("h")
        if hq is not None:
            res.update({
                "coord_time_ms": round(hq.coord_time * 1000, 2),
                "requests": hq.stats["requests"], "refusals": hq.stats["refusals"],
                "failures": hq.stats["failures"], "reassignments": hq.stats["reassignments"],
                "cancels": hq.stats["cancels"], "tasks_created": len(hq.tasks),
                "coordinator": hq.coordinator.summary(),
            })
            crews = [a for a in sim.agents.values() if a.kind == "crew"]
            busy = [c.busy_ticks for c in crews]
            waits = [w for a in sim.agents.values() if a.kind == "water" for w in a.wait_times]
            res.update({
                "crew_busy": {c.id: c.busy_ticks for c in crews},
                "gini_busy": round(gini(busy), 3),
                "water_used": int(sum(b.stats["water_used"] for b in env.bodies.values() if b.kind == "crew")),
                "cells_extinguished_by_crews": int(sum(b.stats["extinguished"] for b in env.bodies.values()
                                                       if b.kind == "crew")),
                "refill_wait_mean": round(float(np.mean(waits)), 2) if waits else 0.0,
                "refill_wait_max": int(max(waits)) if waits else 0,
                "danger_events": int(sum(c.danger_events for c in crews)),
            })
        return res
