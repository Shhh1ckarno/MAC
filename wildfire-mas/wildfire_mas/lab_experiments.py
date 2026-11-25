"""Reproducible experiments for refill arbitration and simulation reliability."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np

from .config import SimConfig
from .experiments import paired_diff
from .llm import LLMCoordinator, LLMError, LLMReply
from .sim import Simulation

LAB3_POLICIES = ("fifo", "priority")
LAB4_CASES = ("bad_report", "water_outage", "llm_outage")
LAB4_MODES = ("normal", "before", "after")
URGENT_PRIORITY = 8
LLM_OUTAGE_TICKS = 30


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def _p95(values):
    return float(np.percentile(values, 95)) if values else None


def _write_results(out, lab, rows, summary):
    with (out / f"{lab}_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / f"{lab}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def lab3_config(seed: int, policy: str, horizon: int = 300) -> SimConfig:
    return SimConfig(seed=seed, horizon=horizon, scenario="outbreak", n_water=1, n_crews=8,
                     crew_tank=8, refill_policy=policy)


def _lab3_job(job):
    seed, policy, trace = job
    sim = Simulation(lab3_config(seed, policy), trace_path=trace)
    result = sim.run()
    post = sim.agents["w1"]
    groups = {
        "ordinary": [w for priority, w in post.waits_by_priority if priority < URGENT_PRIORITY],
        "urgent": [w for priority, w in post.waits_by_priority if priority >= URGENT_PRIORITY],
    }
    return {
        "seed": seed, "policy": policy, "requests": post.requests, "served": post.served,
        "wait_mean": _mean(post.wait_times), "peak_serving": post.peak_serving, "peak_waiting": post.peak_waiting,
        **{k: result[k] for k in ("M_sent", "J_burn", "J_set")},
        "groups": groups,
    }


def _paired(before, after, field):
    ref = {r["seed"]: r[field] for r in before if r[field] is not None}
    pairs = [(ref[r["seed"]], r[field]) for r in after if r["seed"] in ref and r[field] is not None]
    if not pairs:
        return {"n": 0, "difference": None, "low95": None, "high95": None}
    baseline, treatment = zip(*pairs)
    d, low, high = paired_diff(treatment, baseline)
    return {"n": len(pairs), "difference": d, "low95": low, "high95": high}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def run_lab3(out: Path, n_seeds: int, workers: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, policy, out / f"{policy}.jsonl" if seed == 1 else None)
            for seed, policy in product(range(1, n_seeds + 1), LAB3_POLICIES)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_lab3_job, jobs))
    by_policy = {p: [r for r in rows if r["policy"] == p] for p in LAB3_POLICIES}
    choices = {}
    for policy, group in by_policy.items():
        choices[policy] = {
            **{k: _mean(r[k] for r in group) for k in ("M_sent", "J_burn", "J_set")},
            "unfinished_rate": _rate(sum(r["requests"] - r["served"] for r in group), sum(r["requests"] for r in group)),
            "peak_serving": max(r["peak_serving"] for r in group),
            "runs_with_contention": sum(r["peak_waiting"] >= 2 for r in group),
        }
        for name in ("ordinary", "urgent"):
            waits = [w for r in group for w in r["groups"][name]]
            choices[policy][name] = {"n": len(waits), "mean": _mean(waits), "p95": _p95(waits)}
    summary = {
        "seeds": n_seeds, "urgent_priority": URGENT_PRIORITY, "policies": choices,
        "paired": {k: _paired(by_policy["fifo"], by_policy["priority"], k) for k in ("J_burn", "wait_mean")},
    }
    for row in rows:
        for group, waits in row.pop("groups").items():
            row.update({f"{group}_n": len(waits), f"{group}_mean": _mean(waits), f"{group}_p95": _p95(waits)})
    _write_results(out, "lab3", rows, summary)
    return summary


@dataclass
class Fault:
    case: str
    llm_retry_ticks: int
    enabled: bool = True
    start: int | None = None
    recovery: int | None = None
    post: str | None = None
    crew: str | None = None
    message: int | None = None
    last_llm_error: int | None = None
    violations: int = 0
    tick: int = 0

    def before_delivery(self, sim: Simulation, t: int) -> None:
        self.tick = t
        if not self.enabled or self.case != "water_outage":
            return
        if self.start is None:
            for agent in sim.agents.values():
                if agent.kind == "water" and agent.serving:
                    self.start, self.post, self.crew = t, agent.id, next(iter(agent.serving))
                    sim.env.bodies[self.post].broken = True
                    sim.record("fault", case=self.case, post=self.post, crew=self.crew)
                    break
        elif t == self.start + 1 and sim.agents[self.post].serving:
            self.violations += 1
            sim.record("violation", case=self.case, reason="reservation_retained", post=self.post)

    def after_delivery(self, sim: Simulation, inbox: dict) -> None:
        if not self.enabled or self.case != "bad_report" or self.start is not None:
            return
        for i, message in enumerate(inbox.get("h", [])):
            if message.ctype == "FireReport":
                payload = {k: v for k, v in message.content.items() if k != "area"}
                inbox["h"][i] = replace(message, content=payload)
                self.start, self.message = self.tick, message.id
                sim.record("fault", case=self.case, message=self.message, sender=message.sender)
                return

    def observe(self, entry: dict) -> None:
        if self.start is None:
            return
        event, t = entry["event"], entry["t"]
        if self.case == "bad_report" and event == "report_rejected" and entry["message"] == self.message:
            self.recovery = t
        elif (self.case == "water_outage" and event == "refill_done" and self.recovery is None
              and entry["crew"] == self.crew and entry["agent"] != self.post):
            self.recovery = t
        elif self.case == "llm_outage":
            if event == "llm_error":
                if self.last_llm_error is not None and t - self.last_llm_error < self.llm_retry_ticks:
                    self.violations += 1
                self.last_llm_error = t
            elif event == "llm_response" and self.recovery is None:
                self.recovery = t


class ScenarioClient:
    """A deterministic client for the controlled external-service outage."""

    def __init__(self, fault: Fault):
        self.fault = fault
        self.sim = None

    def chat(self, system, user, schema):
        fault = self.fault
        if fault.enabled:
            if fault.start is None:
                fault.start = fault.tick
                self.sim.record("fault", case=fault.case, duration=LLM_OUTAGE_TICKS)
            if fault.tick < fault.start + LLM_OUTAGE_TICKS:
                raise LLMError("Сервис временно недоступен в сценарии")
        data = json.loads(user.partition("\n\nДОНЕСЕНИЯ")[0].split("\n", 1)[1])
        crews = {crew["id"]: crew["eta"] for crew in data["free_crews"]}
        assignments = []
        for task in sorted(data["open_tasks"], key=lambda task: -task["priority"]):
            tid = task["id"]
            closest = sorted(crews, key=lambda crew: (crews[crew][tid], crew))
            for crew in closest[:task["open_slots"]]:
                assignments.append({"crew": crew, "task": tid})
                del crews[crew]
        return LLMReply(json.dumps({"assignments": assignments, "comment": ""}), 0.0, {})


def make_lab4_simulation(seed: int, case: str, mode: str, trace=None, *, horizon: int = 300) -> Simulation:
    if case not in LAB4_CASES or mode not in LAB4_MODES:
        raise ValueError("Неизвестный сценарий или режим отказа")
    is_water = case == "water_outage"
    is_llm = case == "llm_outage"
    cfg = SimConfig(seed=seed, horizon=horizon, hq_mode="llm" if is_llm else "rules",
                    reliability_guards=mode == "after")
    if is_water:
        cfg = replace(cfg, scenario="outbreak", n_water=2, crew_tank=8)
    fault = Fault(case, cfg.llm_retry_ticks, enabled=mode != "normal")
    client = ScenarioClient(fault) if is_llm else None
    coordinator = LLMCoordinator(cfg, client) if is_llm else None
    sim = Simulation(cfg, trace_path=trace, coordinator=coordinator, fault=fault)
    if client:
        client.sim = sim
    return sim


def record_report_failure(sim: Simulation, exc: Exception) -> str | None:
    fault = sim.fault
    if (fault and fault.case == "bad_report" and not sim.cfg.reliability_guards
            and fault.start == sim.t and isinstance(exc, KeyError) and exc.args == ("area",)):
        fault.violations += 1
        return "Штаб остановился: в донесении отсутствует область наблюдения"
    return None


def _lab4_job(job):
    seed, case, mode, trace = job
    sim = make_lab4_simulation(seed, case, mode, trace)
    fault = sim.fault
    coordinator = sim.agents["h"].coordinator if case == "llm_outage" else None
    try:
        result = sim.run()
        error = None
    except KeyError as exc:
        if record_report_failure(sim, exc) is None:
            raise
        result = {}
        error = f"{type(exc).__name__}: {exc}"
    return {
        "seed": seed, "case": case, "mode": mode, "completed": error is None, "error": error,
        "fault_tick": fault.start, "recovery_tick": fault.recovery,
        "recovery_delay": fault.recovery - fault.start if fault.recovery is not None else None,
        "violations": fault.violations,
        **{k: result.get(k) for k in ("T", "M_sent", "J_burn", "J_set")},
        "llm_errors": coordinator.stats["errors"] if coordinator else None,
        "fallbacks": coordinator.stats["fallbacks"] if coordinator else None,
    }


def run_lab4(out: Path, n_seeds: int, workers: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, case, mode, out / f"{case}-{mode}.jsonl" if seed == 1 and mode != "normal" else None)
            for case, mode, seed in product(LAB4_CASES, LAB4_MODES, range(1, n_seeds + 1))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_lab4_job, jobs))
    cases = {}
    for case in LAB4_CASES:
        by_mode = {m: [r for r in rows if r["case"] == case and r["mode"] == m] for m in LAB4_MODES}
        modes = {}
        for mode, group in by_mode.items():
            modes[mode] = {
                "completed": sum(r["completed"] for r in group),
                "triggered": sum(r["fault_tick"] is not None for r in group),
                "recovered": sum(r["recovery_tick"] is not None for r in group),
                "runs_with_violations": sum(r["violations"] > 0 for r in group),
                "violations": sum(r["violations"] for r in group),
                **{k: _mean(r[k] for r in group if r[k] is not None)
                   for k in ("recovery_delay", "J_burn", "J_set", "llm_errors", "fallbacks")},
            }
        cases[case] = {
            "modes": modes,
            "paired_damage": _paired(by_mode["before"], by_mode["after"], "J_burn"),
        }
    summary = {"seeds": n_seeds, "cases": cases}
    _write_results(out, "lab4", rows, summary)
    return summary
