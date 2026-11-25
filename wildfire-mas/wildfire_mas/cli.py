from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import fields

from .config import MECHANISMS, SimConfig
from .scenarios import SCENARIOS
from .sim import Simulation


def _parse_overrides(pairs: list[str]) -> dict:
    types = {f.name: f.type for f in fields(SimConfig)}
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        if k not in types:
            raise SystemExit(f"Неизвестный параметр {k}")
        default = getattr(SimConfig(), k)
        if isinstance(default, bool):
            out[k] = v.lower() in ("1", "true", "yes", "да")
        elif isinstance(default, int):
            out[k] = int(v)
        elif isinstance(default, float) or default is None:
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _seeds(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def cmd_run(a) -> None:
    cfg = SimConfig(seed=a.seed, scenario=a.scenario, mechanism=a.mechanism, hq_mode=a.hq,
                    agents_enabled=not a.no_agents, **_parse_overrides(a.set))
    t0 = time.perf_counter()
    res = Simulation(cfg, trace_path=a.trace).run()
    res["wall_time_s"] = round(time.perf_counter() - t0, 2)
    if a.baseline:
        base = Simulation(SimConfig(**{**cfg.to_dict(), "agents_enabled": False})).run()
        res["J_burn_base"] = base["J_burn"]
        res["Save"] = round(1 - res["J_burn"] / base["J_burn"], 3) if base["J_burn"] else None
        res["eta_comm"] = round((base["J_burn"] - res["J_burn"]) / res["M_delivered"], 4) if res["M_delivered"] else None
    print(json.dumps(res, ensure_ascii=False, indent=2))


FLAT_KEYS = ["seed", "scenario", "mechanism", "hq_mode", "J_burn", "J_burn_base", "Save", "eta_comm", "burnt_cells", "burning_at_T",
             "J_set", "incidents", "incidents_extinguished", "T_ext_mean", "T_ext_max", "detect_latency_mean",
             "response_time_mean", "M_sent", "M_delivered", "M_lost", "bytes", "coord_time_ms", "requests",
             "refusals", "failures", "reassignments", "cancels", "tasks_created", "gini_busy", "water_used",
             "cells_extinguished_by_crews", "refill_wait_mean", "refill_wait_max", "danger_events", "wall_time_s"]


def cmd_batch(a) -> None:
    mechs = a.mechanisms.split(",") if a.mechanisms != "all" else list(MECHANISMS)
    seeds = _seeds(a.seeds)
    over = _parse_overrides(a.set)
    rows = []
    for seed in seeds:
        base_cfg = SimConfig(seed=seed, scenario=a.scenario, agents_enabled=False, **over)
        base = Simulation(base_cfg).run()["J_burn"]
        for mech in mechs:
            cfg = SimConfig(seed=seed, scenario=a.scenario, mechanism=mech, **over)
            t0 = time.perf_counter()
            r = Simulation(cfg).run()
            r["wall_time_s"] = round(time.perf_counter() - t0, 2)
            r["J_burn_base"] = base
            r["Save"] = round(1 - r["J_burn"] / base, 3) if base else None
            r["eta_comm"] = round((base - r["J_burn"]) / r["M_delivered"], 4) if r["M_delivered"] else None
            rows.append({k: r.get(k) for k in FLAT_KEYS})
            print(f"seed={seed:3d} {mech:18s} J_burn={r['J_burn']:8.1f} (база {base:8.1f}) "
                  f"J_set={r['J_set']} M={r['M_delivered']:5d}", file=sys.stderr)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FLAT_KEYS)
        w.writeheader()
        w.writerows(rows)
    print(f"Сохранено {len(rows)} строк в {a.out}", file=sys.stderr)


def cmd_lab2(a) -> None:
    from pathlib import Path

    from .experiments import analyze, run_lab2
    out = Path(a.out)
    csv_path = out / "lab2_runs.csv"
    if not a.analyze_only:
        csv_path = run_lab2(out, a.seeds, a.workers)
    analyze(csv_path, out)


def cmd_lab2_llm(a) -> None:
    from pathlib import Path

    from .experiments import analyze_llm, run_llm_series
    if not a.analyze_only:
        run_llm_series(Path(a.out), a.seeds)
    analyze_llm(Path(a.out))


def cmd_lab3(a) -> None:
    from pathlib import Path

    from .lab_experiments import run_lab3
    run_lab3(Path(a.out), a.seeds, a.workers)
    print(f"Результаты ЛР 3: {a.out}")


def cmd_lab4(a) -> None:
    from pathlib import Path

    from .lab_experiments import run_lab4
    run_lab4(Path(a.out), a.seeds, a.workers)
    print(f"Результаты ЛР 4: {a.out}")


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("требуется положительное число")
    return number


def cmd_serve(a) -> None:
    import uvicorn
    uvicorn.run("wildfire_mas.web.server:app", host=a.host, port=a.port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="wildfire", description="МАС тушения лесных пожаров")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="один прогон")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--scenario", default="default", choices=list(SCENARIOS))
    r.add_argument("--mechanism", default="central_greedy", choices=list(MECHANISMS))
    r.add_argument("--hq", default="rules", choices=["rules", "llm"])
    r.add_argument("--no-agents", action="store_true", help="базовый прогон без агентов")
    r.add_argument("--baseline", action="store_true", help="дополнительно посчитать Save относительно прогона без агентов")
    r.add_argument("--trace", help="путь к файлу трассы (JSONL)")
    r.add_argument("--set", nargs="*", help="переопределить параметры: key=value")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("batch", help="серия прогонов → CSV")
    b.add_argument("--mechanisms", default="all", help="список через запятую или all")
    b.add_argument("--seeds", default="1-10", help="например 1-20 или 1,5,7")
    b.add_argument("--scenario", default="default", choices=list(SCENARIOS))
    b.add_argument("--out", default="results.csv")
    b.add_argument("--set", nargs="*")
    b.set_defaults(fn=cmd_batch)

    l2 = sub.add_parser("lab2", help="эксперименты ЛР 2: все механизмы, сценарии, масштаб, потери связи")
    l2.add_argument("--seeds", type=int, default=30)
    l2.add_argument("--workers", type=int, default=None)
    l2.add_argument("--out", default="results/lab2")
    l2.add_argument("--analyze-only", action="store_true")
    l2.set_defaults(fn=cmd_lab2)

    ll = sub.add_parser("lab2-llm", help="серия ЛР 2 со штабом на LLM (нужны LLM_BASE_URL и LLM_MODEL)")
    ll.add_argument("--seeds", type=int, default=15)
    ll.add_argument("--out", default="results/lab2")
    ll.add_argument("--analyze-only", action="store_true")
    ll.set_defaults(fn=cmd_lab2_llm)

    for name, title, handler in (
        ("lab3", "конфликт за место на заправке", cmd_lab3),
        ("lab4", "сценарии отказа и восстановление", cmd_lab4),
    ):
        lab = sub.add_parser(name, help=title)
        lab.add_argument("--seeds", type=_positive, default=30)
        lab.add_argument("--workers", type=_positive, default=4)
        lab.add_argument("--out", default=f"results/{name}")
        lab.set_defaults(fn=handler)

    s = sub.add_parser("serve", help="веб-интерфейс")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
