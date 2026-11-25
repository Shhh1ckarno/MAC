from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import HQ_MODES, MECHANISMS, SimConfig
from ..lab_experiments import (
    LAB3_POLICIES, record_report_failure, lab3_config, make_lab4_simulation,
)
from ..scenarios import SCENARIOS
from ..sim import Simulation

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="МАС тушения лесных пожаров")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

MECH_TITLES = {
    "central_greedy": "Жадный диспетчер",
    "central_hungarian": "Венгерский алгоритм",
    "cnp": "Контрактная сеть (CNP)",
    "auction_first": "Аукцион первой цены",
    "auction_vickrey": "Аукцион Викри",
    "cnp_drone": "Торги от дронов",
}
REFILL_TITLES = {"fifo": "По порядку заявок", "priority": "По приоритету"}
FAULT_TITLES = {
    "bad_report": "Некорректное донесение",
    "water_outage": "Отказ точки заправки",
    "llm_outage": "Недоступность LLM",
}
FAULT_MODES = {"normal": "Штатный режим", "before": "До защиты", "after": "После защиты"}


def choices(titles):
    return [{"id": key, "title": title} for key, title in titles.items()]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/options")
def options():
    return JSONResponse({
        "mechanisms": choices(MECH_TITLES),
        "scenarios": choices({key: value["title"] for key, value in SCENARIOS.items()}),
        "refill_policies": choices(REFILL_TITLES),
        "fault_cases": choices(FAULT_TITLES),
        "fault_modes": choices(FAULT_MODES),
        "hq_modes": list(HQ_MODES),
        "llm_configured": bool(os.environ.get("LLM_BASE_URL")),
        "defaults": SimConfig().to_dict(),
    })


def demo_simulation(lab: int, settings: dict) -> Simulation:
    if lab in (1, 2):
        return Simulation(SimConfig.from_dict(settings))
    seed = int(settings.get("seed", 1))
    horizon = int(settings.get("horizon", 300))
    if seed < 0 or not 1 <= horizon <= 2000:
        raise ValueError("Seed должен быть неотрицательным, горизонт должен быть от 1 до 2000")
    if lab == 3:
        policy = settings.get("refill_policy", "fifo")
        return Simulation(lab3_config(seed, policy, horizon))
    if lab == 4:
        return make_lab4_simulation(seed, settings.get("fault_case", "bad_report"),
                                    settings.get("fault_mode", "after"), horizon=horizon)
    raise ValueError("Неизвестная лабораторная работа")


def fault_state(sim: Simulation) -> dict:
    fault = sim.fault
    return {"case": fault.case, "enabled": fault.enabled, "start": fault.start,
            "recovery": fault.recovery, "violations": fault.violations}


def run_variant(sim: Simulation, label: str) -> dict:
    failure = None
    try:
        summary = sim.run()
    except KeyError as exc:
        failure = record_report_failure(sim, exc)
        if failure is None:
            raise
        summary = {}
    result = {
        "label": label, "failure": failure,
        "series": [[m["t"] + 1, round(m["burnt_value"] + m["burning_value"], 2)] for m in sim.metrics.series],
        "metrics": {key: summary.get(key) for key in (
            "J_burn", "J_set", "M_sent", "response_time_mean", "coord_time_ms", "refill_wait_mean",
        )},
    }
    if sim.fault:
        result["fault"] = fault_state(sim)
    return result


@app.post("/api/compare")
def compare(request: dict):
    lab = request.get("lab")
    settings = request.get("config", {})
    if not isinstance(settings, dict):
        raise HTTPException(422, "Настройки сравнения должны быть объектом")
    allowed = ("scenario", "seed", "horizon", "n_crews", "n_drones", "wind_speed0", "p_loss_cmd", "mechanism")
    config = {key: settings[key] for key in allowed if key in settings}
    try:
        seed = int(config.get("seed", 1))
        horizon = int(config.get("horizon", 300))
        if seed < 0 or not 1 <= horizon <= 2000:
            raise ValueError("Seed должен быть неотрицательным, горизонт должен быть от 1 до 2000")
        config.update(seed=seed, horizon=horizon)
        if lab == 2:
            current = config.get("mechanism", "central_greedy")
            other = request.get("alternative")
            if not isinstance(current, str) or current not in MECHANISMS or other not in MECHANISMS or other == current:
                raise ValueError("Выберите два разных механизма распределения задач")
            cfg = SimConfig.from_dict(config)
            if not 1 <= cfg.n_crews <= 15 or not 0 <= cfg.n_drones <= 10:
                raise ValueError("Допустимо от 1 до 15 расчетов и от 0 до 10 дронов")
            variants = [(Simulation(replace(cfg, mechanism=m)), MECH_TITLES[m]) for m in (current, other)]
        elif lab == 3:
            variants = [(Simulation(lab3_config(seed, p, horizon)), REFILL_TITLES[p]) for p in LAB3_POLICIES]
        elif lab == 4:
            case = settings.get("fault_case", "bad_report")
            variants = [(make_lab4_simulation(seed, case, mode, horizon=horizon), FAULT_MODES[mode])
                        for mode in ("before", "after")]
        else:
            raise ValueError("Сравнение доступно для ЛР 2, 3 и 4")
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"seed": seed, "horizon": horizon, "variants": [run_variant(sim, label) for sim, label in variants]}


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.sim: Simulation | None = None
        self.lab = 1
        self.failed = False
        self.playing = False
        self.delay = 0.15
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()

    async def send(self, payload: dict) -> None:
        await self.ws.send_json(payload)

    def live_llm(self):
        if self.lab in (1, 2) and self.sim.cfg.hq_mode == "llm":
            hq = self.sim.agents.get("h")
            return hq.coordinator if hq else None
        return None

    async def send_tick(self) -> None:
        sim = self.sim
        payload = sim.snapshot()
        if self.lab in (3, 4):
            payload["events"] = sim.tick_events
            payload["water"] = [agent.queue_snapshot(sim.t) for agent in sim.agents.values() if agent.kind == "water"]
        if sim.fault:
            payload["fault"] = fault_state(sim)
        coordinator = self.live_llm()
        if coordinator:
            client = coordinator.client
            payload["llm_status"] = {"configured": client.configured, "model": client.model,
                                     "http_attempts": client.http_attempts, "fallbacks": coordinator.stats["fallbacks"]}
        await self.send(payload)

    async def advance(self) -> None:
        coordinator = self.live_llm()
        if not coordinator or not coordinator.client.configured:
            await asyncio.to_thread(self.sim.step)
            return

        client = coordinator.client
        attempts = client.http_attempts
        t = self.sim.t
        work = asyncio.create_task(asyncio.to_thread(self.sim.step))
        while not (await asyncio.wait({work}, timeout=0.2))[0]:
            if client.http_attempts != attempts:
                attempts = client.http_attempts
                try:
                    await self.send({"type": "llm_request", "t": t, "model": client.model,
                                     "http_attempts": attempts})
                except WebSocketDisconnect:
                    break
        await work

    async def reset(self, cfg_dict: dict, lab: int) -> None:
        self.playing = False
        async with self.lock:
            try:
                sim = await asyncio.to_thread(demo_simulation, lab, cfg_dict)
            except (ValueError, TypeError) as exc:
                await self.send({"type": "error", "message": str(exc)})
                return
            self.lab, self.sim, self.failed = lab, sim, False
            await self.send({**sim.init_payload(), "lab": lab})
            await self.send_tick()

    async def step_once(self) -> None:
        async with self.lock:
            if self.sim is None or self.sim.done or self.failed:
                return
            try:
                await self.advance()
            except Exception as exc:
                self.failed = True
                self.playing = False
                failure = record_report_failure(self.sim, exc)
                if failure:
                    await self.send_tick()
                    await self.send({"type": "outcome", "message": failure, "t": self.sim.t})
                else:
                    await self.send({"type": "error", "message": str(exc)})
                return
            await self.send_tick()
            if self.sim.done:
                self.playing = False
                await self.finish()

    async def finish(self) -> None:
        sim = self.sim
        summary = sim.metrics.summary()
        if self.lab in (1, 2):
            base_cfg = replace(sim.cfg, agents_enabled=False, hq_mode="rules")
            base = await asyncio.to_thread(lambda: Simulation(base_cfg).run())
            summary["J_burn_base"] = base["J_burn"]
            summary["Save"] = round(1 - summary["J_burn"] / base["J_burn"], 3) if base["J_burn"] else None
        await self.send({"type": "summary", "summary": summary})

    async def play_loop(self) -> None:
        while self.playing and self.sim is not None and not self.sim.done and not self.failed:
            await self.step_once()
            if self.playing:
                await asyncio.sleep(self.delay)

    async def handle(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        if cmd == "reset":
            await self.reset(msg.get("config", {}), msg.get("lab", 1))
        elif cmd == "play":
            self.delay = float(msg.get("delay", self.delay))
            if self.sim and not self.sim.done and not self.failed:
                self.playing = True
                if self.task is None or self.task.done():
                    self.task = asyncio.create_task(self.play_loop())
        elif cmd == "pause":
            self.playing = False
        elif cmd == "step":
            self.playing = False
            await self.step_once()
        elif cmd == "speed":
            self.delay = float(msg.get("delay", self.delay))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session(ws)
    try:
        while True:
            await session.handle(await ws.receive_json())
    except WebSocketDisconnect:
        session.playing = False
        if session.task:
            session.task.cancel()
