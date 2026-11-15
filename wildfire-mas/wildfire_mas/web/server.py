from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import HQ_MODES, MECHANISMS, SimConfig
from ..scenarios import SCENARIOS
from ..sim import Simulation

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="МАС тушения лесных пожаров")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

MECH_TITLES = {
    "central_greedy": "Централизованная диспетчеризация (жадная)",
    "central_hungarian": "Централизованная диспетчеризация (венгерский алгоритм)",
    "cnp": "Контрактная сеть (CNP)",
    "auction_first": "Аукцион (первая цена)",
    "auction_vickrey": "Аукцион (вторая цена, Викри)",
    "cnp_drone": "Контрактная сеть от дронов (без штаба)",
}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/options")
def options():
    return JSONResponse({
        "mechanisms": [{"id": m, "title": MECH_TITLES[m]} for m in MECHANISMS],
        "scenarios": [{"id": k, "title": v["title"]} for k, v in SCENARIOS.items()],
        "hq_modes": list(HQ_MODES),
        "llm_configured": bool(os.environ.get("LLM_BASE_URL")),
        "defaults": SimConfig().to_dict(),
    })


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.sim: Simulation | None = None
        self.playing = False
        self.delay = 0.15
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()

    async def send(self, payload: dict) -> None:
        await self.ws.send_json(payload)

    async def reset(self, cfg_dict: dict) -> None:
        self.playing = False
        try:
            cfg = SimConfig.from_dict({**SimConfig().to_dict(), **cfg_dict})
            self.sim = await asyncio.to_thread(Simulation, cfg)
        except Exception as e:
            await self.send({"type": "error", "message": str(e)})
            return
        await self.send(self.sim.init_payload())
        await self.send(self.sim.snapshot())

    async def step_once(self) -> None:
        async with self.lock:
            if self.sim is None or self.sim.done:
                return
            await asyncio.to_thread(self.sim.step)
            await self.send(self.sim.snapshot())
            if self.sim.done:
                self.playing = False
                await self.finish()

    async def finish(self) -> None:
        sim = self.sim
        summary = sim.metrics.summary()
        base_cfg = SimConfig(**{**sim.cfg.to_dict(), "agents_enabled": False, "hq_mode": "rules"})
        base = await asyncio.to_thread(lambda: Simulation(base_cfg).run())
        summary["J_burn_base"] = base["J_burn"]
        summary["Save"] = round(1 - summary["J_burn"] / base["J_burn"], 3) if base["J_burn"] else None
        await self.send({"type": "summary", "summary": summary})

    async def play_loop(self) -> None:
        while self.playing and self.sim is not None and not self.sim.done:
            await self.step_once()
            await asyncio.sleep(self.delay)

    async def handle(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        if cmd == "reset":
            await self.reset(msg.get("config", {}))
        elif cmd == "play":
            self.delay = float(msg.get("delay", self.delay))
            if not self.playing:
                self.playing = True
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
    s = Session(ws)
    try:
        while True:
            await s.handle(await ws.receive_json())
    except WebSocketDisconnect:
        s.playing = False
