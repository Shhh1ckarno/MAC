from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import SimConfig
from .fire import FireModel, Phase
from .terrain import Cell, Pos, Terrain, generate_terrain


def cheb(a: Pos, b: Pos) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


@dataclass
class Body:
    id: str
    kind: str
    pos: Pos
    water: int = 0
    battery: int = 0
    broken: bool = False
    stats: dict = field(default_factory=lambda: {"moves": 0, "water_used": 0, "extinguished": 0})


@dataclass
class Observation:
    cells: list[Pos]
    burning: list[bool]
    intensity: list[float]
    burnt: list[bool]


class Environment:
    def __init__(self, cfg: SimConfig, seed_seq: np.random.SeedSequence, scenario: dict | None = None):
        self.cfg = cfg
        scenario = scenario or {}
        s_map, s_fire, s_ign, s_wind, s_sens = seed_seq.spawn(5)
        self.rng_map = np.random.default_rng(s_map)
        self.rng_fire = np.random.default_rng(s_fire)
        self.rng_ign = np.random.default_rng(s_ign)
        self.rng_wind = np.random.default_rng(s_wind)
        self.seed_sensors = s_sens

        self.terrain: Terrain = generate_terrain(self.rng_map, cfg.width, cfg.height, cfg.n_water)
        self.fire = FireModel(self.terrain, cfg)
        self.t = 0
        self.bodies: dict[str, Body] = {}

        T = cfg.horizon + 2
        d0 = cfg.wind_dir0 if cfg.wind_dir0 is not None else float(self.rng_wind.uniform(0, 2 * np.pi))
        dirs = np.empty(T); sp = np.empty(T)
        d, v = d0, cfg.wind_speed0
        for i in range(T):
            d = d + cfg.wind_revert * (d0 - d) + self.rng_wind.normal(0, cfg.wind_dir_sigma)
            v = float(np.clip(v + cfg.wind_revert * (cfg.wind_speed0 - v)
                              + self.rng_wind.normal(0, cfg.wind_speed_sigma), 0.0, cfg.wind_max))
            dirs[i], sp[i] = d, v
        self.wind_dir, self.wind_speed = dirs, sp

        self.ignition_schedule: dict[int, list[Pos]] = {}
        flam = self._ignition_candidates()
        explicit = scenario.get("fires")
        if explicit:
            for (t0, x, y) in explicit:
                self.ignition_schedule.setdefault(int(t0), []).append((int(x), int(y)))
        else:
            for _ in range(cfg.initial_fires):
                self.ignition_schedule.setdefault(0, []).append(flam[int(self.rng_ign.integers(len(flam)))])
        if scenario.get("poisson", True):
            for t in range(1, cfg.horizon):
                for _ in range(int(self.rng_ign.poisson(cfg.ignition_rate))):
                    self.ignition_schedule.setdefault(t, []).append(flam[int(self.rng_ign.integers(len(flam)))])
        self._apply_ignitions(0)

    def _ignition_candidates(self) -> list[Pos]:
        ter = self.terrain
        far = []
        sett = [p for s in ter.settlements for p in s] + [ter.base]
        for y in range(ter.height):
            for x in range(ter.width):
                if ter.kind[y, x] in (Cell.FOREST, Cell.MEADOW) and all(cheb((x, y), q) >= 5 for q in sett):
                    far.append((x, y))
        return far

    def _apply_ignitions(self, t: int) -> list[Pos]:
        done = []
        for (x, y) in self.ignition_schedule.get(t, []):
            if self.fire.ignite(x, y, t):
                done.append((x, y))
        return done

    def add_body(self, body: Body) -> None:
        self.bodies[body.id] = body

    def wind(self) -> tuple[float, float]:
        i = min(self.t, len(self.wind_dir) - 1)
        return float(self.wind_dir[i]), float(self.wind_speed[i])

    def observe(self, body: Body, radius: int, rng: np.random.Generator | None = None,
                p_fn: float = 0.0, p_fp: float = 0.0, noise: float = 0.0) -> Observation:
        x0, y0 = body.pos
        ter, fire = self.terrain, self.fire
        cells, burning, inten, burnt = [], [], [], []
        for y in range(max(0, y0 - radius), min(ter.height, y0 + radius + 1)):
            for x in range(max(0, x0 - radius), min(ter.width, x0 + radius + 1)):
                if (x - x0) ** 2 + (y - y0) ** 2 > radius * radius + radius:
                    continue
                b = fire.phase[y, x] == Phase.BURNING
                I = float(fire.I[y, x])
                if rng is not None:
                    if b and rng.random() < p_fn:
                        b, I = False, 0.0
                    elif not b and ter.combust[y, x] > 0 and rng.random() < p_fp:
                        b, I = True, 0.3
                    if b and noise > 0:
                        I = float(np.clip(I + rng.normal(0, noise), 0.05, 1.0))
                cells.append((x, y)); burning.append(bool(b)); inten.append(I)
                burnt.append(bool(fire.phase[y, x] == Phase.BURNT))
        return Observation(cells, burning, inten, burnt)

    def is_burning(self, p: Pos) -> bool:
        return self.fire.phase[p[1], p[0]] == Phase.BURNING

    def can_enter(self, p: Pos) -> bool:
        return self.terrain.passable(p) and not self.is_burning(p)

    def move_crew(self, body: Body, path: list[Pos]) -> int:
        if body.broken:
            return 0
        budget, steps = 1.0, 0
        for p in path:
            if cheb(p, body.pos) != 1 or not self.can_enter(p):
                break
            cost = float(self.terrain.move_cost[p[1], p[0]])
            if cost > budget + 1e-9:
                break
            budget -= cost
            body.pos = p
            steps += 1
        body.stats["moves"] += steps
        return steps

    def move_drone(self, body: Body, target: Pos) -> None:
        if body.broken or body.battery <= 0:
            return
        x, y = body.pos
        for _ in range(self.cfg.drone_speed):
            if (x, y) == target:
                break
            x += int(np.sign(target[0] - x)); y += int(np.sign(target[1] - y))
        x = int(np.clip(x, 0, self.terrain.width - 1)); y = int(np.clip(y, 0, self.terrain.height - 1))
        if (x, y) != body.pos:
            body.stats["moves"] += 1
        body.pos = (x, y)

    def extinguish(self, body: Body, target: Pos) -> bool:
        if body.broken or body.water <= 0 or cheb(body.pos, target) > 1 or not self.terrain.inside(target):
            return False
        x, y = target
        was_burning = self.fire.phase[y, x] == Phase.BURNING
        inc = int(self.fire.incident[y, x])
        put_out = self.fire.apply_water(x, y, 1.0)
        body.water -= 1
        body.stats["water_used"] += 1
        if put_out:
            body.stats["extinguished"] += 1
        if was_burning and inc in self.fire.incidents and self.fire.incidents[inc].t_first_attack is None:
            self.fire.incidents[inc].t_first_attack = self.t
        return True

    def refill(self, body: Body, amount: int) -> None:
        body.water = min(self.cfg.crew_tank, body.water + amount)

    def step_dynamics(self) -> list[Pos]:
        H, W = self.terrain.height, self.terrain.width
        u = self.rng_fire.random((H, W))
        wd, ws = self.wind()
        self.fire.step(self.t, wd, ws, u)
        self.t += 1
        return self._apply_ignitions(self.t)
