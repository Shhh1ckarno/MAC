from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from ..config import SimConfig
from .terrain import Terrain


class Phase(IntEnum):
    UNBURNT = 0
    BURNING = 1
    BURNT = 2
    EXTINGUISHED = 3


DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
DELTA = [1.0, 1.0, 1.0, 1.0, 2 ** -0.5, 2 ** -0.5, 2 ** -0.5, 2 ** -0.5]


def _shift(a: np.ndarray, dx: int, dy: int, fill=0.0) -> np.ndarray:
    out = np.full_like(a, fill)
    H, W = a.shape
    ys_dst = slice(max(dy, 0), H + min(dy, 0))
    xs_dst = slice(max(dx, 0), W + min(dx, 0))
    ys_src = slice(max(-dy, 0), H + min(-dy, 0))
    xs_src = slice(max(-dx, 0), W + min(-dx, 0))
    out[ys_dst, xs_dst] = a[ys_src, xs_src]
    return out


@dataclass
class Incident:
    id: int
    t_ign: int
    origin: tuple[int, int]
    t_detect: int | None = None
    t_first_attack: int | None = None
    t_ext: int | None = None
    cells_burnt: int = 0


class FireModel:
    def __init__(self, terrain: Terrain, cfg: SimConfig):
        self.t = terrain
        self.cfg = cfg
        H, W = terrain.height, terrain.width
        self.phase = np.zeros((H, W), dtype=np.int8)
        self.I = np.zeros((H, W))
        self.fuel = terrain.fuel0.copy()
        self.moist = np.zeros((H, W))
        self.incident = np.full((H, W), -1, dtype=np.int32)
        self.incidents: dict[int, Incident] = {}
        self._next_inc = 0
        self.water_applied = np.zeros((H, W))

    def ignitable(self) -> np.ndarray:
        return ((self.phase == Phase.UNBURNT) | (self.phase == Phase.EXTINGUISHED)) \
            & (self.t.combust > 0) & (self.fuel > 0)

    def burning_mask(self) -> np.ndarray:
        return self.phase == Phase.BURNING

    def ignite(self, x: int, y: int, now: int, incident: int | None = None) -> bool:
        if not self.ignitable()[y, x]:
            return False
        if incident is None:
            incident = self._next_inc
            self._next_inc += 1
            self.incidents[incident] = Incident(incident, now, (x, y))
        self.phase[y, x] = Phase.BURNING
        self.I[y, x] = self.cfg.ignite_intensity
        self.incident[y, x] = incident
        return True

    def apply_water(self, x: int, y: int, units: float = 1.0) -> bool:
        self.water_applied[y, x] += units
        self.moist[y, x] = 1.0
        if self.phase[y, x] == Phase.BURNING:
            self.I[y, x] -= self.cfg.water_effect * units
            if self.I[y, x] <= 1e-9:
                self.I[y, x] = 0.0
                self.phase[y, x] = Phase.EXTINGUISHED
                return True
        return False

    def step(self, now: int, wind_dir: float, wind_speed: float, u: np.ndarray) -> list[tuple[int, int]]:
        c = self.cfg
        burning = self.phase == Phase.BURNING
        I_src = np.where(burning, self.I, 0.0)
        target_ok = self.ignitable()
        q = np.ones_like(self.I)
        wvx, wvy = np.cos(wind_dir), np.sin(wind_dir)
        base = c.p0 * self.t.combust * self.fuel * (1.0 - self.moist)
        for (dx, dy), delta in zip(DIRS, DELTA):
            src = _shift(I_src, dx, dy)
            if not src.any():
                continue
            cos_a = (dx * wvx + dy * wvy) / np.hypot(dx, dy)
            omega = np.exp(c.wind_c1 * wind_speed) * np.exp(c.wind_c2 * wind_speed * (cos_a - 1.0))
            p = np.minimum(1.0, base * src * omega * delta)
            q *= 1.0 - p
        new = target_ok & (u < 1.0 - q)
        new_cells = [(int(x), int(y)) for y, x in np.argwhere(new)]

        self.I[burning] = np.minimum(1.0, self.I[burning] + c.growth * self.fuel[burning])
        self.fuel[burning] = np.maximum(0.0, self.fuel[burning] - c.burn_rate * self.I[burning])
        out = burning & (self.fuel <= 1e-9)
        self.phase[out] = Phase.BURNT
        self.I[out] = 0.0
        for y, x in np.argwhere(out):
            inc = self.incidents.get(int(self.incident[y, x]))
            if inc:
                inc.cells_burnt += 1

        for (x, y) in new_cells:
            best, best_I = -1, -1.0
            for dx, dy in DIRS:
                sx, sy = x - dx, y - dy
                if 0 <= sx < self.t.width and 0 <= sy < self.t.height and burning[sy, sx] and self.I[sy, sx] > best_I:
                    best, best_I = int(self.incident[sy, sx]), self.I[sy, sx]
            self.phase[y, x] = Phase.BURNING
            self.I[y, x] = c.ignite_intensity
            self.incident[y, x] = best

        self.moist = np.maximum(0.0, self.moist - c.moisture_decay)

        alive = set(np.unique(self.incident[self.phase == Phase.BURNING]).tolist())
        for inc in self.incidents.values():
            if inc.t_ext is None and inc.id not in alive:
                inc.t_ext = now
        return new_cells
