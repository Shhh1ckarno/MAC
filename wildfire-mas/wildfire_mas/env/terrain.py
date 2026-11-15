from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Cell(IntEnum):
    FOREST = 0
    MEADOW = 1
    ROAD = 2
    WATER = 3
    SETTLEMENT = 4
    BASE = 5


CELL_PROPS = {
    Cell.FOREST:     (1.0, 1.0, 1.0, 1.0),
    Cell.MEADOW:     (0.4, 1.3, 1.0, 0.3),
    Cell.ROAD:       (0.0, 0.0, 0.5, 0.0),
    Cell.WATER:      (0.0, 0.0, np.inf, 0.0),
    Cell.SETTLEMENT: (0.8, 0.8, 1.0, 20.0),
    Cell.BASE:       (0.0, 0.0, 0.5, 0.0),
}

Pos = tuple[int, int]


@dataclass
class Terrain:
    width: int
    height: int
    kind: np.ndarray
    fuel0: np.ndarray
    combust: np.ndarray
    move_cost: np.ndarray
    value: np.ndarray
    base: Pos
    water_points: list[Pos]
    settlements: list[list[Pos]] = field(default_factory=list)

    def inside(self, p: Pos) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def passable(self, p: Pos) -> bool:
        return self.inside(p) and np.isfinite(self.move_cost[p[1], p[0]])

    def flammable_cells(self) -> np.ndarray:
        return np.argwhere(self.combust > 0)


def _smooth_noise(rng: np.random.Generator, w: int, h: int, scale: int = 6) -> np.ndarray:
    gw, gh = w // scale + 2, h // scale + 2
    grid = rng.random((gh, gw))
    ys = np.linspace(0, gh - 1.001, h)
    xs = np.linspace(0, gw - 1.001, w)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = ys - y0, xs - x0
    a = grid[np.ix_(y0, x0)]
    b = grid[np.ix_(y0, x0 + 1)]
    c = grid[np.ix_(y0 + 1, x0)]
    d = grid[np.ix_(y0 + 1, x0 + 1)]
    top = a * (1 - fx) + b * fx
    bot = c * (1 - fx) + d * fx
    return top * (1 - fy[:, None]) + bot * fy[:, None]


def generate_terrain(rng: np.random.Generator, width: int, height: int, n_water_points: int,
                     n_settlements: int = 3) -> Terrain:
    W, H = width, height
    kind = np.full((H, W), Cell.FOREST, dtype=np.int8)

    noise = _smooth_noise(rng, W, H, scale=7)
    kind[noise > 0.68] = Cell.MEADOW

    x = int(rng.integers(W // 5, W // 3))
    for y in range(H):
        for dx in (0, 1):
            if 0 <= x + dx < W:
                kind[y, x + dx] = Cell.WATER
        x = int(np.clip(x + rng.integers(-1, 2), 2, W // 2 - 2))
    river_x = x

    cx, cy = int(rng.integers(W * 2 // 3, W - 5)), int(rng.integers(4, H // 2))
    r = 2.6
    for yy in range(H):
        for xx in range(W):
            if (xx - cx) ** 2 + ((yy - cy) * 1.3) ** 2 <= r * r:
                kind[yy, xx] = Cell.WATER

    road_y = int(rng.integers(H // 2, H * 3 // 4))
    road_x = int(rng.integers(W // 2, W * 2 // 3))
    kind[road_y, :] = Cell.ROAD
    kind[:, road_x] = Cell.ROAD

    base = (road_x + 1, road_y + 1)
    kind[base[1], base[0]] = Cell.BASE

    settlements: list[list[Pos]] = []
    candidates = []
    for _ in range(400):
        on_h = rng.random() < 0.5
        if on_h:
            sx = int(rng.integers(2, W - 5)); sy = road_y + (2 if rng.random() < 0.5 else -4)
        else:
            sy = int(rng.integers(2, H - 5)); sx = road_x + (2 if rng.random() < 0.5 else -4)
        if not (1 <= sx <= W - 4 and 1 <= sy <= H - 4):
            continue
        block = kind[sy:sy + 3, sx:sx + 3]
        if np.any(block == Cell.WATER) or np.any(block == Cell.ROAD) or np.any(block == Cell.BASE):
            continue
        if abs(sx + 1 - base[0]) + abs(sy + 1 - base[1]) < 8:
            continue
        if any(abs(sx - px) + abs(sy - py) < 10 for px, py in candidates):
            continue
        candidates.append((sx, sy))
        if len(candidates) >= n_settlements:
            break
    for sx, sy in candidates:
        cells = [(sx + i, sy + j) for j in range(3) for i in range(3)]
        for (px, py) in cells:
            kind[py, px] = Cell.SETTLEMENT
        settlements.append(cells)

    shore = []
    for yy in range(H):
        for xx in range(W):
            if kind[yy, xx] in (Cell.FOREST, Cell.MEADOW, Cell.ROAD):
                nb = kind[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2]
                if np.any(nb == Cell.WATER):
                    shore.append((xx, yy))
    water_points: list[Pos] = []
    if shore:
        lake_shore = [p for p in shore if p[0] > W // 2]
        first = lake_shore[int(rng.integers(len(lake_shore)))] if lake_shore else shore[0]
        water_points.append(first)
        while len(water_points) < n_water_points:
            best = max(shore, key=lambda p: min(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in water_points))
            water_points.append(best)
    for (px, py) in water_points:
        kind[py, px] = Cell.ROAD

    props = np.array([CELL_PROPS[Cell(k)] for k in range(len(Cell))])
    fuel0 = props[kind, 0]
    combust = props[kind, 1]
    move_cost = props[kind, 2]
    value = props[kind, 3]
    _ = river_x
    return Terrain(W, H, kind, fuel0, combust, move_cost, value, base, water_points, settlements)
