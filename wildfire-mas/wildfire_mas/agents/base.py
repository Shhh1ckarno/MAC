from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..comm import Channel, Message, Perf
from ..env import Body, Observation, Pos, Terrain, cheb

if TYPE_CHECKING:
    from ..sim import Simulation


ALLOWED_ACTIONS = {
    "hq": set(),
    "drone": {"fly", "charge", "wait"},
    "crew": {"move", "extinguish", "wait"},
    "water": {"serve"},
}


class AgentAPI:

    def __init__(self, sim: "Simulation", body: Body):
        self._sim = sim
        self._body = body
        self.pending_action: tuple | None = None

    @property
    def pos(self) -> Pos:
        return self._body.pos

    @property
    def water(self) -> int:
        return self._body.water

    @property
    def battery(self) -> int:
        return self._body.battery

    @property
    def broken(self) -> bool:
        return self._body.broken

    def observe(self) -> Observation:
        kind = self._body.kind
        cfg = self._sim.cfg
        if kind == "drone":
            return self._sim.env.observe(self._body, cfg.drone_radius, self._sim.agent_rngs[self._body.id],
                                         cfg.drone_p_fn, cfg.drone_p_fp, cfg.drone_noise)
        if kind == "crew":
            return self._sim.env.observe(self._body, cfg.crew_radius)
        raise PermissionError(f"Агент {self._body.id} ({kind}) не имеет сенсоров обзора местности")

    def weather(self) -> tuple[float, float]:
        return self._sim.env.wind()

    def crews_at_post(self) -> list[str]:
        if self._body.kind != "water":
            raise PermissionError("Только точка заправки видит свой пост")
        return [b.id for b in self._sim.env.bodies.values()
                if b.kind == "crew" and b.pos == self._body.pos and not b.broken]

    def act(self, action: tuple) -> None:
        if action[0] not in ALLOWED_ACTIONS[self._body.kind]:
            raise PermissionError(f"Действие {action[0]} недоступно агенту {self._body.id}")
        self.pending_action = action

    def send(self, perf: Perf, receivers: list[str] | str, content: dict, channel: Channel = Channel.CMD,
             cid: str | None = None, reply: int | None = None, hops: int = 0) -> Message:
        if isinstance(receivers, str):
            receivers = [receivers]
        msg = Message(perf=perf, sender=self._body.id, receivers=list(receivers), content=content,
                      channel=channel, cid=cid, reply=reply, hops=hops)
        return self._sim.bus.send(msg, self._sim.env.t)


@dataclass
class Knowledge:
    terrain: Terrain
    hq_id: str
    drones: list[str]
    crews: list[str]
    water_points: dict[str, Pos]
    base: Pos
    cfg: Any


class FireBelief:

    def __init__(self, terrain: Terrain):
        H, W = terrain.height, terrain.width
        self.burning = np.zeros((H, W), dtype=bool)
        self.I = np.zeros((H, W))
        self.t_obs = np.full((H, W), -1, dtype=np.int32)
        self.source: dict[Pos, str] = {}

    def update_cell(self, p: Pos, burning: bool, I: float, t: int, src: str = "") -> bool:
        x, y = p
        if t < self.t_obs[y, x]:
            return False
        changed = self.burning[y, x] != burning or abs(self.I[y, x] - I) > 0.25
        self.burning[y, x] = burning
        self.I[y, x] = I if burning else 0.0
        self.t_obs[y, x] = t
        if src:
            self.source[p] = src
        return changed

    def apply_area_report(self, area: tuple[int, int, int], cells: list, t: int, src: str = "") -> bool:
        cx, cy, r = area
        listed = {}
        for c in cells:
            listed[(int(c[0]), int(c[1]))] = float(c[2])
        changed = False
        H, W = self.burning.shape
        for y in range(max(0, cy - r), min(H, cy + r + 1)):
            for x in range(max(0, cx - r), min(W, cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 > r * r + r:
                    continue
                if (x, y) in listed:
                    changed |= self.update_cell((x, y), True, listed[(x, y)], t, src)
                elif self.burning[y, x]:
                    changed |= self.update_cell((x, y), False, 0.0, t, src)
        for p, I in listed.items():
            if not (0 <= p[0] < W and 0 <= p[1] < H):
                continue
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 > r * r + r:
                changed |= self.update_cell(p, True, I, t, src)
        return changed

    def burning_cells(self) -> list[Pos]:
        return [(int(x), int(y)) for y, x in np.argwhere(self.burning)]

    def clusters(self) -> list[set[Pos]]:
        seen = np.zeros_like(self.burning)
        H, W = self.burning.shape
        out = []
        for y, x in np.argwhere(self.burning):
            if seen[y, x]:
                continue
            comp, stack = set(), [(int(x), int(y))]
            seen[y, x] = True
            while stack:
                cx, cy = stack.pop()
                comp.add((cx, cy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < W and 0 <= ny < H and self.burning[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            out.append(comp)
        return out


def astar(terrain: Terrain, start: Pos, goals: set[Pos], blocked: np.ndarray | None = None,
          max_expand: int = 5000) -> list[Pos] | None:
    if not goals:
        return None
    if start in goals:
        return []
    gl = list(goals)

    def h(p: Pos) -> float:
        return 0.5 * min(cheb(p, g) for g in gl) if len(gl) <= 16 else 0.0

    openq = [(h(start), 0.0, start)]
    came: dict[Pos, Pos] = {}
    g = {start: 0.0}
    expanded = 0
    W, H = terrain.width, terrain.height
    while openq and expanded < max_expand:
        _, gc, cur = heapq.heappop(openq)
        if gc > g.get(cur, np.inf):
            continue
        expanded += 1
        if cur in goals:
            path = [cur]
            while path[-1] in came and came[path[-1]] != start:
                path.append(came[path[-1]])
            return list(reversed(path))
        cx, cy = cur
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                c = terrain.move_cost[ny, nx]
                if not np.isfinite(c) or (blocked is not None and blocked[ny, nx]):
                    continue
                ng = gc + float(c)
                if ng < g.get((nx, ny), np.inf):
                    g[(nx, ny)] = ng
                    came[(nx, ny)] = cur
                    heapq.heappush(openq, (ng + h((nx, ny)), ng, (nx, ny)))
    return None


def path_cost(terrain: Terrain, path: list[Pos]) -> float:
    return float(sum(terrain.move_cost[y, x] for x, y in path))


class Agent:
    kind = "agent"
    roles: tuple[str, ...] = ()

    def __init__(self, aid: str, api: AgentAPI, knowledge: Knowledge):
        self.id = aid
        self.api = api
        self.K = knowledge
        self.cfg = knowledge.cfg
        self.state = "IDLE"
        self.log: list[str] = []

    def note(self, t: int, text: str) -> None:
        self.log.append(f"t={t} {self.id}: {text}")
        if len(self.log) > 50:
            self.log = self.log[-50:]

    def step(self, t: int, inbox: list[Message]) -> None:
        raise NotImplementedError

    def public_state(self) -> dict:
        return {"id": self.id, "kind": self.kind, "state": self.state}
