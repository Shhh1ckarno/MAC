import itertools

import numpy as np
import pytest

from wildfire_mas import MECHANISMS, SimConfig, Simulation
from wildfire_mas.agents import astar
from wildfire_mas.comm import Channel, Message, Perf
from wildfire_mas.coordination import hungarian
from wildfire_mas.env import Phase


@pytest.mark.parametrize("n,m", [(1, 1), (3, 3), (3, 5), (5, 3), (6, 6)])
def test_hungarian_matches_bruteforce(n, m):
    rng = np.random.default_rng(n * 10 + m)
    for _ in range(20):
        cost = rng.integers(0, 50, (n, m)).astype(float).tolist()
        assign = hungarian(cost)
        k = min(n, m)
        if n <= m:
            got = sum(cost[i][assign[i]] for i in range(n))
            best = min(sum(cost[i][p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
        else:
            got = sum(cost[i][assign[i]] for i in range(n) if assign[i] >= 0)
            best = min(sum(cost[p[j]][j] for j in range(m)) for p in itertools.permutations(range(n), m))
            assert sum(1 for a in assign if a >= 0) == k
        assert got == pytest.approx(best)


def test_hungarian_matches_scipy():
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(7)
    for _ in range(30):
        n, m = rng.integers(1, 9), rng.integers(1, 9)
        cost = rng.random((n, m))
        assign = hungarian(cost.tolist())
        r, c = scipy.linear_sum_assignment(cost)
        got = sum(cost[i, j] for i, j in enumerate(assign) if j >= 0)
        assert got == pytest.approx(cost[r, c].sum())


def test_environment_is_deterministic():
    a = Simulation(SimConfig(seed=3, horizon=80, agents_enabled=False)).run()
    b = Simulation(SimConfig(seed=3, horizon=80, agents_enabled=False)).run()
    assert a["J_burn"] == b["J_burn"] and a["burnt_cells"] == b["burnt_cells"]


def test_full_run_is_deterministic():
    a = Simulation(SimConfig(seed=2, horizon=80)).run()
    b = Simulation(SimConfig(seed=2, horizon=80)).run()
    assert a["J_burn"] == b["J_burn"] and a["M_delivered"] == b["M_delivered"]


def test_fire_spreads_and_burns_out():
    sim = Simulation(SimConfig(seed=1, horizon=120, agents_enabled=False, scenario="calm_single"))
    first = int((sim.env.fire.phase == Phase.BURNING).sum())
    sim.run()
    assert first == 1
    assert (sim.env.fire.phase == Phase.BURNT).sum() > 1


def test_bus_latency_loss_and_channels():
    sim = Simulation(SimConfig(seed=1, horizon=10))
    bus = sim.bus
    m = bus.send(Message(Perf.INFORM, "d1", ["h"], {"type": "Status"}), t=5)
    assert m.sig
    assert "h" not in bus.deliver(5)
    assert [x.id for x in bus.deliver(6)["h"]] == [m.id]
    with pytest.raises(PermissionError):
        bus.send(Message(Perf.INFORM, "d1", ["f1"], {"type": "x"}, channel=Channel.CMD), t=5)
    with pytest.raises(PermissionError):
        bus.send(Message(Perf.REQUEST, "d1", ["w1"], {"type": "x"}, channel=Channel.RES), t=5)
    sim.env.bodies["f1"].pos = (0, 0); sim.env.bodies["d1"].pos = (sim.cfg.width - 1, sim.cfg.height - 1)
    before = bus.stats["dropped_range"]
    bus.send(Message(Perf.INFORM, "d1", ["f1"], {"type": "x"}, channel=Channel.RADIO), t=6)
    assert bus.stats["dropped_range"] == before + 1


def test_bus_loss_probability():
    sim = Simulation(SimConfig(seed=1, horizon=10, p_loss_cmd=0.5))
    for i in range(400):
        sim.bus.send(Message(Perf.INFORM, "f1", ["h"], {"type": "Status"}), t=1)
    got = len(sim.bus.deliver(2).get("h", []))
    assert 140 < got < 260


def test_astar_avoids_water_and_burning():
    sim = Simulation(SimConfig(seed=1, horizon=1, agents_enabled=False))
    ter = sim.env.terrain
    blocked = np.zeros((ter.height, ter.width), dtype=bool)
    blocked[:, 5] = True
    blocked[ter.height // 2, 5] = False
    path = astar(ter, (2, ter.height // 2), {(8, ter.height // 2)}, blocked)
    assert path is not None
    for x, y in path:
        assert np.isfinite(ter.move_cost[y, x]) and not blocked[y, x]


def test_least_privilege():
    sim = Simulation(SimConfig(seed=1, horizon=5))
    with pytest.raises(PermissionError):
        sim.agents["d1"].api.act(("extinguish", (1, 1)))
    with pytest.raises(PermissionError):
        sim.agents["h"].api.observe()
    with pytest.raises(PermissionError):
        sim.agents["f1"].api.crews_at_post()


@pytest.mark.parametrize("mech", MECHANISMS)
def test_all_mechanisms_run_and_help(mech):
    base = Simulation(SimConfig(seed=1, agents_enabled=False)).run()
    r = Simulation(SimConfig(seed=1, mechanism=mech)).run()
    assert r["T"] == 300
    assert r["J_burn"] < base["J_burn"]
    assert r["tasks_created"] > 0 and r["M_delivered"] > 0


def test_water_refill_cycle():
    sim = Simulation(SimConfig(seed=2))
    r = sim.run()
    assert r["water_used"] > sim.cfg.crew_tank
    assert sum(a.served for a in sim.agents.values() if a.kind == "water") > 0


def test_drone_contract_net_works_without_hq_allocation():
    sim = Simulation(SimConfig(seed=1, mechanism="cnp_drone"))
    r = sim.run()
    awards = sum(sim.agents[d].cm.stats["awards"] for d in ("d1", "d2", "d3"))
    assert awards > 0
    assert r["by_channel"].get("radio", 0) > 0
    assert r["requests"] == 0
