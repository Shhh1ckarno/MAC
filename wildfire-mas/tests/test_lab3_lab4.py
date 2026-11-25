from dataclasses import replace
import json

import pytest

from wildfire_mas import SimConfig, Simulation
from wildfire_mas.agents.water import Reservation
from wildfire_mas.comm import Channel, Message, Perf
from wildfire_mas.lab_experiments import _lab4_job


def test_refill_priority_and_aging():
    cfg = SimConfig(n_water=1, refill_policy="priority")
    sim = Simulation(cfg)
    post = sim.agents["w1"]
    post.queue = [Reservation("f1", 0, 0, 0, 1, "first", 0),
                  Reservation("f2", 1, 1, 1, 2, "second", 10)]
    post.arrived_at = {"f1": 0, "f2": 9}
    assert [r.crew for r in post.order_queue(9)] == ["f2", "f1"]
    assert [r.crew for r in post.order_queue(10)] == ["f1", "f2"]

    for crew in ("f1", "f2"):
        sim.env.bodies[crew].pos = post.api.pos
    post.step(10, [])
    assert list(post.serving) == ["f1"]
    assert post.peak_serving == 1
    post.step(10 + cfg.refill_time, [])
    assert list(post.serving) == ["f2"]


def test_timeout_and_delayed_reply_do_not_keep_or_clear_wrong_reservation():
    sim = Simulation(SimConfig(reliability_guards=True))
    crew = sim.agents["f1"]
    crew.slot = {"wp": "w1", "cid": "stale", "deadline": 0}
    crew._do_refill(0)
    assert crew.slot["wp"] == "w2"
    current = crew.slot.copy()
    crew._handle_refill(1, Message(Perf.INFORM, "w2", ["f1"], {"type": "RefillDone"},
                                   channel=Channel.RES, cid="stale"))
    assert crew.slot == current



def test_confirmed_outage_keeps_the_post_out_of_route_selection():
    sim = Simulation(SimConfig(reliability_guards=True))
    crew = sim.agents["f1"]
    crew._avoid_post(5, "w1", "out_of_service")
    sim.env.bodies["f1"].pos = sim.agents["w1"].api.pos
    assert crew._choose_wp(150) == "w2"


def test_report_validation_checks_before_detection_and_agent_processing():
    cfg = SimConfig(reliability_guards=True)
    sim = Simulation(cfg)
    reports = [Message(Perf.INFORM, "d1", ["h"], {"type": "FireReport", "area": [0, 0, 2],
                       "cells": [[0, 1, 0.5]], "t_obs": 0}, id=1)]
    valid = reports[0]
    reports += [replace(valid, id=i, content={**valid.content, **change}) for i, change in enumerate(
        ({"cells": [[cfg.width, 0, 0.5]]}, {"t_obs": 1}, {"area": []}), start=2)]
    inbox = {"h": reports}
    sim._validate_reports(0, inbox)
    assert inbox == {"h": [valid]}
    assert [event["reason"] for event in sim.tick_events] == ["report_cell", "report_time", "report_area"]


@pytest.mark.parametrize("case", ["bad_report", "water_outage", "llm_outage"])
def test_fault_replay_before_and_after_guards(case, tmp_path):
    paths = [tmp_path / f"{mode}.jsonl" for mode in ("before", "after")]
    before, after = [_lab4_job((1, case, mode, path)) for mode, path in zip(("before", "after"), paths)]
    assert before["fault_tick"] == after["fault_tick"]
    assert before["violations"] > 0
    assert after["completed"] and after["violations"] == 0
    assert after["recovery_delay"] is not None
    for path, row in zip(paths, (before, after)):
        trace = [json.loads(line) for line in path.read_text().splitlines()]
        assert trace[-1]["type"] == "outcome"
        assert (trace[-1]["status"] == "completed") == row["completed"]
        assert any(e["event"] == "fault" for record in trace for e in record.get("events", []))
