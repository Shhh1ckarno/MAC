import json

from wildfire_mas import SimConfig, Simulation
from wildfire_mas.llm import LLMCoordinator, LLMError, LLMReply, extract_json, validate_plan


def test_extract_json_strips_thinking_and_fences():
    text = "<think>рассуждаю...</think>\n```json\n{\"assignments\": [], \"comment\": \"ok\"}\n```"
    assert extract_json(text) == {"assignments": [], "comment": "ok"}


def test_validator_rejects_bad_assignments():
    plan = {"assignments": [{"crew": "f1", "task": "T1"}, {"crew": "f1", "task": "T2"},
                            {"crew": "f9", "task": "T1"}, {"crew": "f2", "task": "T7"},
                            {"crew": "f3", "task": "T1"}]}
    ok, viol = validate_plan(plan, free=["f1", "f2", "f3"], open_slots={"T1": 1, "T2": 1})
    assert ok == [("f1", "T1")]
    assert len(viol) == 4


class FakeClient:
    configured = True

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def chat(self, system, user, schema=None, max_tokens=1200):
        self.calls += 1
        if self.fail:
            raise LLMError("нет связи")
        data = json.loads(user.split("ОБСТАНОВКА (JSON):\n")[1].split("\n\nДОНЕСЕНИЯ")[0])
        task = data["open_tasks"][0]["id"]
        asg = [{"crew": c["id"], "task": task} for c in data["free_crews"]] + [{"crew": "f99", "task": task}]
        return LLMReply(json.dumps({"assignments": asg, "comment": "тест"}), 0.01, {})


def test_llm_coordinator_assigns_valid_part():
    cfg = SimConfig(seed=1, horizon=60, hq_mode="llm")
    client = FakeClient()
    sim = Simulation(cfg, coordinator=LLMCoordinator(cfg, client))
    r = sim.run()
    co = sim.agents["h"].coordinator
    assert client.calls > 0 and co.stats["ok"] > 0
    assert co.stats["violations"] > 0
    assert r["requests"] > 0


def test_llm_coordinator_falls_back_to_rules():
    cfg = SimConfig(seed=1, horizon=60, hq_mode="llm")
    sim = Simulation(cfg, coordinator=LLMCoordinator(cfg, FakeClient(fail=True)))
    r = sim.run()
    co = sim.agents["h"].coordinator
    assert co.stats["errors"] > 0 and co.stats["fallbacks"] == co.stats["errors"]
    assert r["requests"] > 0
