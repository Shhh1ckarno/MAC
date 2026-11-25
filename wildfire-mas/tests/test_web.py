from fastapi.testclient import TestClient

from wildfire_mas.web.server import app


def test_web_endpoints_and_ws():
    c = TestClient(app)
    assert "МАС" in c.get("/").text
    opts = c.get("/api/options").json()
    assert {m["id"] for m in opts["mechanisms"]} >= {"central_greedy", "cnp"}
    with c.websocket_connect("/ws") as ws:
        ws.send_json({"cmd": "reset", "config": {"seed": 2, "mechanism": "cnp", "horizon": 30}})
        init = ws.receive_json()
        assert init["type"] == "init" and len(init["terrain"]) == init["width"] * init["height"]
        first = ws.receive_json()
        assert first["type"] == "tick" and first["t"] == 0
        for k in range(3):
            ws.send_json({"cmd": "step"})
            tk = ws.receive_json()
            assert tk["type"] == "tick" and tk["t"] == k + 1
        assert any(a["kind"] == "crew" for a in tk["agents"])


def test_lab_web_comparisons():
    client = TestClient(app)
    configuration = {"seed": 1, "horizon": 40}
    r = client.post("/api/compare", json={"lab": 2, "config": configuration, "alternative": "cnp"})
    assert r.status_code == 200
    rules = r.json()
    assert rules["seed"] == 1 and rules["horizon"] == 40
    assert len(rules["variants"]) == 2
    assert all(len(variant["series"]) == 40 for variant in rules["variants"])

    r = client.post("/api/compare", json={"lab": 3, "config": configuration})
    assert r.status_code == 200
    assert all(v["metrics"]["refill_wait_mean"] is not None for v in r.json()["variants"])

    r = client.post("/api/compare", json={"lab": 4, "config": {**configuration, "fault_case": "bad_report"}})
    assert r.status_code == 200
    before, after = r.json()["variants"]
    assert before["failure"] and before["metrics"]["J_burn"] is None
    assert before["fault"]["start"] == after["fault"]["start"] == after["fault"]["recovery"] == 14
    assert before["fault"]["violations"] == 1 and after["fault"]["violations"] == 0
    assert len(before["series"]) == 14 and len(after["series"]) == 40


def test_lab_web_live_queue_and_fault():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"cmd": "reset", "lab": 3, "config": {"refill_policy": "priority"}})
        config = ws.receive_json()["config"]
        assert config["refill_policy"] == "priority" and config["n_crews"] == 8
        assert ws.receive_json()["water"][0]["capacity"] == 1
        ws.send_json({"cmd": "reset", "lab": 4, "config": {"fault_case": "bad_report", "fault_mode": "before"}})
        assert ws.receive_json()["lab"] == 4
        assert ws.receive_json()["fault"]["enabled"]
        for _ in range(15):
            ws.send_json({"cmd": "step"})
            tick = ws.receive_json()
        assert tick["fault"]["start"] == 14
        assert any(event["event"] == "fault" for event in tick["events"])
        assert ws.receive_json()["type"] == "outcome"



def test_live_llm_reports_http_fallback_and_response(monkeypatch):
    from threading import Event
    from types import SimpleNamespace

    client = TestClient(app)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"cmd": "reset", "config": {"hq_mode": "llm"}})
        assert ws.receive_json()["config"]["hq_mode"] == "llm"
        assert ws.receive_json()["llm_status"]["configured"] is False
        for _ in range(15):
            ws.send_json({"cmd": "step"})
            tick = ws.receive_json()
        assert tick["llm_status"]["http_attempts"] == 0
        assert tick["llm_status"]["fallbacks"] == 1

    release = Event()
    endpoints = []

    def post(url, **options):
        endpoints.append((url, options["json"]["model"]))
        release.wait(4)
        reply = '{"assignments":[{"crew":"f1","task":"T1"}],"comment":""}'
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"choices": [{"message": {"content": reply}}]})

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "test-qwen")
    monkeypatch.setattr("wildfire_mas.llm.client.httpx.post", post)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"cmd": "reset", "config": {"hq_mode": "llm"}})
        ws.receive_json()
        initial = ws.receive_json()["llm_status"]
        assert initial["configured"] is True and initial["http_attempts"] == 0
        for _ in range(14):
            ws.send_json({"cmd": "step"})
            assert ws.receive_json()["type"] == "tick"
        ws.send_json({"cmd": "step"})
        pending = ws.receive_json()
        release.set()
        assert pending == {"type": "llm_request", "t": 14, "model": "test-qwen", "http_attempts": 1}
        tick = ws.receive_json()
        assert tick["llm"][0]["valid"] == [["f1", "T1"]]
        assert tick["llm_status"]["http_attempts"] == 1
        assert tick["llm_status"]["fallbacks"] == 0
    assert endpoints == [("http://localhost:1234/v1/chat/completions", "test-qwen")]
