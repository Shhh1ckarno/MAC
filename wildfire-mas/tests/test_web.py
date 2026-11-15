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
