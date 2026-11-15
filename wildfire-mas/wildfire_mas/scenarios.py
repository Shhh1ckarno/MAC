from __future__ import annotations

from dataclasses import fields, replace

from .config import SimConfig

SCENARIOS: dict[str, dict] = {
    "default": {
        "title": "Базовый: 2 начальных очага + случайные возгорания",
        "cfg": {},
        "scenario": {},
    },
    "calm_single": {
        "title": "Один очаг, слабый ветер, без новых возгораний (демонстрация)",
        "cfg": {"initial_fires": 1, "wind_speed0": 2.0},
        "scenario": {"poisson": False},
    },
    "windy": {
        "title": "Сильный ветер: быстрое распространение по ветру",
        "cfg": {"wind_speed0": 9.0, "initial_fires": 2},
        "scenario": {},
    },
    "outbreak": {
        "title": "Массовые возгорания: дефицит расчётов и воды",
        "cfg": {"initial_fires": 5, "ignition_rate": 0.03},
        "scenario": {},
    },
}


def apply_scenario(cfg: SimConfig) -> tuple[SimConfig, dict]:
    if cfg.scenario not in SCENARIOS:
        raise ValueError(f"Неизвестный сценарий {cfg.scenario!r}; доступны: {list(SCENARIOS)}")
    sc = SCENARIOS[cfg.scenario]
    defaults = SimConfig()
    over = {}
    for f in fields(SimConfig):
        if f.name in sc["cfg"] and getattr(cfg, f.name) == getattr(defaults, f.name):
            over[f.name] = sc["cfg"][f.name]
    return replace(cfg, **over), dict(sc["scenario"])