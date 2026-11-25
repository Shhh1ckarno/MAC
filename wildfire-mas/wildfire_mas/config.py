from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

MECHANISMS = ("central_greedy", "central_hungarian", "cnp", "auction_first", "auction_vickrey", "cnp_drone")
HQ_MODES = ("rules", "llm")


@dataclass
class SimConfig:
    seed: int = 1
    scenario: str = "default"
    width: int = 40
    height: int = 30
    horizon: int = 300
    agents_enabled: bool = True

    n_drones: int = 3
    n_crews: int = 5
    n_water: int = 2

    mechanism: str = "central_greedy"
    hq_mode: str = "rules"

    p0: float = 0.045
    growth: float = 0.15
    burn_rate: float = 0.035
    ignite_intensity: float = 0.3
    wind_c1: float = 0.045
    wind_c2: float = 0.131
    wind_speed0: float = 4.0
    wind_dir0: float | None = None
    wind_speed_sigma: float = 0.3
    wind_dir_sigma: float = 0.05
    wind_revert: float = 0.05
    wind_max: float = 12.0
    moisture_decay: float = 0.02
    water_effect: float = 0.6
    ignition_rate: float = 0.015
    initial_fires: int = 2

    drone_speed: int = 3
    drone_radius: int = 4
    drone_battery: int = 120
    drone_recharge: int = 20
    drone_reserve: int = 6
    drone_p_fn: float = 0.05
    drone_p_fp: float = 0.0
    drone_noise: float = 0.1

    crew_radius: int = 2
    crew_tank: int = 30
    crew_refill_threshold: int = 1
    cells_per_crew: int = 8

    refill_policy: str = "fifo"
    reliability_guards: bool = False
    water_capacity: int = 1
    refill_time: int = 5
    noshow_timeout: int = 15

    radio_range: int = 10
    cmd_latency: int = 1
    radio_latency: int = 1
    p_loss_cmd: float = 0.0
    p_loss_radio: float = 0.0
    hb_period: int = 5
    hb_timeout: int = 15

    bid_deadline: int = 2

    llm_base_url: str = ""
    llm_model: str = ""
    llm_timeout: float = 60.0
    llm_min_interval: int = 5
    llm_retry_ticks: int = 15

    extra: dict = field(default_factory=dict)

    def validate(self) -> "SimConfig":
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"Неизвестный механизм {self.mechanism!r}; допустимо: {MECHANISMS}")
        if self.hq_mode not in HQ_MODES:
            raise ValueError(f"Неизвестный режим штаба {self.hq_mode!r}; допустимо: {HQ_MODES}")
        if self.refill_policy not in ("fifo", "priority"):
            raise ValueError("Режим очереди должен быть fifo или priority")
        if self.water_capacity < 1 or self.refill_time < 1:
            raise ValueError("Вместимость и время заправки должны быть положительными")
        if self.llm_retry_ticks < 1:
            raise ValueError("Пауза между повторными запросами LLM должна быть положительной")
        if self.width < 10 or self.height < 10:
            raise ValueError("Карта слишком мала")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names}).validate()
