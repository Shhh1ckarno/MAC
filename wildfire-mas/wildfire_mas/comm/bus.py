from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import numpy as np

from ..config import SimConfig
from ..env import Body, cheb
from .message import RADIO_BROADCAST, Channel, Message


class MessageBus:
    def __init__(self, cfg: SimConfig, bodies: dict[str, Body], rng: np.random.Generator,
                 secrets: dict[str, str] | None = None):
        self.cfg = cfg
        self.bodies = bodies
        self.rng = rng
        self.secrets = secrets if secrets is not None else {}
        self._next_id = 1
        self._pending: list[tuple[int, str, Message]] = []
        self.sent_log: list[Message] = []
        self.stats = {
            "sent": 0, "deliveries": 0, "lost": 0, "dropped_range": 0, "bytes": 0,
            "by_perf": Counter(), "by_channel": Counter(), "by_type": Counter(), "by_sender_kind": Counter(),
        }
        self.tap: Callable[[Message], None] | None = None

    def _kind(self, aid: str) -> str:
        b = self.bodies.get(aid)
        return b.kind if b else "?"

    def _check_channel(self, msg: Message, rcv: str) -> bool:
        ks, kr = self._kind(msg.sender), self._kind(rcv)
        if msg.channel == Channel.CMD:
            return "hq" in (ks, kr)
        if msg.channel == Channel.RES:
            return {ks, kr} == {"crew", "water"}
        if msg.channel == Channel.RADIO:
            return ks in ("drone", "crew") and kr in ("drone", "crew")
        return False

    def _latency(self, ch: Channel) -> int:
        return self.cfg.radio_latency if ch == Channel.RADIO else max(1, self.cfg.cmd_latency)

    def _p_loss(self, ch: Channel) -> float:
        if ch == Channel.RADIO:
            return self.cfg.p_loss_radio
        if ch == Channel.CMD:
            return self.cfg.p_loss_cmd
        return 0.0

    def send(self, msg: Message, t: int) -> Message:
        msg.id = self._next_id; self._next_id += 1
        msg.t_send = t
        if not msg.sig and msg.sender in self.secrets:
            msg.sig = msg.compute_sig(self.secrets[msg.sender])
        receivers = msg.receivers
        if msg.channel == Channel.RADIO:
            sp = self.bodies[msg.sender].pos
            cand = [b.id for b in self.bodies.values() if b.kind in ("drone", "crew") and b.id != msg.sender]
            if receivers == [RADIO_BROADCAST]:
                receivers = cand
            in_range = [r for r in receivers if r in self.bodies and cheb(self.bodies[r].pos, sp) <= self.cfg.radio_range]
            self.stats["dropped_range"] += len(receivers) - len(in_range)
            receivers = in_range
        st = self.stats
        st["sent"] += 1; st["bytes"] += msg.size_bytes()
        st["by_perf"][msg.perf.value] += 1; st["by_channel"][msg.channel.value] += 1
        st["by_type"][msg.ctype] += 1; st["by_sender_kind"][self._kind(msg.sender)] += 1
        self.sent_log.append(msg)
        if self.tap:
            self.tap(msg)
        for r in receivers:
            if not self._check_channel(msg, r):
                raise PermissionError(f"Канал {msg.channel.value} не соединяет {msg.sender} и {r}")
            if self.rng.random() < self._p_loss(msg.channel):
                st["lost"] += 1
                continue
            self._pending.append((t + self._latency(msg.channel), r, msg))
        return msg

    def deliver(self, t: int) -> dict[str, list[Message]]:
        inbox: dict[str, list[Message]] = defaultdict(list)
        keep = []
        for item in self._pending:
            td, r, m = item
            if td <= t:
                inbox[r].append(m)
                self.stats["deliveries"] += 1
            else:
                keep.append(item)
        self._pending = keep
        for msgs in inbox.values():
            msgs.sort(key=lambda m: m.id)
        return inbox

    def new_tick(self) -> None:
        self.sent_log = []
