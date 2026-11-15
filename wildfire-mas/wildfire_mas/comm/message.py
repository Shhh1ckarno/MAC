from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum


class Perf(str, Enum):
    INFORM = "INFORM"
    REQUEST = "REQUEST"
    AGREE = "AGREE"
    REFUSE = "REFUSE"
    FAILURE = "FAILURE"
    CFP = "CFP"
    PROPOSE = "PROPOSE"
    ACCEPT_PROPOSAL = "ACCEPT_PROPOSAL"
    REJECT_PROPOSAL = "REJECT_PROPOSAL"
    CANCEL = "CANCEL"
    QUERY_REF = "QUERY_REF"


class Channel(str, Enum):
    CMD = "cmd"
    RADIO = "radio"
    RES = "res"


RADIO_BROADCAST = "*"


@dataclass
class Message:
    perf: Perf
    sender: str
    receivers: list[str]
    content: dict
    channel: Channel = Channel.CMD
    cid: str | None = None
    reply: int | None = None
    hops: int = 0
    id: int = 0
    t_send: int = 0
    sig: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def ctype(self) -> str:
        return str(self.content.get("type", ""))

    def size_bytes(self) -> int:
        body = json.dumps(self.content, ensure_ascii=False, separators=(",", ":"), default=str)
        return 64 + len(body.encode("utf-8"))

    def compute_sig(self, secret: str) -> str:
        body = json.dumps(self.content, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(f"{secret}|{self.sender}|{self.perf.value}|{body}".encode()).hexdigest()[:16]

    def brief(self) -> dict:
        return {"id": self.id, "t": self.t_send, "perf": self.perf.value, "from": self.sender,
                "to": self.receivers, "type": self.ctype, "ch": self.channel.value, "cid": self.cid}
