from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx


@dataclass
class LLMReply:
    text: str
    latency: float
    raw: dict


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str = "", model: str = "", timeout: float = 60.0):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.api_key = os.environ.get("LLM_API_KEY", "lm-studio")
        self.timeout = timeout
        self.use_schema = True
        self.http_attempts = 0
        self.reasoning_effort = os.environ.get("LLM_REASONING_EFFORT", "none")

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def chat(self, system: str, user: str, schema: dict | None = None, max_tokens: int = 1200) -> LLMReply:
        if not self.configured:
            raise LLMError("LLM не настроена: задайте LLM_BASE_URL (и LLM_MODEL)")
        body = {
            "model": self.model or "local-model",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if schema is not None and self.use_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "plan", "strict": True, "schema": schema}}
        t0 = time.perf_counter()
        try:
            self.http_attempts += 1
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                           headers={"Authorization": f"Bearer {self.api_key}"})
            if r.status_code == 400 and "response_format" in body:
                self.use_schema = False
                body.pop("response_format")
                self.http_attempts += 1
                r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                               headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"Ошибка обращения к LLM: {e}") from e
        data = r.json()
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"Неожиданный формат ответа: {data}") from e
        return LLMReply(text=text, latency=time.perf_counter() - t0, raw=data)