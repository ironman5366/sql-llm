from __future__ import annotations

from typing import Any

import httpx


class Sampler:
    def __init__(self, endpoint: str, *, timeout_seconds: float = 300.0):
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout_seconds

    async def sample(self, payload: dict[str, Any]) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._endpoint}/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        return _extract_text(data)


def _extract_text(data: Any) -> str:
    if isinstance(data, dict):
        text = data.get("text")
        if isinstance(text, str):
            return text
        if isinstance(text, list) and text and isinstance(text[0], str):
            return text[0]
        generated_text = data.get("generated_text")
        if isinstance(generated_text, str):
            return generated_text
    if isinstance(data, list) and data:
        return _extract_text(data[0])
    raise ValueError(f"SGLang response did not contain generated text: {data!r}")
