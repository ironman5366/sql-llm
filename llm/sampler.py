from __future__ import annotations

from typing import Any

import httpx


class Sampler:
    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 300.0,
        max_concurrency: int = 64,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            limits=httpx.Limits(
                max_connections=max_concurrency,
                max_keepalive_connections=max_concurrency,
            ),
        )

    async def sample(self, payload: dict[str, Any]) -> str:
        response = await self._client.post(f"{self._endpoint}/generate", json=payload)
        response.raise_for_status()
        return _extract_text(response.json())


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
