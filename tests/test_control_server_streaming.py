import json

import httpx
import pytest

from llm.control_server import create_app
from llm.testing import RecordingPipeline


@pytest.mark.asyncio
async def test_mutation_endpoint_can_stream_progress_and_final_response():
    app = create_app(RecordingPipeline())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/mutations/apply",
            headers={"accept": "application/x-ndjson"},
            json={
                "type": "apply_mutation",
                "base_catalog_version": "v0",
                "operations": [],
            },
        )

    response.raise_for_status()
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["event"] == "progress"
    assert events[0]["phase"] == "adapter"
    assert events[-1]["event"] == "mutation_result"
    assert events[-1]["response"]["status"] == "applied"
