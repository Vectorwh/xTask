"""Integration checks against the AISBench installed in the benchmark image."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ais_bench")
from aiohttp import ClientSession, ClientTimeout, web
from mmengine import Config
from ais_bench.benchmark.registry import TEXT_POSTPROCESSORS
from xtask.benchmarks.aisbench.request_timeout import XTaskGPQAChat


@pytest.mark.parametrize("shared,timeout,success", [(True, 1, True), (False, 1, True), (True, 0.005, False)])
def test_client_timeout_reaches_actual_http_request(shared, timeout, success, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    async def exercise():
        async def handler(request):
            await asyncio.sleep(0.05)
            return web.json_response({"choices": [{"message": {"content": "Answer: B"}}]})

        async def noop(*args):
            pass

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            model = XTaskGPQAChat(model="test", host_ip="127.0.0.1", host_port=port,
                                 generation_kwargs={}, request_timeout=timeout, retry=1)
            output = SimpleNamespace(content="", reasoning_content="", success=False,
                                     error_info="", record_time_point=noop, clear_time_points=noop,
                                     update_extra_details_data_from_text_response=lambda _: None)
            async with ClientSession(timeout=ClientTimeout(total=0.005)) as session:
                await model.generate("Question", 8, output, session=session if shared else None)
                assert not session.closed
            assert output.success is success, output.error_info
            if success:
                assert output.content == "Answer: B"
            else:
                assert "TimeoutError" in output.error_info
        finally:
            await runner.cleanup()

    asyncio.run(exercise())
