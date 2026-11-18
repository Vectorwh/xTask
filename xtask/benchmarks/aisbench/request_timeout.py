"""Optional client timeout for one AISBench case; shared defaults are unchanged."""
import aiohttp
from ais_bench.benchmark.models.api_models.vllm_custom_api_chat import VLLMCustomAPIChat
from ais_bench.benchmark.registry import MODELS

@MODELS.register_module()
class XTaskGPQAChat(VLLMCustomAPIChat):
    """Bound each request without modifying AISBench's global timeout."""

    def __init__(self, request_timeout=3600, **kwargs):
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self.request_timeout = request_timeout
        super().__init__(**kwargs)

    async def generate(self, input_data, max_out_len, output, session=None, **args):
        if session is not None:
            return await super().generate(
                input_data, max_out_len, output,
                session=_RequestTimeoutSession(session, self.request_timeout), **args
            )
        async with aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=self.request_timeout)
        ) as request_session:
            return await super().generate(
                input_data, max_out_len, output, session=request_session, **args
            )


class _RequestTimeoutSession:
    """Retain AISBench's shared connection pool and set a per-request timeout."""

    def __init__(self, session, timeout):
        self.session = session
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    def post(self, *args, **kwargs):
        kwargs["timeout"] = self.timeout
        return self.session.post(*args, **kwargs)
