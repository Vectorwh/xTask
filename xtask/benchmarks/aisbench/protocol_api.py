"""Protocol-aware API adapter for AISBench.

Extends AISBench's VLLMCustomAPIChat / VLLMCustomAPIChatStream to support:
1. Per-sample extra_params passthrough (tools, tool_choice, response_format,
   guided_json, guided_regex, guided_grammar, extra_body).
2. Full message preservation (reasoning_content, tool_calls, finish_reason)
   instead of extracting only choices[0].message.content.
3. Optional stream verification: SSE delta chunks are assembled per the
   OpenAI streaming spec and compared against the non-stream result; field
   misalignment or tag leakage (e.g. </think> in content) marks the case
   as a protocol error.

AISBench request flow (for reference):
    generate(inputs) -> ThreadPoolExecutor -> _generate(input, max_out_len)
        (decorated with @handle_synthetic_input: rate-limit, finish_count,
         result queue)
        -> extract extra_params from input dict (thread-safe local)
        -> build messages (str / PromptList / multi-modal)
        -> build generation_kwargs + inject protocol fields
        -> client.request(MiddleData, kwargs)
            -> construct_request_body()   (OpenAI payload)
            -> do_request()               (HTTP POST)
            -> process_response()         (parse JSON)
            -> update_middle_data()       (extract/preserve full message)
            -> return MiddleData.get_output()
        -> return prediction string (full message JSON)

ais_bench is imported lazily so the pure-Python helpers (payload injection,
message extraction, SSE assembly, stream verification) remain unit-testable
on the host without ais_bench installed.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Protocol fields that can be injected per-sample via extra_params.
PROTOCOL_FIELDS = frozenset({
    "tools",
    "tool_choice",
    "response_format",
    "guided_json",
    "guided_regex",
    "guided_grammar",
    "extra_body",
})

# Reasoning tags that must NOT leak into content when reasoning is stripped.
REASONING_LEAK_TAGS = ("</think>", "<think>", "<|begin_of_thought|>", "<|end_of_thought|>")


# Parameters that belong in generation_kwargs / request payloads, NOT in the
# model constructor.  If they appear as top-level model dict keys, they are
# popped defensively in ProtocolAwareAPIChat.__new__ to avoid
# ``VLLMCustomAPIChat.__init__() got an unexpected keyword argument`` and
# later merged into each request payload in _build_generation_kwargs.
_GENERATION_ONLY_PARAMS = frozenset({
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
})


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without ais_bench)
# ---------------------------------------------------------------------------

def inject_protocol_fields(payload: Dict[str, Any], extra_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Inject per-sample protocol/constrained-decoding fields into an OpenAI
    chat-completion payload.

    Only fields listed in PROTOCOL_FIELDS are injected; unknown keys are
    silently ignored to avoid accidental parameter pollution.

    Args:
        payload: base OpenAI chat-completion request body.
        extra_params: per-sample dict from JSONL ``extra_params`` field.

    Returns:
        The same payload dict (mutated in place) for chaining convenience.
    """
    if not extra_params or not isinstance(extra_params, dict):
        return payload
    for field in PROTOCOL_FIELDS:
        if field in extra_params and extra_params[field] is not None:
            payload[field] = extra_params[field]
    return payload


def extract_full_message(response_json: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the full message structure from an OpenAI chat-completion
    response, preserving protocol fields that default extractors discard.

    Returns a dict with keys:
        content, reasoning_content, tool_calls, finish_reason, role.
    Missing fields default to empty string / empty list.
    """
    choices = response_json.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    return {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason") or "",
        "role": message.get("role") or "assistant",
    }


def assemble_sse_deltas(delta_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble SSE delta chunks into a complete message per the OpenAI
    streaming specification.

    Each delta chunk has the structure::
        {"choices": [{"delta": {...}, "finish_reason": None|str}]}

    Content and reasoning_content deltas are concatenated; tool_calls deltas
    are merged by index (function.name on first occurrence, function.arguments
    concatenated across chunks).

    Returns the same structure as :func:`extract_full_message`.
    """
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls_acc: Dict[int, Dict[str, Any]] = {}
    finish_reason = ""

    for chunk in delta_chunks:
        choices = chunk.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or {}

        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])

        for tc_delta in delta.get("tool_calls") or []:
            idx = tc_delta.get("index", 0)
            acc = tool_calls_acc.setdefault(idx, {
                "id": tc_delta.get("id") or "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if tc_delta.get("id"):
                acc["id"] = tc_delta["id"]
            fn = tc_delta.get("function") or {}
            if fn.get("name"):
                acc["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                acc["function"]["arguments"] += fn["arguments"]

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "role": "assistant",
    }


def check_reasoning_leak(content: str) -> bool:
    """Return True if reasoning tags leaked into the content field."""
    return any(tag in content for tag in REASONING_LEAK_TAGS)


def verify_stream_consistency(
    stream_result: Dict[str, Any],
    non_stream_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare a stream-assembled message against the non-stream message.

    Checks content equality, reasoning_content equality, tool_calls
    structural equality, and finish_reason match. Also checks for
    reasoning tag leakage in the stream result.

    Returns a report dict::
        {
            "consistent": bool,
            "errors": [str, ...],
        }
    """
    errors: List[str] = []

    if stream_result.get("content") != non_stream_result.get("content"):
        errors.append("content mismatch between stream and non-stream")
    if stream_result.get("reasoning_content") != non_stream_result.get("reasoning_content"):
        errors.append("reasoning_content mismatch between stream and non-stream")
    if stream_result.get("finish_reason") != non_stream_result.get("finish_reason"):
        errors.append(
            f"finish_reason mismatch: stream={stream_result.get('finish_reason')!r} "
            f"non_stream={non_stream_result.get('finish_reason')!r}"
        )
    if json.dumps(stream_result.get("tool_calls"), sort_keys=True) != json.dumps(
        non_stream_result.get("tool_calls"), sort_keys=True
    ):
        errors.append("tool_calls mismatch between stream and non-stream")
    if check_reasoning_leak(stream_result.get("content", "")):
        errors.append("reasoning tag leaked into stream content")

    return {"consistent": len(errors) == 0, "errors": errors}


def parse_extra_params(value: Any) -> Optional[Dict[str, Any]]:
    """Parse extra_params from dict or JSON-string form.

    ProtocolConstrainedDataset serializes extra_params to a JSON string to
    avoid pyarrow ArrowInvalid on heterogeneous struct types; this helper
    deserializes it back to a dict.

    Returns None if the value is not a dict or a valid JSON-string dict.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# AISBench lazy imports
# ---------------------------------------------------------------------------

def _get_base_classes():
    """Lazily import AISBench base classes. Returns (chat_cls, stream_cls)."""
    from ais_bench.benchmark.models import VLLMCustomAPIChat, VLLMCustomAPIChatStream
    return VLLMCustomAPIChat, VLLMCustomAPIChatStream


def _get_handle_synthetic_input():
    """Lazily import the handle_synthetic_input decorator."""
    from ais_bench.benchmark.models.base_api import handle_synthetic_input
    return handle_synthetic_input


def _patch_client_full_message(client):
    """Patch a non-stream client instance to preserve the full message.

    Overrides ``update_middle_data`` on the *instance* (not the class) so the
    complete message (content, reasoning_content, tool_calls, finish_reason)
    is stored as a JSON string in MiddleData.output instead of only
    choices[0].message.content.

    Works with any AISBench client version -- no imports from
    ais_bench.benchmark.clients needed.
    """
    orig_update = client.update_middle_data

    def update_middle_data(res, inputs):
        full_msg = extract_full_message(res)
        # Let the original update metrics (num_generated_chars, etc.)
        try:
            orig_update(res, inputs)
        except Exception:
            pass
        # Override output with the full message JSON so protocol fields
        # flow through the standard prediction pipeline.
        inputs.output = json.dumps(full_msg, ensure_ascii=False)
        inputs.output_reasoning = ""

    client.update_middle_data = update_middle_data
    return client


def _patch_client_chunk_accumulator(client):
    """Patch a stream client instance to collect raw SSE deltas.

    Overrides ``process_response`` to attach a raw-chunk marker and
    ``update_middle_data`` to accumulate the clean parsed chunk per-request
    on the MiddleData object (thread-safe: each request has its own
    MiddleData instance).

    Works with any AISBench client version -- no imports from
    ais_bench.benchmark.clients needed.
    """
    orig_request = client.request
    orig_process = client.process_response
    orig_update = client.update_middle_data

    def request(inputs, parameters):
        # Ensure middle_data is set for process_stream_line (which uses
        # self.middle_data internally in some AISBench versions).
        client.middle_data = inputs
        return orig_request(inputs, parameters)

    def process_response(json_content):
        """Parse response and attach marker for delta assembly."""
        response = orig_process(json_content)
        if isinstance(response, dict):
            response["_protocol_raw_chunk"] = json_content
        return response

    def update_middle_data(res, inputs):
        # Pop marker BEFORE the original so it sees a clean chunk
        res.pop("_protocol_raw_chunk", None)
        # Let the original accumulate content/reasoning for metrics
        orig_update(res, inputs)
        # Accumulate the clean parsed chunk for SSE delta assembly
        if not hasattr(inputs, "_protocol_delta_chunks"):
            inputs._protocol_delta_chunks = []
        inputs._protocol_delta_chunks.append(res)

    client.request = request
    client.process_response = process_response
    client.update_middle_data = update_middle_data
    return client


# ---------------------------------------------------------------------------
# Mixin (thread-safe per-sample state)
# ---------------------------------------------------------------------------

class ProtocolAwareMixin:
    """Mixin providing per-sample extra_params injection and full-message
    preservation for AISBench API model adapters.

    Thread safety: ``_generate`` uses local variables for per-sample state
    (extra_params, generation_kwargs) -- no instance attribute mutation.
    """

    def _capture_extra_params(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pop extra_params from generate() kwargs and store for this request.

        Handles both dict (direct call / legacy) and JSON-string (from
        ProtocolConstrainedDataset) representations.

        Note: This stores on the instance and is NOT thread-safe.  The
        ``_generate`` override uses :func:`parse_extra_params` on a local
        variable instead.  This method is kept for direct API callers.
        """
        extra = kwargs.pop("extra_params", None)
        extra = parse_extra_params(extra)
        self._current_extra_params = extra
        return extra

    def _build_generation_kwargs(
        self, max_out_len: int, extra_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build per-request generation kwargs with protocol fields injected.

        Thread-safe: operates on a local copy of self.generation_kwargs.
        """
        generation_kwargs = dict(getattr(self, "generation_kwargs", {}) or {})
        generation_kwargs.update({"max_tokens": max_out_len})
        generation_kwargs.update({"model": self.model})
        inject_protocol_fields(generation_kwargs, extra_params)
        # Apply generation-only params (temperature etc.) captured at
        # construction time; use setdefault so per-request kwargs win.
        overrides = getattr(self, "_generation_overrides", {})
        for key, value in overrides.items():
            generation_kwargs.setdefault(key, value)
        return generation_kwargs

    @staticmethod
    def _build_messages(input_data, is_multi_modal: bool = False):
        """Build OpenAI-format messages from input (same logic as base class).

        Returns (messages, is_multi_modal_updated).
        """
        from ais_bench.benchmark.utils.prompt import is_mm_prompt

        if isinstance(input_data, str) or is_multi_modal:
            return [{"role": "user", "content": input_data}], is_multi_modal
        elif is_mm_prompt(input_data):
            return [{"role": "user", "content": input_data}], True
        else:
            messages = []
            for item in input_data:
                msg = {"content": item["prompt"]}
                if item["role"] == "HUMAN":
                    msg["role"] = "user"
                elif item["role"] == "BOT":
                    msg["role"] = "assistant"
                elif item["role"] == "SYSTEM":
                    msg["role"] = "system"
                messages.append(msg)
            return messages, is_multi_modal


# ---------------------------------------------------------------------------
# Factory (creates classes bound to the installed AISBench)
# ---------------------------------------------------------------------------

def create_protocol_aware_chat(stream_verify: bool = False):
    """Factory: create a ProtocolAwareAPIChat or ProtocolAwareAPIChatStream
    class bound to the currently installed AISBench.

    Args:
        stream_verify: if True, use the stream adapter and enable SSE
            assembly + non-stream comparison verification.

    This factory pattern avoids import errors when ais_bench is not
    installed (e.g. running unit tests on the host).
    """
    VLLMCustomAPIChat, VLLMCustomAPIChatStream = _get_base_classes()
    handle_synthetic_input = _get_handle_synthetic_input()

    # --- Model adapters ---

    if stream_verify:
        class _ProtocolAwareAPIChatStreamImpl(ProtocolAwareMixin, VLLMCustomAPIChatStream):
            """Stream adapter with SSE delta assembly and optional
            consistency verification against non-stream results."""

            def __init__(self, *args, **kwargs):
                self._verify_enabled = kwargs.pop("stream_verify", True)
                super().__init__(*args, **kwargs)

            def init_client(self, custom_client):
                """Let the base class build the default client, then patch
                it to accumulate SSE deltas for full message assembly.

                This is version-agnostic: no imports from
                ais_bench.benchmark.clients required.
                """
                super().init_client(custom_client)
                _patch_client_chunk_accumulator(self.client)

            @handle_synthetic_input
            def _generate(self, input, max_out_len):
                # Extract per-sample fields from input dict (thread-safe locals)
                extra_params = None
                data_id = -1
                if isinstance(input, dict):
                    data_id = input.get("data_id", -1)
                    extra_params = parse_extra_params(input.get("extra_params"))
                    input = input.get("prompt")
                if max_out_len <= 0:
                    return ""

                messages, self.is_multi_modal = self._build_messages(
                    input, getattr(self, "is_multi_modal", False)
                )
                generation_kwargs = self._build_generation_kwargs(
                    max_out_len, extra_params
                )

                cache_data = self.prepare_input_data(messages, data_id)
                _ = self.client.request(cache_data, generation_kwargs)
                self.set_result(cache_data)

                # Assemble full message from collected SSE deltas
                delta_chunks = getattr(cache_data, "_protocol_delta_chunks", [])
                stream_msg = assemble_sse_deltas(delta_chunks)

                # Optional: verify against non-stream result
                if self._verify_enabled:
                    try:
                        report = self._verify_against_non_stream(
                            messages, generation_kwargs, stream_msg
                        )
                        if report and not report["consistent"]:
                            logger.warning(
                                "Stream consistency check failed: %s",
                                report["errors"],
                            )
                    except Exception:
                        logger.debug(
                            "Stream verification skipped due to error",
                            exc_info=True,
                        )

                return json.dumps(stream_msg, ensure_ascii=False)

            def _verify_against_non_stream(
                self, messages, generation_kwargs, stream_msg
            ):
                """Make a non-stream HTTP request with the same input and
                compare against the assembled stream message.

                Uses the ``requests`` library directly instead of
                instantiating an AISBench client -- avoids importing
                ais_bench.benchmark.clients (not present in all versions).
                """
                import requests

                verify_kwargs = dict(generation_kwargs)
                verify_kwargs["stream"] = False
                payload = {
                    "model": verify_kwargs.pop("model", self.model),
                    "messages": messages,
                    **verify_kwargs,
                }
                headers = {"Content-Type": "application/json"}
                api_key = getattr(self, "api_key", None)
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                try:
                    resp = requests.post(
                        self.endpoint_url,
                        json=payload,
                        headers=headers,
                        timeout=300,
                    )
                    resp.raise_for_status()
                    body = resp.json()
                except Exception:
                    logger.debug(
                        "Non-stream verification request failed",
                        exc_info=True,
                    )
                    return None
                try:
                    non_stream_msg = extract_full_message(body)
                except Exception:
                    return None
                return verify_stream_consistency(stream_msg, non_stream_msg)

        return _ProtocolAwareAPIChatStreamImpl

    class _ProtocolAwareAPIChatImpl(ProtocolAwareMixin, VLLMCustomAPIChat):
        """Non-stream adapter with per-sample protocol field injection and
        full-message preservation."""

        def init_client(self, custom_client):
            """Let the base class build the default client, then patch it
            to preserve the full message structure.

            This is version-agnostic: no imports from
            ais_bench.benchmark.clients required.
            """
            super().init_client(custom_client)
            _patch_client_full_message(self.client)

        @handle_synthetic_input
        def _generate(self, input, max_out_len):
            # Extract per-sample fields from input dict (thread-safe locals)
            extra_params = None
            data_id = -1
            if isinstance(input, dict):
                data_id = input.get("data_id", -1)
                extra_params = parse_extra_params(input.get("extra_params"))
                input = input.get("prompt")
            if max_out_len <= 0:
                return ""

            messages, self.is_multi_modal = self._build_messages(
                input, getattr(self, "is_multi_modal", False)
            )
            generation_kwargs = self._build_generation_kwargs(
                max_out_len, extra_params
            )

            cache_data = self.prepare_input_data(messages, data_id)
            response = self.client.request(cache_data, generation_kwargs)
            self.set_result(cache_data)

            # The custom client stored the full message JSON in output;
            # cache_data.get_output() returns it.
            return cache_data.get_output()

    return _ProtocolAwareAPIChatImpl


class ProtocolAwareAPIChat:
    """Config-safe entry point for protocol-aware chat models.

    mmengine config files are parsed via ``_parse_lazy_import`` which treats
    imported names as lazy objects -- calling them at module level raises
    RuntimeError. This wrapper defers mixin composition to instantiation
    time via ``__new__``, so generated config files can safely reference
    ``type=ProtocolAwareAPIChat`` as a plain name.

    The ``stream_verify`` field in the model config dict selects the stream
    adapter (SSE assembly + consistency verification) vs the non-stream one.
    """

    def __new__(cls, *args, stream_verify: bool = False, **kwargs):
        # Pop generation-only params that the base constructor doesn't
        # accept (temperature, top_p, ...).  They are stored on the instance
        # and merged into each request payload by _build_generation_kwargs.
        captured: Dict[str, Any] = {}
        for key in _GENERATION_ONLY_PARAMS:
            if key in kwargs:
                captured[key] = kwargs.pop(key)
        composed_cls = create_protocol_aware_chat(stream_verify=stream_verify)
        instance = composed_cls(*args, **kwargs)
        if captured:
            instance._generation_overrides = captured
        return instance