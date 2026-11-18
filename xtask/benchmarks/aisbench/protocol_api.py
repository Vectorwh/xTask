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
# later merged into each request payload in _build_payload.
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


# ---------------------------------------------------------------------------
# AISBench model adapters (import ais_bench lazily)
# ---------------------------------------------------------------------------

def _get_base_classes():
    """Lazily import AISBench base classes. Returns (chat_cls, stream_cls)."""
    from ais_bench.benchmark.models import VLLMCustomAPIChat, VLLMCustomAPIChatStream
    return VLLMCustomAPIChat, VLLMCustomAPIChatStream


class ProtocolAwareMixin:
    """Mixin providing per-sample extra_params injection and full-message
    preservation for AISBench API model adapters.

    Subclasses must call ``_capture_extra_params(kwargs)`` at the start of
    ``generate()`` and use ``_current_extra_params`` when building payloads.
    """

    def _capture_extra_params(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pop extra_params from generate() kwargs and store for this request.

        Handles both dict (direct call / legacy) and JSON-string (from
        ProtocolConstrainedDataset, which serializes complex fields to
        avoid pyarrow struct inference failures) representations.
        """
        extra = kwargs.pop("extra_params", None)
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except (json.JSONDecodeError, TypeError):
                extra = None
        self._current_extra_params = extra if isinstance(extra, dict) else None
        return self._current_extra_params

    def _inject_current(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Inject the captured extra_params and generation-only overrides
        into the payload."""
        payload = inject_protocol_fields(
            payload, getattr(self, "_current_extra_params", None)
        )
        # Apply generation-only params (temperature etc.) captured at
        # construction time; use setdefault so per-request kwargs win.
        overrides = getattr(self, "_generation_overrides", {})
        for key, value in overrides.items():
            payload.setdefault(key, value)
        return payload

    def _store_full_message(self, request_id: str, message: Dict[str, Any]):
        """Store the full message for later retrieval by the evaluator."""
        if not hasattr(self, "_full_messages"):
            self._full_messages = {}
        self._full_messages[request_id] = message


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

    if stream_verify:
        class ProtocolAwareAPIChatStream(ProtocolAwareMixin, VLLMCustomAPIChatStream):
            """Stream adapter with SSE assembly and consistency verification."""

            def __init__(self, *args, **kwargs):
                self._verify_enabled = kwargs.pop("stream_verify", True)
                super().__init__(*args, **kwargs)
                self._current_extra_params = None
                self._full_messages: Dict[str, Dict[str, Any]] = {}

            async def generate(self, prompt, max_out_len, output, session=None, **kwargs):
                self._capture_extra_params(kwargs)
                # Let the parent handle the streaming request; the raw SSE
                # chunks are intercepted via _parse_stream_chunk hook below.
                result = await super().generate(prompt, max_out_len, output, session=session, **kwargs)
                return result

            def _build_payload(self, prompt, **kwargs):
                payload = super()._build_payload(prompt, **kwargs)
                return self._inject_current(payload)

            def _parse_stream_chunk(self, raw_line: str) -> Optional[Dict[str, Any]]:
                """Intercept each SSE data line for delta assembly."""
                self._sse_chunks = getattr(self, "_sse_chunks", [])
                if not raw_line or not raw_line.startswith("data:"):
                    return None
                data_str = raw_line[len("data:"):].strip()
                if data_str == "[DONE]":
                    return None
                try:
                    chunk = json.loads(data_str)
                    self._sse_chunks.append(chunk)
                    return chunk
                except json.JSONDecodeError:
                    logger.warning("Failed to parse SSE chunk: %s", data_str[:100])
                    return None

            def get_assembled_message(self) -> Dict[str, Any]:
                """Assemble collected SSE deltas into a complete message."""
                return assemble_sse_deltas(getattr(self, "_sse_chunks", []))

        return ProtocolAwareAPIChatStream

    class _ProtocolAwareAPIChatImpl(ProtocolAwareMixin, VLLMCustomAPIChat):
        """Non-stream adapter with per-sample protocol field injection and
        full-message preservation."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._current_extra_params = None
            self._full_messages: Dict[str, Dict[str, Any]] = {}

        async def generate(self, prompt, max_out_len, output, session=None, **kwargs):
            self._capture_extra_params(kwargs)
            return await super().generate(prompt, max_out_len, output, session=session, **kwargs)

        def _build_payload(self, prompt, **kwargs):
            payload = super()._build_payload(prompt, **kwargs)
            return self._inject_current(payload)

        def _parse_response(self, response_json: Dict[str, Any]) -> str:
            """Override response parsing to preserve the full message.

            Returns the full message (content + reasoning_content +
            tool_calls + finish_reason) as a JSON string so that protocol
            fields flow through AISBench's standard prediction pipeline.
            The ProtocolAndConstrainedEvaluator parses this JSON back.
            """
            full_msg = extract_full_message(response_json)
            request_id = response_json.get("id", str(id(response_json)))
            self._store_full_message(request_id, full_msg)
            return json.dumps(full_msg, ensure_ascii=False)

    return _ProtocolAwareAPIChatImpl


class ProtocolAwareAPIChat:
    """Config-safe entry point for protocol-aware chat models.

    mmengine config files are parsed via ``_parse_lazy_import`` which treats
    imported names as lazy objects — calling them at module level raises
    RuntimeError. This wrapper defers mixin composition to instantiation
    time via ``__new__``, so generated config files can safely reference
    ``type=ProtocolAwareAPIChat`` as a plain name.

    The ``stream_verify`` field in the model config dict selects the stream
    adapter (SSE assembly + consistency verification) vs the non-stream one.
    """

    def __new__(cls, *args, stream_verify: bool = False, **kwargs):
        # Pop generation-only params that the base constructor doesn't
        # accept (temperature, top_p, ...).  They are stored on the instance
        # and merged into each request payload by _inject_current.
        captured: Dict[str, Any] = {}
        for key in _GENERATION_ONLY_PARAMS:
            if key in kwargs:
                captured[key] = kwargs.pop(key)
        composed_cls = create_protocol_aware_chat(stream_verify=stream_verify)
        instance = composed_cls(*args, **kwargs)
        if captured:
            instance._generation_overrides = captured
        return instance