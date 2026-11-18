"""Protocol-aware API adapter for AISBench (PyPI 3.1.x async architecture).

Extends AISBench's VLLMCustomAPIChat to support:
1. Per-sample extra_params passthrough (tools, tool_choice, response_format,
   guided_json, guided_regex, guided_grammar, extra_body).
2. Full message preservation (reasoning_content, tool_calls, finish_reason)
   instead of extracting only choices[0].message.content.
3. Optional stream verification: SSE delta chunks are assembled per the
   OpenAI streaming spec and compared against the non-stream result; field
   misalignment or tag leakage (e.g. <think> in content) marks the case
   as a protocol error.

PyPI AISBench (>= 3.1) async request flow:
    GenInferencer.do_request(data, ...)
        -> data.pop("index"/"prompt"/"data_abbr"/"max_out_len"/"gold")
        -> await model.generate(input, max_out_len, output, session, **data)
    BaseAPIModel.generate(input_data, max_out_len, output, session, **args)
        -> request_body = await get_request_body(input, max_out_len, output, **args)
        -> await text_infer(request_body, output)   # non-stream
           or stream_infer(request_body, output)     # stream
    text_infer -> parse_text_response(json_content, output)
    stream_infer -> parse_stream_response(json_content, output) per chunk

Per-sample extra_params flows through:
    ProtocolGenInferencer.get_data_list() puts extra_params into data_list
    -> do_request(**data) includes extra_params
    -> generate(**args) includes extra_params
    -> get_request_body(**args) extracts and injects protocol fields

Warmup defense:
    AISBench's task runner calls inferencer.warmup(data_list, ...) before
    the main inference loop, sending warmup requests through the same
    do_request path.  data_list entries carry extra_params (injected by
    ProtocolGenInferencer.get_data_list()); if those reach the server in
    a warmup request, some servers return 400 Bad Request and the task
    aborts.  ProtocolGenInferencer.warmup() strips extra_params from
    warmup data (creating stripped copies) so get_request_body() receives
    extra_params=None during warmup and inject_protocol_fields() is a
    no-op.  The original data_list is untouched for real inference.

Full message preservation:
    parse_text_response() / stream_infer() store the complete message as a
    JSON string in output.content (clearing output.reasoning_content) so
    that Output.get_prediction() returns the full message JSON. The
    evaluator (ProtocolAndConstrainedEvaluator) parses it back to a dict.

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
REASONING_LEAK_TAGS = ("<think>", "</think>", "<|begin_of_thought|>", "<|end_of_thought|>")


# Parameters that belong in generation_kwargs / request payloads, NOT in the
# model constructor.  If they appear as top-level model dict keys, they are
# popped defensively in ProtocolAwareAPIChat.__new__ to avoid
# ``VLLMCustomAPIChat.__init__() got an unexpected keyword argument`` and
# merged into generation_kwargs so they reach each request payload.
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

def _get_base_class():
    """Lazily import AISBench VLLMCustomAPIChat.

    PyPI 3.1.x merges stream/non-stream into a single class controlled by
    the ``stream`` constructor flag; VLLMCustomAPIChatStream is a deprecated
    subclass that just sets stream=True.
    """
    from ais_bench.benchmark.models import VLLMCustomAPIChat
    return VLLMCustomAPIChat


# ---------------------------------------------------------------------------
# Factory (creates classes bound to the installed AISBench)
# ---------------------------------------------------------------------------

def create_protocol_aware_chat(stream_verify: bool = False):
    """Factory: create a ProtocolAwareAPIChat class bound to the currently
    installed AISBench (PyPI 3.1.x async architecture).

    Args:
        stream_verify: if True, enable SSE assembly + non-stream comparison
            verification after stream_infer completes.

    This factory pattern avoids import errors when ais_bench is not
    installed (e.g. running unit tests on the host).
    """
    VLLMCustomAPIChat = _get_base_class()

    class _ProtocolAwareAPIChatImpl(VLLMCustomAPIChat):
        """Protocol-aware chat adapter for PyPI AISBench async architecture.

        Overrides async hooks in the BaseAPIModel call chain:
        - generate: convert server rejections (400) of protocol-field
          requests into evaluable predictions (finish_reason="error")
        - get_request_body: inject per-sample protocol fields from extra_params
        - parse_text_response: store full message JSON in output.content
        - parse_stream_response + stream_infer: assemble SSE deltas into a
          full message and optionally verify against non-stream result
        """

        def __init__(self, *args, **kwargs):
            self._verify_enabled = kwargs.pop("stream_verify", stream_verify)
            super().__init__(*args, **kwargs)

        async def generate(
            self, input_data, max_out_len, output, session=None, **args
        ):
            """Override to convert server rejections into evaluable predictions.

            For protocol testing, a server rejection (e.g., 400 Bad Request
            for an unsupported response_format) IS the test result — the
            server doesn't support that protocol feature.  The base class
            marks these as failures (output.success=False), causing AISBench
            to exclude them from predictions; the evaluator then sees "No
            predictions found" and can't score anything.

            This override converts failures into synthetic predictions that
            record the server error, so the evaluator can score them as
            protocol failures (finish_reason="error" won't match expected,
            content="" fails syntax validation).

            Only requests that carried protocol fields (extra_params) are
            converted.  Warmup requests and requests without protocol fields
            still report failures normally.
            """
            await super().generate(
                input_data, max_out_len, output, session=session, **args
            )
            if not output.success and getattr(
                output, "_has_protocol_fields", False
            ):
                error_msg = {
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [],
                    "finish_reason": "error",
                    "role": "assistant",
                    "server_error": output.error_info or "unknown error",
                }
                output.content = json.dumps(error_msg, ensure_ascii=False)
                output.reasoning_content = ""
                output.error_info = ""
                output.success = True
            return output

        async def get_request_body(
            self, input, max_out_len, output, **args
        ):
            # Extract per-sample extra_params (dict or JSON string from
            # ProtocolConstrainedDataset serialization).
            extra_params = parse_extra_params(args.pop("extra_params", None))
            # Let the base class build the standard OpenAI payload
            # (messages, model, max_tokens, stream, generation_kwargs, tools).
            request_body = await super().get_request_body(
                input, max_out_len, output, **args
            )
            # Inject per-sample protocol/constrained-decoding fields.
            # Warmup safety: ProtocolGenInferencer.warmup() strips
            # extra_params from warmup data_list entries, so during
            # warmup extra_params is None and this is a no-op.
            inject_protocol_fields(request_body, extra_params)
            # Flag for generate(): whether this request carries protocol
            # fields. Used to decide if a server rejection (400) should be
            # converted to an evaluable prediction instead of a failure.
            output._has_protocol_fields = bool(extra_params)
            return request_body

        async def parse_text_response(self, json_content, output):
            # Let the base class fill content / reasoning_content / usage
            # and call update_extra_details_data_from_text_response.
            await super().parse_text_response(json_content, output)
            # Override: replace content with the full message JSON so
            # get_prediction() returns the complete message structure
            # (content + reasoning_content + tool_calls + finish_reason).
            full_msg = extract_full_message(json_content)
            output.content = json.dumps(full_msg, ensure_ascii=False)
            output.reasoning_content = ""

        async def parse_stream_response(self, json_content, output):
            # Let the base class accumulate content / reasoning_content.
            await super().parse_stream_response(json_content, output)
            # Collect raw chunks for SSE delta assembly in stream_infer.
            if not hasattr(output, "_protocol_delta_chunks"):
                output._protocol_delta_chunks = []
            output._protocol_delta_chunks.append(json_content)

        async def stream_infer(self, request_body, output):
            await super().stream_infer(request_body, output)
            if not output.success:
                return
            # Assemble full message from collected SSE deltas.
            delta_chunks = getattr(output, "_protocol_delta_chunks", [])
            if not delta_chunks:
                return
            stream_msg = assemble_sse_deltas(delta_chunks)
            # Optional: verify stream consistency against a non-stream
            # request with the same payload.
            if self._verify_enabled:
                try:
                    report = await self._verify_against_non_stream(
                        request_body, stream_msg
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
            # Store full message JSON as content so get_prediction()
            # returns the complete message structure.
            output.content = json.dumps(stream_msg, ensure_ascii=False)
            output.reasoning_content = ""

        async def _verify_against_non_stream(
            self, request_body: dict, stream_msg: Dict[str, Any]
        ) -> Optional[Dict[str, Any]]:
            """Make a non-stream HTTP request with the same payload and
            compare against the assembled stream message.

            Reuses the aiohttp session already opened by generate().
            """
            verify_body = dict(request_body)
            verify_body["stream"] = False
            verify_body.pop("stream_options", None)
            try:
                async with self.session.post(
                    url=self.url, json=verify_body, headers=self.headers
                ) as response:
                    if response.status != 200:
                        logger.debug(
                            "Non-stream verification got status %s",
                            response.status,
                        )
                        return None
                    body = await response.json()
            except Exception:
                logger.debug(
                    "Non-stream verification request failed",
                    exc_info=True,
                )
                return None
            non_stream_msg = extract_full_message(body)
            return verify_stream_consistency(stream_msg, non_stream_msg)

    return _ProtocolAwareAPIChatImpl


class ProtocolAwareAPIChat:
    """Config-safe entry point for protocol-aware chat models.

    mmengine config files are parsed via ``_parse_lazy_import`` which treats
    imported names as lazy objects -- calling them at module level raises
    RuntimeError. This wrapper defers mixin composition to instantiation
    time via ``__new__``, so generated config files can safely reference
    ``type=ProtocolAwareAPIChat`` as a plain name.

    The ``stream_verify`` field in the model config dict enables SSE
    assembly + consistency verification after stream inference.
    """

    def __new__(cls, *args, stream_verify: bool = False, **kwargs):
        # Pop generation-only params that the base constructor doesn't
        # accept (temperature, top_p, ...).  Merge them into
        # generation_kwargs so they reach each request payload via
        # VLLMCustomAPIChat.get_request_body().
        gen_kwargs = dict(kwargs.get("generation_kwargs") or {})
        for key in _GENERATION_ONLY_PARAMS:
            if key in kwargs:
                gen_kwargs[key] = kwargs.pop(key)
        if gen_kwargs:
            kwargs["generation_kwargs"] = gen_kwargs
        composed_cls = create_protocol_aware_chat(stream_verify=stream_verify)
        return composed_cls(*args, **kwargs)