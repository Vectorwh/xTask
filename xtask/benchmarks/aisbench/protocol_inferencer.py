"""Protocol-aware GenInferencer for AISBench (PyPI 3.1.x async architecture).

The stock GenInferencer.get_data_list() only puts prompt / data_abbr /
index / max_out_len / gold into each data item.  The per-sample
``extra_params`` column (tools, tool_choice, response_format, guided_*,
extra_body) never reaches do_request(), so it cannot flow through
``generate(**args)`` to ``get_request_body(**args)``.

This subclass injects ``extra_params`` into each data_list entry so the
standard async pipeline carries it end-to-end:

    get_data_list() -> data["extra_params"] = ...
    -> do_request(**data)  (extra_params survives the pops)
    -> model.generate(input, max_out_len, output, session, **data)
    -> get_request_body(input, max_out_len, output, **args)
    -> ProtocolAwareAPIChat extracts & injects protocol fields

Warmup defense:
    AISBench's task runner (openicl_api_infer.py) calls
    ``inferencer.warmup(data_list, warmup_times, concurrency)`` before
    the main inference loop, sending warmup requests through the same
    ``do_request`` path.  Since ``get_data_list()`` has already injected
    ``extra_params`` into every data_list entry, warmup requests would
    carry protocol fields (response_format, guided_*, tools) that some
    inference servers reject with 400 Bad Request, aborting the entire
    task ("all warmup requests failed").

    ``warmup()`` is overridden to strip ``extra_params`` from the
    data_list entries (creating stripped copies, leaving the originals
    untouched for real inference) before delegating to the parent.

Referenced as ``type=ProtocolGenInferencer`` in the dataset config
(protocol_constrained_gen.py), which is lazy-import-safe.
"""

from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer


class ProtocolGenInferencer(GenInferencer):
    """GenInferencer that passes the extra_params column to the model."""

    def get_data_list(self, retriever):
        """Build data_list with extra_params from the dataset test split."""
        data_list = super().get_data_list(retriever)
        try:
            extra_params_list = retriever.test_ds["extra_params"]
        except (KeyError, TypeError, IndexError):
            # Dataset has no extra_params column; nothing to inject.
            return data_list
        if extra_params_list is None:
            return data_list
        for index, item in enumerate(data_list):
            if index < len(extra_params_list):
                item["extra_params"] = extra_params_list[index]
        return data_list

    # ------------------------------------------------------------------
    # Warmup defense
    # ------------------------------------------------------------------

    async def warmup(self, data_list, warmup_times=1, concurrency=1):
        """Strip extra_params from warmup data to avoid 400 Bad Request.

        AISBench's task runner sends warmup requests through the same
        ``do_request`` path as real inference.  data_list entries
        produced by ``get_data_list()`` carry ``extra_params`` (protocol
        fields like response_format / guided_*); if those reach the
        server in a warmup request, some servers return 400 and the
        entire task aborts.

        This override creates stripped copies of data_list entries
        (without extra_params) and passes them to the parent warmup().
        The original data_list is NOT modified — the task runner reuses
        it for the real inference loop after warmup completes.
        """
        stripped_list = [
            {k: v for k, v in item.items() if k != "extra_params"}
            if isinstance(item, dict) else item
            for item in data_list
        ]
        return await super().warmup(stripped_list, warmup_times, concurrency)