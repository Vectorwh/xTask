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