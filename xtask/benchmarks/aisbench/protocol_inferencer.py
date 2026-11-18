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
    AISBench sends warmup requests using data_list entries before the
    main inference loop.  Since get_data_list() injects extra_params
    (which may contain response_format / guided_* that some servers
    reject with 400 Bad Request), warm_up() is overridden to temporarily
    strip extra_params from data_list during the warmup phase and restore
    them afterwards.  The model-side defense (ProtocolAwareAPIChat
    ``_in_warmup`` flag) provides a second layer of protection.

Referenced as ``type=ProtocolGenInferencer`` in the dataset config
(protocol_constrained_gen.py), which is lazy-import-safe.
"""

import inspect

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
    # Warmup defense (Layer 1: inferencer level)
    # ------------------------------------------------------------------

    def warm_up(self, *args, **kwargs):
        """Strip extra_params from data_list during warmup.

        AISBench's default warmup sends requests using data_list entries.
        If those entries carry extra_params (response_format, guided_*,
        etc.), the warmup request body will include protocol fields that
        some inference servers reject with 400 Bad Request, causing the
        entire task to abort ("all warmup requests failed").

        This override temporarily removes extra_params from every
        data_list item before delegating to the parent warm_up(), then
        restores them in a finally block so the real inference requests
        still carry their per-sample protocol parameters.

        Handles both sync and async parent signatures defensively.
        """
        parent_warm_up = getattr(super(), "warm_up", None)
        if parent_warm_up is None:
            return  # Parent has no warm_up; nothing to defend.

        saved = self._strip_extra_params()
        try:
            result = parent_warm_up(*args, **kwargs)
        except BaseException:
            self._restore_extra_params(saved)
            raise
        # If the parent warm_up is async, wrap the coroutine so
        # extra_params are restored after it completes.
        if inspect.iscoroutine(result):
            async def _wrapped_warm_up():
                try:
                    return await result
                finally:
                    self._restore_extra_params(saved)
            return _wrapped_warm_up()
        self._restore_extra_params(saved)
        return result

    def _strip_extra_params(self):
        """Remove extra_params from all data_list items.

        Returns a list of the removed values (or None per item) aligned
        with data_list indices for restoration.
        """
        saved = []
        data_list = getattr(self, "data_list", None) or []
        for item in data_list:
            if isinstance(item, dict):
                saved.append(item.pop("extra_params", None))
            else:
                saved.append(None)
        return saved

    def _restore_extra_params(self, saved):
        """Restore previously stripped extra_params into data_list."""
        data_list = getattr(self, "data_list", None) or []
        for item, ep in zip(data_list, saved):
            if ep is not None and isinstance(item, dict):
                item["extra_params"] = ep