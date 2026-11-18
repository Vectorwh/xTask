"""Protocol-constrained dataset for AISBench.

These classes live in a regular Python module (NOT inside an mmengine config
file) because mmengine's ``_parse_lazy_import`` treats ``from X import Y`` as
a lazy object — subclassing a lazy object at module level (``class Foo(Bar)``)
triggers ``LazyAttr.__call__`` which raises ``RuntimeError``.

Config files reference these classes as ``type=ProtocolConstrainedDataset``
values, which is the lazy-import-safe pattern (same as ``type=PromptTemplate``).
"""

from ais_bench.benchmark.datasets import CustomDataset


class ProtocolConstrainedDataset(CustomDataset):
    """CustomDataset variant that serializes dict-typed extra_params and
    ground_truth into JSON strings for HuggingFace Dataset compatibility.

    Rationale: ``Dataset.from_list()`` uses pyarrow struct inference, which
    raises ``ArrowInvalid: cannot mix struct and non-struct, non-null values``
    when the same key has different value types across rows — e.g.
    ``tool_choice`` is the string ``"none"`` in one row and a dict
    ``{"type": "function", ...}`` in another (both valid per OpenAI API spec).

    Serializing to JSON strings sidesteps pyarrow type inference entirely.
    Consumers deserialize back to dicts via ``_ensure_dict()``:
      - protocol_api._capture_extra_params() for extra_params
      - protocol_constrained_evaluator.score() for ground_truth references
    """

    def load(self, path=None, meta_path=None, **kwargs):
        """Load JSONL, serializing extra_params / ground_truth to JSON strings.

        AISBench's CustomDataset.__init__ calls ``self.load(**kwargs)`` with
        all dataset-config dict keys (path, meta_path, abbr, reader_cfg, ...).
        The actual JSONL data path is the ``path`` key; ``meta_path`` may be
        an empty string.  Both are accepted explicitly so they don't get
        swallowed by **kwargs.

        Returns a HuggingFace ``datasets.Dataset`` (required by AISBench's
        ``check_dataset``), NOT a plain Python list.
        """
        import json as _json
        from datasets import Dataset as _HFDataset

        file_path = path or meta_path or getattr(self, "path", None)
        if not file_path:
            raise ValueError(
                "ProtocolConstrainedDataset.load() requires a non-empty "
                "path, meta_path, or self.path"
            )
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = _json.loads(line)
                except _json.JSONDecodeError as exc:
                    raise ValueError(
                        f"protocol_constrained JSONL parse error at line {line_no}: {exc}"
                    ) from exc
                # Serialize complex fields to JSON strings so pyarrow never
                # attempts struct inference (which fails on heterogeneous
                # value types across rows, e.g. tool_choice str vs dict).
                for _field in ("extra_params", "ground_truth"):
                    _val = row.get(_field)
                    if isinstance(_val, dict):
                        row[_field] = _json.dumps(_val, ensure_ascii=False)
                    elif not isinstance(_val, str):
                        row[_field] = "{}"
                data.append(row)
        return _HFDataset.from_list(data)