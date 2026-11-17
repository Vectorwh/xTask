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
    """CustomDataset variant that preserves dict-typed extra_params and
    ground_truth fields instead of stringifying them.

    AISBench's base CustomDataset may coerce complex fields; this subclass
    re-parses them from the raw line to guarantee the dict structure.
    """

    def load(self, path=None, meta_path=None, **kwargs):
        """Load JSONL and ensure extra_params / ground_truth stay as dicts.

        AISBench's CustomDataset.__init__ calls ``self.load(**kwargs)`` with
        all dataset-config dict keys (path, meta_path, abbr, reader_cfg, ...).
        The actual JSONL data path is the ``path`` key; ``meta_path`` may be
        an empty string.  Both are accepted explicitly so they don't get
        swallowed by **kwargs.
        """
        import json as _json
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
                # Ensure dict types survive the pipeline.
                if "extra_params" not in row or not isinstance(row["extra_params"], dict):
                    row["extra_params"] = row.get("extra_params") or {}
                if "ground_truth" not in row or not isinstance(row["ground_truth"], dict):
                    row["ground_truth"] = row.get("ground_truth") or {}
                data.append(row)
        return data