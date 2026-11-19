# Copyright 2026 The xTask Authors.
"""Validated external JoyOV2 features in the native xLLM Tensor JSON format."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .artifacts import sha256


def load_embedding(descriptor, manifest_root: Path, dimension: int, provenance):
    if not isinstance(descriptor, dict) or set(descriptor) - {"path", "sha256"}:
        raise ValueError("embedding must contain path and sha256")
    path = Path(descriptor.get("path", ""))
    path = path if path.is_absolute() else manifest_root / path
    expected_hash = descriptor.get("sha256", "")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or sha256(path) != expected_hash:
        raise ValueError(f"Embedding hash mismatch: {path}")
    if not isinstance(provenance, dict) or not all(provenance.get(k) for k in (
        "checkpoint", "tokenizer", "template_sha256", "feature", "dtype",
    )):
        raise ValueError("Embedding requires encoder provenance: checkpoint/tokenizer/template_sha256/feature/dtype")
    if not isinstance(provenance["template_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", provenance["template_sha256"]):
        raise ValueError("Encoder template_sha256 must be an actual SHA256 digest")
    if provenance["feature"] != "qwen3_vl_pre_norm_residual":
        raise ValueError("JoyOV2 requires its matching Qwen3-VL pre-norm residual, not H3/generic pooled features")
    tensor = json.loads(path.read_text())
    if not isinstance(tensor, dict) or set(tensor) - {"name", "datatype", "shape", "contents", "parameters"}:
        raise ValueError("Unsupported embedding Tensor fields")
    # Typed FP32 protobuf transport is deliberately the first supported format.
    # BF16/FP16 binary requires an additional audited attachment implementation.
    if tensor.get("datatype") != "FP32" or provenance["dtype"] != "FP32":
        raise ValueError("Stage one external embedding transport supports explicit FP32 Tensor JSON only")
    shape = tensor.get("shape")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)
            or shape[-1] != dimension):
        # DiTBatch adds its own leading batch dimension. Sending [1,T,D]
        # would produce [1,1,T,D] and fail inside the model.
        raise ValueError(f"JoyOV2 wire embedding must be [T,{dimension}]; DiTBatch supplies batch dimension")
    contents = tensor.get("contents") or {}
    values = contents.get("fp32_contents")
    if set(contents) != {"fp32_contents"} or not isinstance(values, list) or len(values) != math.prod(shape):
        raise ValueError("Embedding shape/FP32 contents length mismatch")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 3.4028234663852886e38 for v in values):
        raise ValueError("Embedding must contain finite numbers")
    parameters = tensor.get("parameters") or {}
    if set(parameters) - {"joyo_text_seqlens"}:
        raise ValueError("Only joyo_text_seqlens Tensor parameter is supported")
    if "joyo_text_seqlens" in parameters:
        block = parameters["joyo_text_seqlens"]
        seq = block.get("int64_tensor_param") if isinstance(block, dict) else None
        if (not isinstance(seq, list) or not seq or set(block) != {"int64_tensor_param"}
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in seq)
                or sum(seq) != shape[-2]):
            raise ValueError("Invalid joyo_text_seqlens")
    return tensor, {"source_path": str(path), "sha256": expected_hash, "shape": shape,
                    "datatype": "FP32", "encoder": provenance, "encoder_provenance_verified": False}


def validate_cfg(tensor, negative, guidance):
    if guidance <= 1:
        seq = (tensor.get("parameters") or {}).get("joyo_text_seqlens", {}).get("int64_tensor_param")
        if seq is not None and len(seq) != 1:
            raise ValueError("Non-CFG embedding requires one sequence")
        return
    if negative is not None:
        for value in (tensor, negative):
            seq = (value.get("parameters") or {}).get("joyo_text_seqlens", {}).get("int64_tensor_param")
            if seq is not None and len(seq) != 1:
                raise ValueError("Separate positive/negative features must each contain one sequence")
        return
    seq = (tensor.get("parameters") or {}).get("joyo_text_seqlens", {}).get("int64_tensor_param")
    if seq is None or len(seq) != 2:
        raise ValueError("CFG embedding must provide negative_prompt_embed or explicit packed joyo_text_seqlens; never silently split positive-only features")
