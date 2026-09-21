#!/usr/bin/env python3
"""Offline VBench I2V prompt sampler (delivery report 5.1.2.1).

Standalone CLI.  It is not wired into the xTask scheduler, VBench runner, or
existing YAML cases.  The default ``full_info.json`` symlink is never rewritten.

Method (must match the report):
  CLIP ViT-B/32 text embeddings + KMeans, stratified by
  dimension / image_type / the 7 camera-motion types, taking the sample
  nearest each cluster centroid.

  Official 1118 prompts * 10% -> 107 representative items.
  Only entries whose reference image is already downloaded are eligible.

CLIP ViT-B/32 weights (sampling) are resolved into
``$VBENCH_CACHE_DIR/clip_model/`` (report default: ``~/.cache/vbench/clip_model/``).

Example:
    python tools/sample_vbench_i2v_mini_full_info.py
    python tools/sample_vbench_i2v_mini_full_info.py \\
        --full-info /export/home/models/xllm_test_060/vbench_data/vbench2_i2v_full_info.json \\
        --output /export/home/models/xllm_test_060/vbench_data/vbench2_i2v_mini_full_info.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_DATA_HOME = "/export/home/models/xllm_test_060/vbench_data"
DEFAULT_FULL_INFO = f"{DEFAULT_DATA_HOME}/vbench2_i2v_full_info.json"
DEFAULT_IMAGE_ROOT = f"{DEFAULT_DATA_HOME}/data"
DEFAULT_OUTPUT = f"{DEFAULT_DATA_HOME}/vbench2_i2v_mini_full_info.json"
DEFAULT_RATIO = 0.10
DEFAULT_TARGET = 107
DEFAULT_SEED = 42
DEFAULT_RESOLUTION = "16-9"
CLIP_WEIGHT_NAME = "ViT-B-32.pt"
OPENAI_CLIP_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
)

# Official VBench I2V camera_motion phrases (camera_motion.get_type).
CAMERA_MOTION_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("camera pans left", "pan_left"),
    ("camera pans right", "pan_right"),
    ("camera tilts up", "tilt_up"),
    ("camera tilts down", "tilt_down"),
    ("camera zooms in", "zoom_in"),
    ("camera zooms out", "zoom_out"),
    ("camera static", "static"),
)
PRIMARY_DIMENSIONS = ("camera_motion", "i2v_subject", "i2v_background")
StratumKey = Tuple[str, str, str]


def default_vbench_cache_dir() -> Path:
    env = (os.environ.get("VBENCH_CACHE_DIR") or "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "vbench"


def default_clip_model_dir() -> Path:
    return default_vbench_cache_dir() / "clip_model"


def records_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("VBench full_info must be a JSON list or object")
    for key in ("full_info", "data", "items", "prompts"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    raise ValueError(
        "VBench full_info object must contain one of: full_info, data, items, prompts"
    )


def prompt_text(record: Mapping[str, Any]) -> str:
    for key in ("prompt_en", "prompt", "text", "caption", "description", "text_prompt"):
        value = record.get(key)
        if value not in (None, "", []):
            return str(value)
    return ""


def primary_dimension(record: Mapping[str, Any]) -> str:
    dims = record.get("dimension") or []
    if not isinstance(dims, list):
        dims = [dims]
    as_text = [str(item) for item in dims]
    for name in PRIMARY_DIMENSIONS:
        if name in as_text:
            return name
    return as_text[0] if as_text else "unknown"


def camera_motion_type(record: Mapping[str, Any]) -> str:
    """Parse one of the 7 camera motions from the prompt; else ``none``."""
    dims = record.get("dimension") or []
    if not isinstance(dims, list):
        dims = [dims]
    if "camera_motion" not in [str(item) for item in dims]:
        return "none"
    prompt = prompt_text(record).lower()
    for phrase, name in CAMERA_MOTION_PHRASES:
        if phrase in prompt:
            return name
    return "unknown"


def stratum_key(record: Mapping[str, Any]) -> StratumKey:
    image_type = str(record.get("image_type") or "unknown")
    return (primary_dimension(record), image_type, camera_motion_type(record))


def resolve_image_path(
    record: Mapping[str, Any],
    image_root: Path,
    resolution: str = DEFAULT_RESOLUTION,
) -> Optional[Path]:
    candidate = record.get("image_name") or record.get("image_path") or record.get("image")
    if isinstance(candidate, (list, tuple)):
        candidate = candidate[0] if candidate else None
    if not candidate:
        return None
    path = Path(str(candidate))
    if path.is_absolute():
        return path if path.is_file() else None
    subdirs = []
    if resolution:
        subdirs.append(Path("crop") / resolution)
    subdirs.append(Path("origin"))
    for subdir in subdirs:
        candidate_path = image_root / subdir / path
        if candidate_path.is_file():
            return candidate_path
    fallback = image_root / path
    return fallback if fallback.is_file() else None


def filter_downloaded(
    records: Sequence[Mapping[str, Any]],
    image_root: Path,
    resolution: str = DEFAULT_RESOLUTION,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for record in records:
        if resolve_image_path(record, image_root, resolution) is not None:
            kept.append(dict(record))
    return kept


def hamilton_allocate(
    sizes: Mapping[StratumKey, int],
    target: int,
) -> Dict[StratumKey, int]:
    """Largest-remainder allocation of ``target`` seats across strata.

    Caps each stratum at its size, then redistributes leftover seats so the
    returned counts still sum to ``min(target, sum(sizes))``.
    """
    keys = sorted(sizes)
    total = sum(sizes[key] for key in keys)
    if target <= 0 or total <= 0:
        return {key: 0 for key in keys}
    target = min(int(target), total)
    quotas = {key: (sizes[key] * target / total) for key in keys}
    alloc = {key: min(sizes[key], math.floor(quotas[key])) for key in keys}
    leftover = target - sum(alloc.values())
    remainders = sorted(
        ((quotas[key] - math.floor(quotas[key]), key) for key in keys),
        key=lambda item: (-item[0], item[1]),
    )
    for _, key in remainders:
        if leftover <= 0:
            break
        if alloc[key] < sizes[key]:
            alloc[key] += 1
            leftover -= 1
    # If floor+remainder still left seats (tiny strata already capped),
    # keep handing extras to the largest remaining capacity.
    if leftover > 0:
        capacity = sorted(
            keys,
            key=lambda key: (-(sizes[key] - alloc[key]), key),
        )
        while leftover > 0:
            progressed = False
            for key in capacity:
                if leftover <= 0:
                    break
                if alloc[key] < sizes[key]:
                    alloc[key] += 1
                    leftover -= 1
                    progressed = True
            if not progressed:
                break
    return alloc


def _kmeans_centroids(embeddings, k: int, random_state: int):
    """Return (centroids, labels).  Prefer sklearn; numpy fallback otherwise."""
    import numpy as np

    matrix = np.asarray(embeddings, dtype=np.float64)
    n = int(matrix.shape[0])
    if k <= 0:
        raise ValueError("k must be positive")
    if k == 1:
        centroid = matrix.mean(axis=0, keepdims=True)
        return centroid, np.zeros(n, dtype=int)
    if k >= n:
        return matrix.copy(), np.arange(n, dtype=int)

    try:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(matrix)
        return np.asarray(model.cluster_centers_, dtype=np.float64), labels
    except ImportError:
        return _numpy_kmeans(matrix, k, random_state)


def _numpy_kmeans(matrix, k: int, random_state: int, max_iter: int = 100):
    """Lloyd k-means with k-means++ init; used when sklearn is absent."""
    import numpy as np

    rng = np.random.RandomState(random_state)
    n = matrix.shape[0]
    centroids = np.empty((k, matrix.shape[1]), dtype=np.float64)
    first = int(rng.randint(0, n))
    centroids[0] = matrix[first]
    closest_sq = np.full(n, np.inf)
    for i in range(1, k):
        diff = matrix - centroids[i - 1]
        dist_sq = np.sum(diff * diff, axis=1)
        closest_sq = np.minimum(closest_sq, dist_sq)
        total = float(closest_sq.sum())
        if total <= 0:
            centroids[i] = matrix[int(rng.randint(0, n))]
            continue
        centroids[i] = matrix[int(rng.choice(n, p=closest_sq / total))]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = matrix[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
    return centroids, labels


def pick_centroid_representatives(
    embeddings,
    k: int,
    random_state: int,
) -> List[int]:
    """Return local indices nearest each KMeans centroid (no replacement)."""
    import numpy as np

    matrix = np.asarray(embeddings, dtype=np.float64)
    n = int(matrix.shape[0])
    if n == 0 or k <= 0:
        return []
    k = min(int(k), n)
    if k >= n:
        return list(range(n))
    centroids, _labels = _kmeans_centroids(matrix, k, random_state)
    chosen: List[int] = []
    taken = set()
    for centroid in centroids:
        distances = np.linalg.norm(matrix - centroid, axis=1)
        order = distances.argsort()
        for idx in order:
            local = int(idx)
            if local not in taken:
                taken.add(local)
                chosen.append(local)
                break
    return chosen


def group_by_stratum(
    records: Sequence[Mapping[str, Any]],
) -> Dict[StratumKey, List[int]]:
    grouped: Dict[StratumKey, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[stratum_key(record)].append(index)
    return dict(grouped)


def sample_records(
    records: Sequence[Mapping[str, Any]],
    embeddings,
    target: int,
    random_state: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """Stratified centroid sampling.  ``embeddings`` is (N, D) aligned with records."""
    import numpy as np

    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.shape[0] != len(records):
        raise ValueError("embeddings must align 1:1 with records")
    grouped = group_by_stratum(records)
    sizes = {key: len(indexes) for key, indexes in grouped.items()}
    alloc = hamilton_allocate(sizes, target)
    selected_global: List[int] = []
    for key in sorted(grouped):
        k = alloc.get(key, 0)
        if k <= 0:
            continue
        local_indexes = grouped[key]
        local_emb = matrix[local_indexes]
        picked = pick_centroid_representatives(local_emb, k, random_state)
        selected_global.extend(local_indexes[i] for i in picked)
    selected_global.sort()
    return [dict(records[i]) for i in selected_global]


def shared_clip_candidates() -> List[Path]:
    here = Path(__file__).resolve()
    project_root = here.parents[1]
    return [
        Path(DEFAULT_DATA_HOME) / "clip" / CLIP_WEIGHT_NAME,
        project_root / "outputs" / "vbench_data" / "cache" / "clip_model" / CLIP_WEIGHT_NAME,
        Path("/export/home/models/xllm_test_060/vbench_data/clip") / CLIP_WEIGHT_NAME,
    ]


def ensure_clip_weights(
    clip_model_dir: Path,
    clip_weight: Optional[Path] = None,
    allow_download: bool = True,
) -> Path:
    """Materialize ViT-B-32.pt under ``clip_model_dir`` without touching full_info."""
    clip_model_dir.mkdir(parents=True, exist_ok=True)
    dest = clip_model_dir / CLIP_WEIGHT_NAME
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    sources: List[Path] = []
    if clip_weight is not None:
        sources.append(Path(clip_weight))
    sources.extend(shared_clip_candidates())
    for source in sources:
        if source.is_file() and source.stat().st_size > 0:
            if source.resolve() == dest.resolve():
                return dest
            try:
                os.symlink(source, dest)
            except OSError:
                shutil.copy2(source, dest)
            return dest

    if not allow_download:
        raise FileNotFoundError(
            f"CLIP ViT-B/32 weights not found at {dest} and download is disabled"
        )
    _download_file(OPENAI_CLIP_URL, dest)
    return dest


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def encode_prompts_clip(
    prompts: Sequence[str],
    clip_weight: Path,
    device: str = "cpu",
    batch_size: int = 64,
):
    """Encode prompts with OpenAI CLIP ViT-B/32; returns L2-normalized float32."""
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import numpy as np
    import torch

    try:
        import clip
    except ImportError as exc:
        raise ImportError(
            "The OpenAI `clip` package is required to embed prompts. "
            "Run this script in the VBench runner image "
            "(quay.io/jd_xllm/xllm-ai:8.1.rc1-a3-openeuler22.03-py3.10) "
            "with TORCH_DEVICE_BACKEND_AUTOLOAD=0."
        ) from exc

    model, _preprocess = clip.load(str(clip_weight), device=device, jit=False)
    model.eval()
    vectors: List[Any] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            tokens = clip.tokenize(chunk, truncate=True).to(device)
            features = model.encode_text(tokens).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            vectors.append(features.cpu().numpy())
    if not vectors:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(vectors, axis=0).astype(np.float32, copy=False)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)


def load_full_info(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = records_from_payload(payload)
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def summarize_strata(
    records: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> List[str]:
    before = group_by_stratum(records)
    after = group_by_stratum(selected)
    lines = []
    for key in sorted(before):
        lines.append(
            f"  {key[0]:16s} {key[1]:16s} {key[2]:12s} "
            f"{len(before[key]):4d} -> {len(after.get(key, []))}"
        )
    return lines


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a representative VBench I2V mini subset "
            "(CLIP ViT-B/32 + stratified KMeans)."
        )
    )
    parser.add_argument(
        "--full-info",
        type=Path,
        default=Path(DEFAULT_FULL_INFO),
        help="Official 1118-entry full_info JSON (never rewritten)",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(DEFAULT_IMAGE_ROOT),
        help="Reference-image root (crop/<resolution>/ and origin/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Mini full_info JSON to write",
    )
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO, help="Sample ratio (default 0.10)")
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help="Exact sample count to aim for (report: 107). 0 = round(N * ratio)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="KMeans random_state")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="Crop subdirectory, e.g. 16-9")
    parser.add_argument(
        "--clip-cache-dir",
        type=Path,
        default=None,
        help="Directory for ViT-B-32.pt (default: $VBENCH_CACHE_DIR/clip_model)",
    )
    parser.add_argument(
        "--clip-weight",
        type=Path,
        default=None,
        help="Existing CLIP ViT-B-32.pt to copy/symlink into the cache dir",
    )
    parser.add_argument("--device", default="cpu", help="CLIP device (cpu by default)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Download OpenAI ViT-B-32.pt if no local weight is found",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stratum allocation only; do not load CLIP or write output",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    full_info = Path(args.full_info)
    image_root = Path(args.image_root)
    output = Path(args.output)
    default_symlink = Path(DEFAULT_DATA_HOME) / "full_info.json"
    if output.resolve() == default_symlink.resolve() or output.name == "full_info.json":
        raise SystemExit(
            f"refusing to overwrite the default full_info path {output}; "
            "write vbench2_i2v_mini_full_info.json instead"
        )

    records = load_full_info(full_info)
    eligible = filter_downloaded(records, image_root, args.resolution)
    if not eligible:
        raise SystemExit("no full_info entries have a downloaded image under --image-root")

    if int(args.target) > 0:
        target = min(int(args.target), len(eligible))
    else:
        target = max(1, int(round(len(eligible) * float(args.ratio))))

    print(f"loaded {len(records)} prompts from {full_info}")
    print(f"eligible with downloaded images: {len(eligible)} / {len(records)}")
    print(f"target sample size: {target} (ratio={args.ratio}, seed={args.seed})")

    grouped = group_by_stratum(eligible)
    alloc = hamilton_allocate({k: len(v) for k, v in grouped.items()}, target)
    print("stratum allocation (dimension / image_type / camera_motion):")
    for key in sorted(grouped):
        print(
            f"  {key[0]:16s} {key[1]:16s} {key[2]:12s} "
            f"n={len(grouped[key]):4d}  k={alloc.get(key, 0)}"
        )
    print(f"allocated sum = {sum(alloc.values())}")

    if args.dry_run:
        return 0

    clip_model_dir = Path(args.clip_cache_dir) if args.clip_cache_dir else default_clip_model_dir()
    clip_weight = ensure_clip_weights(
        clip_model_dir,
        clip_weight=Path(args.clip_weight) if args.clip_weight else None,
        allow_download=bool(args.allow_download),
    )
    print(f"CLIP ViT-B/32 weights: {clip_weight}")

    prompts = [prompt_text(record) or " " for record in eligible]
    embeddings = encode_prompts_clip(
        prompts,
        clip_weight,
        device=str(args.device),
        batch_size=int(args.batch_size),
    )
    selected = sample_records(eligible, embeddings, target, random_state=int(args.seed))
    atomic_write_json(output, selected)

    print(f"wrote {len(selected)} records to {output}")
    missing = [
        record.get("image_name")
        for record in selected
        if resolve_image_path(record, image_root, args.resolution) is None
    ]
    if missing:
        raise SystemExit(f"sampled records missing images: {missing[:5]}")
    print("sampled images: all present")
    print("per-stratum sampled:")
    for line in summarize_strata(eligible, selected):
        print(line)
    if default_symlink.exists() or default_symlink.is_symlink():
        print(f"left default symlink untouched: {default_symlink}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
