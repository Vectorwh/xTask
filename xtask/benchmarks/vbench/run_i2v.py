#!/usr/bin/env python3
"""Generate a VBench I2V video set through xLLM's video API.

This driver deliberately uses only Python's standard library.  It can run in
the VBench container without importing xTask or the xLLM Python package:
images are encoded locally, requests are sent to the already running xLLM
endpoint, and returned MP4 data is written to the case output directory.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LOGGER = logging.getLogger("xtask.vbench")


def _records_from_payload(payload: Any) -> Tuple[List[Dict[str, Any]], Any]:
    if isinstance(payload, list):
        records = payload
        return [item for item in records if isinstance(item, dict)], records
    if not isinstance(payload, dict):
        raise ValueError("VBench full_info must be a JSON list or object")
    for key in ("full_info", "data", "items", "prompts"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)], payload
    raise ValueError(
        "VBench full_info object must contain one of: full_info, data, items, prompts"
    )


def _first_value(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", []):
            return value
    return None


def _image_candidate(record: Dict[str, Any]) -> Optional[str]:
    value = _first_value(
        record,
        (
            "image_path",
            "image_name",
            "image_file",
            "image",
            "input_image",
            "image_url",
        ),
    )
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = _first_value(value, ("path", "image_path", "name", "url"))
    return str(value) if value not in (None, "") else None


def _prompt(record: Dict[str, Any]) -> str:
    # Official VBench I2V full_info uses prompt_en; keep the generic aliases
    # for portable/custom datasets accepted by earlier xTask prototypes.
    value = _first_value(
        record,
        ("prompt_en", "prompt", "text", "caption", "description", "text_prompt"),
    )
    if isinstance(value, dict):
        value = _first_value(value, ("text", "prompt", "content"))
    if value in (None, ""):
        raise ValueError("VBench entry has no prompt/text field")
    return str(value)


def _resolve_image(image_root: Path, candidate: str, resolution: str = "") -> Path:
    if candidate.startswith("data:"):
        raise ValueError("data URI images are not supported; provide an image file")
    path = Path(candidate)
    if path.is_absolute():
        return path
    # 官方 VBench 数据集把图片放在 image_root 的子目录里：
    # data/origin/（原图）与 data/crop/{16-9, 8-5, ...}/（裁剪图）。
    # 生成请求按分辨率优先取裁剪图，回退原图。
    subdirs = []
    if resolution:
        subdirs.append(Path("crop") / resolution)
    subdirs.append(Path("origin"))
    for subdir in subdirs:
        candidate_path = image_root / subdir / path
        if candidate_path.is_file():
            return candidate_path
    return image_root / path


def _encode_image(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"input image does not exist: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return encoded


def _request_video(
    endpoint: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"xLLM returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"xLLM request failed: {exc.reason}") from exc
    try:
        decoded = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("xLLM returned a non-JSON video response") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("xLLM video response is not a JSON object")
    return decoded


def _extract_video(response: Dict[str, Any]) -> str:
    output = response.get("output")
    results = output.get("results") if isinstance(output, dict) else None
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise RuntimeError("xLLM video response contains no output.results[0]")
    video = results[0].get("video")
    if not isinstance(video, str) or not video:
        raise RuntimeError("xLLM video response contains no Base64 video")
    if "," in video and video.startswith("data:"):
        video = video.split(",", 1)[1]
    return video


def _decode_video(video: str) -> bytes:
    try:
        return base64.b64decode(video, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise RuntimeError("xLLM returned invalid Base64 video data") from exc


def _video_filename(record: Dict[str, Any], index: int) -> str:
    """Deterministic file name that preserves the camera-motion keyword.

    VBench's camera_motion dimension parses the movement type from the video
    file name (``get_type(os.path.basename(...))`` matches phrases like
    "camera pans left"), so a purely numeric name breaks that dimension.
    Only entries that actually carry the camera_motion dimension get the
    prompt slug; everything else keeps the plain numeric name so old outputs
    stay resumable.
    """
    if "camera_motion" not in (record.get("dimension") or []):
        return f"{index:05d}.mp4"
    prompt = _prompt(record) or ""
    slug = re.sub(r"[^\w\s-]", "", prompt).strip().replace(" ", "_")
    if len(slug) > 80:
        slug = slug[:80]
    return f"{index:05d}_{slug}.mp4" if slug else f"{index:05d}.mp4"


def _existing_video(
    record: Dict[str, Any], save_path: Path, default_path: Optional[Path] = None
) -> Optional[Path]:
    """Return a usable video from a prior run, if one is available.

    VBench manifests normally carry ``video_list`` after generation.  The
    official generator also treats an already materialized output file as a
    resumable item, so include the deterministic per-record path as a fallback
    for interrupted runs whose manifest was not written yet.
    """
    values = record.get("video_list")
    candidates = list(values) if isinstance(values, list) else []
    if default_path is not None:
        candidates.append(default_path)
    for value in candidates:
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = save_path / path
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _copy_payload_with_records(payload: Any, records: List[Dict[str, Any]]) -> Any:
    if isinstance(payload, list):
        return records
    output = dict(payload)
    for key in ("full_info", "data", "items", "prompts"):
        if isinstance(payload.get(key), list):
            output[key] = records
            return output
    return records


def generate(args: argparse.Namespace) -> int:
    full_info_path = Path(args.full_info)
    image_root = Path(args.image_root)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    output_full_info = Path(args.output_full_info)
    output_full_info.parent.mkdir(parents=True, exist_ok=True)

    with full_info_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    records, _ = _records_from_payload(payload)
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    if not records:
        raise ValueError("VBench full_info contains no usable records")

    generated = 0
    resumed = 0
    failures: List[Dict[str, Any]] = []
    latencies: List[float] = []
    output_records: List[Dict[str, Any]] = []
    endpoint = args.endpoint.rstrip("/")
    for index, original in enumerate(records):
        record = dict(original)
        # camera_motion 的官方实现从文件名解析运动类型
        # （get_type(os.path.basename(video_path)) 匹配 "camera pans left"
        # 等关键字），纯数字文件名会让该维度直接 ValueError。
        video_path = save_path / f"{_video_filename(original, index)}"
        existing = _existing_video(record, save_path, video_path)
        if existing is not None:
            video_path = existing
            record["video_list"] = [str(video_path.resolve())]
            resumed += 1
            output_records.append(record)
            continue

        try:
            image_name = _image_candidate(record)
            if not image_name:
                raise ValueError("VBench entry has no image path/name field")
            image_path = _resolve_image(image_root, image_name, args.resolution)
            prompt = _prompt(record)
            seed = record.get("seed", args.seed + index)
            parameters: Dict[str, Any] = {
                "size": args.size,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "seed": int(seed),
                "num_frames": args.num_frames,
                "fps": args.fps,
            }
            optional_parameters = {
                "guidance_scale_2": args.guidance_scale_2,
                "num_videos_per_prompt": args.num_videos_per_prompt,
                "max_sequence_length": args.max_sequence_length,
                "seconds": args.seconds,
                "boundary_ratio": args.boundary_ratio,
                "flow_shift": args.flow_shift,
                "true_cfg_scale": args.true_cfg_scale,
            }
            parameters.update(
                {key: value for key, value in optional_parameters.items() if value is not None}
            )
            input_data: Dict[str, Any] = {
                "prompt": prompt,
                "image": _encode_image(image_path),
            }
            negative_prompt = _first_value(record, ("negative_prompt", "negative"))
            if negative_prompt not in (None, ""):
                input_data["negative_prompt"] = str(negative_prompt)
            request_payload = {
                "model": args.model,
                "input": input_data,
                "parameters": parameters,
            }
            started = time.monotonic()
            response = _request_video(endpoint, request_payload, args.request_timeout)
            video_bytes = _decode_video(_extract_video(response))
            if not video_bytes:
                raise RuntimeError("xLLM returned an empty video")
            video_path.write_bytes(video_bytes)
            if video_path.stat().st_size == 0:
                raise RuntimeError("generated video is empty")
            latencies.append(time.monotonic() - started)
            record["video_list"] = [str(video_path.resolve())]
            generated += 1
            output_records.append(record)
            LOGGER.info("generated VBench video %d/%d: %s", index + 1, len(records), video_path)
        except Exception as exc:  # keep the manifest useful for partial reruns
            LOGGER.error("VBench video %d failed: %s", index, exc)
            failures.append({"index": index, "error": str(exc)})
            output_records.append(record)

    result_payload = _copy_payload_with_records(payload, output_records)
    with output_full_info.open("w", encoding="utf-8") as stream:
        json.dump(result_payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    metrics_path = save_path.parent / "generation_metrics.json"
    metrics = {
        "total": len(records),
        "generated": generated,
        "resumed": resumed,
        "failed": len(failures),
        "latency_seconds": latencies,
        "average_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "failures": failures,
        "endpoint": endpoint,
        "model": args.model,
    }
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if failures and not args.allow_failures:
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-info", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument(
        "--resolution",
        default="16-9",
        help="参考图裁剪比例，用于在 image_root/crop/{resolution} 优先取图",
    )
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--output-full-info", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--guidance-scale-2", type=float)
    parser.add_argument("--num-videos-per-prompt", type=int)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--seconds", type=int)
    parser.add_argument("--boundary-ratio", type=float)
    parser.add_argument("--flow-shift", type=float)
    parser.add_argument("--true-cfg-scale", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-failures", action="store_true")
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        return generate(_parser().parse_args())
    except Exception as exc:
        LOGGER.error("VBench generation failed before completion: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
