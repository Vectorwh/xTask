# Copyright 2026 The xTask Authors.
"""Explicit decoded-media parity, not byte identity or semantic equivalence."""
from __future__ import annotations

import array
import math
import subprocess
import sys
from pathlib import Path

from .artifacts import write_json


def _decode(path, args, settings, timeout):
    command = [settings.get("ffmpeg", "ffmpeg"), "-nostdin", "-v", "error", "-xerror", "-i", str(path)] + args + ["pipe:1"]
    result = subprocess.run(command, capture_output=True, timeout=timeout, check=True)
    return result.stdout


def compare_pairs(directory: Path, pairs, specification, config):
    size = specification.get("video_size", 64)
    sample_rate = specification.get("audio_sample_rate", 16000)
    timeout = specification.get("timeout", 600)
    scores = []
    for pair in pairs:
        videos = []
        audio = []
        for arm in ("reference", "target"):
            record = pair[arm]
            path = directory / record["video_path"]
            content = _decode(path, ["-map", "0:v:0", "-vf", f"scale={size}:{size}", "-pix_fmt", "rgb24", "-f", "rawvideo"], config["media"], timeout)
            videos.append(content)
            if record.get("media", {}).get("audio"):
                content = _decode(path, ["-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le"], config["media"], timeout)
                values = array.array("f")
                values.frombytes(content)
                if sys.byteorder != "little":
                    values.byteswap()
                if any(not math.isfinite(value) for value in values):
                    raise ValueError("Non-finite decoded audio")
                audio.append(values)
        if not videos[0] or len(videos[0]) != len(videos[1]):
            raise ValueError("Embedding parity video frame count/dimensions differ")
        video_error = sum(abs(a - b) for a, b in zip(*videos)) / (255 * len(videos[0]))
        row = {"sample_id": pair["sample_id"], "repeat": pair["repeat"], "video_mae": video_error}
        if audio:
            if len(audio) != 2 or len(audio[0]) != len(audio[1]) or not audio[0]:
                raise ValueError("Embedding parity audio streams/sample counts differ")
            row["audio_mae"] = sum(abs(a - b) for a, b in zip(*audio)) / len(audio[0])
        scores.append(row)
    metrics = {"video_mae_avg": sum(row["video_mae"] for row in scores) / len(scores),
               "video_mae_max": max(row["video_mae"] for row in scores)}
    if all("audio_mae" in row for row in scores):
        metrics.update(audio_mae_avg=sum(row["audio_mae"] for row in scores) / len(scores),
                       audio_mae_max=max(row["audio_mae"] for row in scores))
    result = {"complete": True, "metrics": metrics, "pairs": scores,
              "method": "decoded_media", "video_size": size, "audio_sample_rate": sample_rate,
              "note": "Video error uses scaled RGB in [0,1]; audio error uses decoded mono PCM. Not semantic similarity or verified encoder feature equality."}
    write_json(directory / "embedding_parity.json", result)
    return result
