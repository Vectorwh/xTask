# Copyright 2026 The xTask Authors.
"""Validate decoded media, not just HTTP success or a nonempty Base64 field."""
from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path


def validate_media(path: Path, generation, config):
    probe = config.get("ffprobe", "ffprobe")
    ffmpeg = config.get("ffmpeg", "ffmpeg")
    result = subprocess.run(
        [probe, "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=config["decode_timeout"],
    )
    metadata = json.loads(result.stdout)
    streams = metadata.get("streams") or []
    videos = [s for s in streams if s.get("codec_type") == "video"]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if len(videos) != 1:
        raise ValueError("Expected exactly one video stream")
    video = videos[0]
    width, height = [int(x) for x in generation["size"].split("*")]
    if video.get("width") != width or video.get("height") != height:
        raise ValueError("Decoded video dimensions differ from requested canvas")
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    if frames != generation["num_frames"]:
        raise ValueError(f"Decoded frame count {frames} != requested {generation['num_frames']}")
    fps = float(Fraction(video.get("avg_frame_rate") or "0"))
    if not math.isfinite(fps) or abs(fps - generation["fps"]) > config["fps_tolerance"]:
        raise ValueError("Decoded FPS differs from request")
    duration = float(video.get("duration") or metadata.get("format", {}).get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0 or abs(duration - frames / fps) > config["duration_tolerance_s"]:
        raise ValueError("Decoded video duration invalid or truncated")
    if config["audio"] == "required" and len(audios) != 1:
        raise ValueError("T2AV requires exactly one decoded audio stream")
    if config["audio"] == "optional" and audios and not generation.get("sampling_rate"):
        raise ValueError("Audio-enabled visual evaluation needs explicit generation.sampling_rate")
    if config["audio"] == "forbidden" and audios:
        raise ValueError("T2V returned audio: declare media.audio: optional for visual-only testing; do not claim audio-disabled performance")
    audio = None
    if audios:
        if len(audios) != 1:
            raise ValueError("Expected at most one audio stream")
        item = audios[0]
        sample_rate = int(item.get("sample_rate") or 0)
        if sample_rate <= 0 or (generation.get("sampling_rate") and sample_rate != generation["sampling_rate"]):
            raise ValueError("Audio sampling rate differs from explicit request")
        channels = int(item.get("channels") or 0)
        if channels <= 0 or (config.get("channels") and channels != config["channels"]):
            raise ValueError("Unexpected audio channel count")
        audio_duration = float(item.get("duration") or 0)
        if (not math.isfinite(audio_duration) or audio_duration <= 0
                or abs(audio_duration - duration) > config["duration_tolerance_s"]):
            raise ValueError("Audio/video duration mismatch")
        audio = {"sample_rate": sample_rate, "channels": channels, "duration_s": audio_duration}
    # Probe metadata alone can accept damaged/truncated media. Decode all streams.
    subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-map", "0:a?", "-f", "null", "-"],
        check=True, capture_output=True, timeout=config["decode_timeout"],
    )
    return {"width": width, "height": height, "frames": frames, "fps": fps,
            "duration_s": duration, "audio": audio, "fully_decoded": True}
